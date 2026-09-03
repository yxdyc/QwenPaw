#!/usr/bin/env python3
"""nano-agent-runtime L2 — distributed durable tool runtime.

K+1 over L1 (single-process durable runtime): the same prepare/commit protocol,
authority, and payload-bound idempotency are preserved, but the runtime is now
a small distributed system:

  * lease + fencing epoch   — a worker/dispatcher must acquire a lease before
                              touching an intent; every takeover increments a
                              durable epoch and the provider rejects stale owners.
  * outbox / inbox          — calling the provider is converted into a durable
                              outbox message; the result is recorded in inbox.
  * timeout / backoff       — failed dispatches retry with exponential backoff
                              up to a max; max exceeded -> NEEDS_HUMAN.
  * compensation            — a committed effect can be undone by a separately
                              authorized compensation intent with its own
                              idempotency key and durable plan.
  * control-plane binding   — principal/session come from trusted runtime
                              context, never from model output; model-selected
                              purpose is checked against a source-account scope.

The provider is still a separate subprocess with its own SQLite database, just
like L1.  All durability is SQLite/WAL; all concurrency is real multi-process
concurrency.  Only the Python standard library is used.

Run in an EMPTY directory:

    python3 -B L2_distributed_runtime.py

Modes (used internally by the orchestrator):
    provider apply <key> <fp> <src> <dst> <amount> <epoch> <crash-point>
    provider query <key>
    submit <intent-json>        enqueue an intent into the outbox
    dispatch-one <key> <owner> <crash-point>
    dispatcher <owner>          process all currently pending outbox items
    recover                     takeover dead/expired leases and finish orphans
    compensate <original-key>   schedule a compensation for a committed intent
    state                       canonical JSON of runtime + provider state
"""
from __future__ import annotations

import hashlib
import json
import os
import signal
import sqlite3
import subprocess
import sys
import time

RUNTIME_DB = "runtime.db"
PROVIDER_DB = "provider.db"
LEASE_TTL = 5.0          # seconds; demo uses short TTL so recovery is observable
MAX_ATTEMPTS = 5
BACKOFF_CAP = 30.0

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

START_BALANCES = {"research": 500, "vendor": 0, "attacker": 0}

# Trusted control plane.  Purpose scopes the allowed source account.
AUTHORITY = {
    "principal": "demo-agent",
    "session": "demo-session",
    "allowed_operations": ("transfer",),
    "max_amount": 30,
    "purposes": {
        "invoice-payment": {"allowed_source": "research"},
        "refund": {"allowed_source": ("research", "vendor")},
    },
}


def durable_now() -> float:
    """Cross-process demo clock for persisted deadlines.

    A process-local monotonic value must never be persisted and compared by
    another process.  Production systems should prefer database/server time
    plus a monotonically increasing fencing epoch; wall time is sufficient for
    this single-host teaching experiment.
    """
    return time.time()


def fingerprint(intent: dict) -> str:
    payload = json.dumps(intent, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def open_db(path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(path, timeout=10.0)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=FULL")
    conn.execute("PRAGMA busy_timeout=10000")
    return conn


def init_runtime(conn: sqlite3.Connection) -> None:
    conn.execute("""CREATE TABLE IF NOT EXISTS events (
        seq INTEGER PRIMARY KEY, type TEXT, key TEXT, data TEXT,
        prev_hash TEXT, hash TEXT)""")
    conn.execute("""CREATE TABLE IF NOT EXISTS commits (
        key TEXT PRIMARY KEY, fingerprint TEXT, receipt TEXT, method TEXT,
        principal TEXT, session TEXT, purpose TEXT, intent TEXT)""")
    conn.execute("""CREATE TABLE IF NOT EXISTS leases (
        scope TEXT PRIMARY KEY, intent_key TEXT, owner TEXT, acquired_at REAL,
        expires_at REAL, epoch INTEGER, status TEXT)""")
    conn.execute("""CREATE TABLE IF NOT EXISTS fence_counters (
        scope TEXT PRIMARY KEY, epoch INTEGER)""")
    conn.execute("""CREATE TABLE IF NOT EXISTS outbox (
        id INTEGER PRIMARY KEY, key TEXT UNIQUE, fingerprint TEXT, intent TEXT,
        principal TEXT, session TEXT, purpose TEXT, status TEXT,
        attempts INTEGER, next_attempt_at REAL, lease_owner TEXT, lease_epoch INTEGER,
        created_at REAL)""")
    conn.execute("""CREATE TABLE IF NOT EXISTS inbox (
        id INTEGER PRIMARY KEY, key TEXT, fingerprint TEXT, receipt TEXT,
        status TEXT, received_at REAL, UNIQUE(key, fingerprint, receipt))""")
    conn.execute("""CREATE TABLE IF NOT EXISTS compensations (
        id INTEGER PRIMARY KEY, original_key TEXT, comp_key TEXT UNIQUE,
        fingerprint TEXT, intent TEXT, status TEXT, receipt TEXT,
        attempts INTEGER, next_attempt_at REAL)""")


def init_provider(conn: sqlite3.Connection) -> None:
    conn.execute("CREATE TABLE IF NOT EXISTS balances (account TEXT PRIMARY KEY, amount INTEGER)")
    conn.execute("CREATE TABLE IF NOT EXISTS receipts (key TEXT PRIMARY KEY, fingerprint TEXT, receipt TEXT)")
    conn.execute("CREATE TABLE IF NOT EXISTS fences (scope TEXT PRIMARY KEY, max_epoch INTEGER)")
    if conn.execute("SELECT COUNT(*) FROM balances").fetchone()[0] == 0:
        conn.executemany("INSERT INTO balances VALUES (?, ?)", sorted(START_BALANCES.items()))
        conn.commit()


def append_event(
    conn: sqlite3.Connection, etype: str, key: str, data: dict, *, commit: bool = True
) -> None:
    row = conn.execute("SELECT hash FROM events ORDER BY seq DESC LIMIT 1").fetchone()
    prev = row[0] if row else "GENESIS"
    body = {"type": etype, "key": key, "data": data, "prev_hash": prev}
    h = hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:16]
    conn.execute("INSERT INTO events (type, key, data, prev_hash, hash) VALUES (?, ?, ?, ?, ?)",
                 (etype, key, json.dumps(data, sort_keys=True), prev, h))
    if commit:
        conn.commit()


def chain_ok(conn: sqlite3.Connection) -> tuple[bool, int]:
    previous, count = "GENESIS", 0
    for etype, key, data, prev_hash, h in conn.execute(
            "SELECT type, key, data, prev_hash, hash FROM events ORDER BY seq"):
        if prev_hash != previous:
            return False, count
        body = {"type": etype, "key": key, "data": json.loads(data), "prev_hash": prev_hash}
        if hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:16] != h:
            return False, count
        previous, count = h, count + 1
    return True, count


def authorize(intent: dict, auth: dict) -> tuple[bool, str]:
    if intent["operation"] not in auth["allowed_operations"]:
        return False, "operation is not allowed"
    purpose = intent.get("purpose")
    scope = auth.get("purposes", {}).get(purpose)
    if scope is None:
        return False, "purpose is not allowed"
    allowed_source = scope["allowed_source"]
    if isinstance(allowed_source, str):
        allowed_source = (allowed_source,)
    if intent["source"] not in allowed_source:
        return False, "source is outside purpose scope"
    if not 0 < intent["amount"] <= auth["max_amount"]:
        return False, "amount exceeds authorized limit"
    return True, "authorized"


def self_cmd(*args: str) -> list[str]:
    return [sys.executable, "-B", os.path.abspath(__file__), *args]


# ------------------------------------------------------------------ leases --

def acquire_lease(
    conn: sqlite3.Connection,
    key: str,
    owner: str,
    fence_scope: str,
    ttl: float = LEASE_TTL,
) -> int | None:
    """Acquire a lease and return an epoch monotonic within its resource scope."""
    now = durable_now()
    conn.execute("BEGIN IMMEDIATE")
    row = conn.execute(
        "SELECT expires_at FROM leases WHERE scope = ?", (fence_scope,)
    ).fetchone()
    if row is None or row[0] < now:
        prior = conn.execute(
            "SELECT epoch FROM fence_counters WHERE scope = ?", (fence_scope,)
        ).fetchone()
        epoch = (prior[0] if prior else 0) + 1
        conn.execute(
            "INSERT OR REPLACE INTO fence_counters (scope, epoch) VALUES (?, ?)",
            (fence_scope, epoch),
        )
        conn.execute(
            "INSERT OR REPLACE INTO leases "
            "(scope, intent_key, owner, acquired_at, expires_at, epoch, status) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (fence_scope, key, owner, now, now + ttl, epoch, "active"),
        )
        conn.commit()
        return epoch
    conn.commit()
    return None


def release_lease(
    conn: sqlite3.Connection, scope: str, owner: str, *, commit: bool = True
) -> None:
    conn.execute("DELETE FROM leases WHERE scope = ? AND owner = ?", (scope, owner))
    if commit:
        conn.commit()


def owner_alive(owner: str) -> bool:
    """Best-effort check whether the process that owns a lease still exists."""
    try:
        pid = int(owner.rsplit("-", 1)[-1])
        os.kill(pid, 0)
        return True
    except (ValueError, OSError, ProcessLookupError):
        return False


# ----------------------------------------------------------------- backoff --

def backoff_delay(attempts: int) -> float:
    return min(2 ** attempts, BACKOFF_CAP)


# --------------------------------------------------------------- provider --

def provider_apply(
    key: str, fp: str, src: str, dst: str, amount: int, epoch: int, crash_point: str
) -> int:
    conn = open_db(PROVIDER_DB)
    init_provider(conn)
    conn.execute("BEGIN IMMEDIATE")
    # The protected resource is the source account, not the idempotency key:
    # a stale owner must also be unable to write under a fresh request key.
    fence_scope = src
    fence = conn.execute(
        "SELECT max_epoch FROM fences WHERE scope = ?", (fence_scope,)
    ).fetchone()
    if fence and epoch < fence[0]:
        conn.commit()
        print("STALE_FENCE")
        return 3
    conn.execute(
        "INSERT OR REPLACE INTO fences (scope, max_epoch) VALUES (?, ?)",
        (fence_scope, epoch),
    )
    row = conn.execute("SELECT fingerprint, receipt FROM receipts WHERE key = ?", (key,)).fetchone()
    if row:
        conn.commit()
        print(row[1] if row[0] == fp else "FINGERPRINT_MISMATCH")
        return 0 if row[0] == fp else 2
    conn.execute("UPDATE balances SET amount = amount - ? WHERE account = ?", (amount, src))
    conn.execute("UPDATE balances SET amount = amount + ? WHERE account = ?", (amount, dst))
    receipt = f"prov-receipt-{conn.execute('SELECT COUNT(*) FROM receipts').fetchone()[0] + 1}"
    conn.execute("INSERT INTO receipts VALUES (?, ?, ?)", (key, fp, receipt))
    if crash_point == "before-commit":
        os.kill(os.getpid(), signal.SIGKILL)
    conn.commit()
    if crash_point == "after-commit":
        os.kill(os.getpid(), signal.SIGKILL)
    print(receipt)
    return 0


def provider_query(key: str) -> str:
    conn = open_db(PROVIDER_DB)
    init_provider(conn)
    row = conn.execute("SELECT receipt FROM receipts WHERE key = ?", (key,)).fetchone()
    return row[0] if row else "NOT_FOUND"


def provider_main(args: list[str]) -> int:
    if args[0] == "apply":
        return provider_apply(
            args[1], args[2], args[3], args[4], int(args[5]), int(args[6]), args[7]
        )
    if args[0] == "query":
        print(provider_query(args[1]))
        return 0
    raise SystemExit(f"unknown provider op: {args[0]}")


# ------------------------------------------------------------------ submit --

def submit_main(intent_json: str) -> int:
    intent = json.loads(intent_json)
    key, fp = intent["idempotency_key"], fingerprint(intent)
    purpose = intent.get("purpose")
    conn = open_db(RUNTIME_DB)
    init_runtime(conn)

    allowed, reason = authorize(intent, AUTHORITY)
    if not allowed:
        append_event(conn, "REJECTED", key, {"fingerprint": fp, "reason": reason})
        print(f"REJECTED {reason}")
        return 5

    conn.execute("BEGIN IMMEDIATE")
    existing = conn.execute(
        "SELECT fingerprint FROM commits WHERE key = ? UNION ALL "
        "SELECT fingerprint FROM outbox WHERE key = ? LIMIT 1",
        (key, key),
    ).fetchone()
    if existing:
        conn.commit()
        if existing[0] != fp:
            append_event(conn, "REJECTED", key, {
                "fingerprint": fp,
                "reason": "idempotency key reused with different payload",
            })
            print("REJECTED idempotency key reused with different payload")
            return 6
        print(f"SUBMITTED key={key}")
        return 0

    append_event(conn, "PREPARED", key, {
        "fingerprint": fp,
        "principal": AUTHORITY["principal"],
        "session": AUTHORITY["session"],
        "purpose": purpose,
        "intent": intent,
    }, commit=False)
    now = durable_now()
    conn.execute("""INSERT INTO outbox
        (key, fingerprint, intent, principal, session, purpose, status, attempts, next_attempt_at, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (key, fp, json.dumps(intent, sort_keys=True), AUTHORITY["principal"], AUTHORITY["session"],
         purpose, "PENDING", 0, now, now))
    conn.commit()
    print(f"SUBMITTED key={key}")
    return 0


# --------------------------------------------------------------- dispatch --

def dispatch_one(conn: sqlite3.Connection, key: str, owner: str, crash_point: str = "none") -> None:
    """Process a single outbox item.  The caller is expected to have acquired
    or attempted to acquire the lease; this function re-acquires defensively."""
    pending = conn.execute("SELECT intent FROM outbox WHERE key = ?", (key,)).fetchone()
    if pending is None:
        return
    fence_scope = json.loads(pending[0])["source"]
    epoch = acquire_lease(conn, key, owner, fence_scope)
    if epoch is None:
        return

    # A prior dispatch may have committed but crashed before marking outbox DONE.
    committed = conn.execute("SELECT fingerprint, receipt FROM commits WHERE key = ?", (key,)).fetchone()
    if committed:
        conn.execute(
            "UPDATE outbox SET status = 'DONE', lease_owner = NULL, lease_epoch = NULL "
            "WHERE key = ?",
            (key,),
        )
        release_lease(conn, fence_scope, owner)
        conn.commit()
        return

    row = conn.execute(
        "SELECT fingerprint, intent, attempts, principal, session, purpose FROM outbox "
        "WHERE key = ? AND status IN ('PENDING', 'LEASED')",
        (key,)).fetchone()
    if not row:
        release_lease(conn, fence_scope, owner)
        return

    fp, intent_json, attempts, principal, session, purpose = row
    intent = json.loads(intent_json)
    new_attempts = attempts + 1
    conn.execute(
        "UPDATE outbox SET status = 'LEASED', lease_owner = ?, lease_epoch = ?, attempts = ? "
        "WHERE key = ?",
        (owner, epoch, new_attempts, key),
    )
    conn.commit()

    provider_crash = "none"
    if crash_point == "provider:before-commit":
        provider_crash = "before-commit"
    elif crash_point == "provider:after-commit":
        provider_crash = "after-commit"

    try:
        proc = subprocess.run(
            self_cmd(
                "provider", "apply", key, fp, intent["source"], intent["destination"],
                str(intent["amount"]), str(epoch), provider_crash,
            ),
            capture_output=True,
            text=True,
            timeout=5,
        )
    except subprocess.TimeoutExpired:
        _schedule_retry_or_human(
            conn, key, owner, fence_scope, epoch, new_attempts, "timeout"
        )
        return

    out = proc.stdout.strip()
    if out == "FINGERPRINT_MISMATCH":
        conn.execute("BEGIN IMMEDIATE")
        append_event(conn, "REJECTED", key, {
            "fingerprint": fp, "reason": "provider fingerprint mismatch"
        }, commit=False)
        conn.execute(
            "UPDATE outbox SET status = 'REJECTED', lease_owner = NULL, lease_epoch = NULL "
            "WHERE key = ? AND lease_owner = ? AND lease_epoch = ?",
            (key, owner, epoch),
        )
        release_lease(conn, fence_scope, owner, commit=False)
        conn.commit()
        return

    if out == "STALE_FENCE":
        # A newer owner already reached the provider.  The stale worker must
        # not reset, commit, or release the winner's local state.
        conn.execute("BEGIN IMMEDIATE")
        append_event(
            conn,
            "STALE_OWNER_REJECTED",
            key,
            {"fingerprint": fp, "owner": owner, "epoch": epoch},
            commit=False,
        )
        conn.execute(
            "UPDATE outbox SET status = 'PENDING', lease_owner = NULL, lease_epoch = NULL "
            "WHERE key = ? AND lease_owner = ? AND lease_epoch = ?",
            (key, owner, epoch),
        )
        release_lease(conn, fence_scope, owner, commit=False)
        conn.commit()
        return

    if proc.returncode != 0 or not out:
        _schedule_retry_or_human(
            conn,
            key,
            owner,
            fence_scope,
            epoch,
            new_attempts,
            f"provider_rc={proc.returncode}",
        )
        return

    receipt = out
    if crash_point == "dispatcher:after-provider":
        os.kill(os.getpid(), signal.SIGKILL)

    # Commit row, event, inbox, outbox transition, and lease release share one
    # local transaction.  A stale owner is not allowed to commit locally even
    # if its provider response arrived after its lease was superseded.
    conn.execute("BEGIN IMMEDIATE")
    active = conn.execute(
        "SELECT 1 FROM leases WHERE scope = ? AND intent_key = ? AND owner = ? AND epoch = ?",
        (fence_scope, key, owner, epoch),
    ).fetchone()
    if active is None:
        conn.commit()
        return
    try:
        conn.execute("""INSERT INTO commits
            (key, fingerprint, receipt, method, principal, session, purpose, intent)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (key, fp, receipt, "outbox", principal, session, purpose, intent_json))
    except sqlite3.IntegrityError:
        winner = conn.execute("SELECT receipt FROM commits WHERE key = ?", (key,)).fetchone()
        receipt = winner[0] if winner else receipt
        append_event(
            conn, "COMMIT_OBSERVED", key,
            {"fingerprint": fp, "receipt": receipt}, commit=False,
        )
    else:
        append_event(
            conn, "COMMITTED", key,
            {"fingerprint": fp, "receipt": receipt, "method": "outbox"},
            commit=False,
        )

    conn.execute("INSERT OR IGNORE INTO inbox (key, fingerprint, receipt, status, received_at) "
                 "VALUES (?, ?, ?, ?, ?)", (key, fp, receipt, "OK", durable_now()))
    conn.execute(
        "UPDATE outbox SET status = 'DONE', lease_owner = NULL, lease_epoch = NULL "
        "WHERE key = ? AND lease_owner = ? AND lease_epoch = ?",
        (key, owner, epoch),
    )
    conn.execute("UPDATE compensations SET status = 'DONE' WHERE comp_key = ?", (key,))
    release_lease(conn, fence_scope, owner, commit=False)
    conn.commit()


def _schedule_retry_or_human(
    conn: sqlite3.Connection,
    key: str,
    owner: str,
    fence_scope: str,
    epoch: int,
    attempts: int,
    reason: str,
) -> None:
    conn.execute("BEGIN IMMEDIATE")
    if attempts >= MAX_ATTEMPTS:
        append_event(
            conn, "NEEDS_HUMAN", key,
            {"fingerprint": reason, "reason": f"max attempts exceeded ({reason})"},
            commit=False,
        )
        conn.execute(
            "UPDATE outbox SET status = 'NEEDS_HUMAN', lease_owner = NULL, lease_epoch = NULL "
            "WHERE key = ? AND lease_owner = ? AND lease_epoch = ?",
            (key, owner, epoch),
        )
    else:
        conn.execute(
            "UPDATE outbox SET status = 'PENDING', lease_owner = NULL, lease_epoch = NULL, "
            "attempts = ?, next_attempt_at = ? "
            "WHERE key = ? AND lease_owner = ? AND lease_epoch = ?",
            (attempts, durable_now() + backoff_delay(attempts), key, owner, epoch),
        )
    release_lease(conn, fence_scope, owner, commit=False)
    conn.commit()


def dispatcher_main(owner: str) -> int:
    conn = open_db(RUNTIME_DB)
    init_runtime(conn)
    owner_id = f"{owner}-{os.getpid()}"
    now = durable_now()
    pending = conn.execute(
        "SELECT key FROM outbox WHERE status = 'PENDING' AND next_attempt_at <= ? ORDER BY next_attempt_at",
        (now,)).fetchall()
    processed = 0
    for (key,) in pending:
        dispatch_one(conn, key, owner_id)
        processed += 1
    print(f"dispatcher owner={owner} processed={processed}")
    return 0


def dispatch_one_main(key: str, owner: str, crash_point: str) -> int:
    # The owner string must identify *this* process so that recovery can tell
    # whether the lease holder is still alive.  Append our PID to the caller's
    # base owner name; owner_alive() extracts the trailing PID with rsplit("-").
    owner = f"{owner}-{os.getpid()}"
    conn = open_db(RUNTIME_DB)
    init_runtime(conn)
    dispatch_one(conn, key, owner, crash_point)
    return 0


# --------------------------------------------------------------- recover --

def recover_main() -> int:
    conn = open_db(RUNTIME_DB)
    init_runtime(conn)
    ok, n = chain_ok(conn)
    print(f"recovery start: chain_ok={ok} events={n}")

    # Take over leases whose owner is dead or whose TTL has expired.
    now = durable_now()
    reset = 0
    for scope, key, owner, expires_at in conn.execute(
        "SELECT scope, intent_key, owner, expires_at FROM leases"
    ).fetchall():
        alive = owner_alive(owner)
        expired = expires_at < now
        if alive and not expired:
            continue
        print(f"lease takeover key={key} reason={'dead-owner' if not alive else 'expired'}")
        conn.execute(
            "UPDATE outbox SET status = 'PENDING', lease_owner = NULL, lease_epoch = NULL "
            "WHERE key = ? AND lease_owner = ?",
            (key, owner),
        )
        conn.execute("DELETE FROM leases WHERE scope = ?", (scope,))
        reset += 1
    conn.commit()

    # Process pending outbox items (including those we just reset).
    pending = conn.execute(
        "SELECT key FROM outbox WHERE status = 'PENDING' AND next_attempt_at <= ? ORDER BY next_attempt_at",
        (now,)).fetchall()
    processed = 0
    owner = f"recover-{os.getpid()}"
    for (key,) in pending:
        dispatch_one(conn, key, owner)
        processed += 1
    print(f"recovery complete: reset_leases={reset} processed={processed}")
    return 0


# ----------------------------------------------------------- compensation --

def compensation_intent(original: dict) -> dict:
    return {
        "operation": "transfer",
        "source": original["destination"],
        "destination": original["source"],
        "amount": original["amount"],
        "purpose": "refund",
        "idempotency_key": f"comp-{original['idempotency_key']}",
    }


def compensate_main(original_key: str) -> int:
    conn = open_db(RUNTIME_DB)
    init_runtime(conn)
    row = conn.execute("SELECT intent, receipt FROM commits WHERE key = ?", (original_key,)).fetchone()
    if not row:
        print(f"COMPENSATION_REJECTED key={original_key} not committed")
        return 7
    original_intent = json.loads(row[0])
    comp = compensation_intent(original_intent)
    comp_fp = fingerprint(comp)

    # Compensation must itself pass authorization (with refund purpose scope).
    allowed, reason = authorize(comp, AUTHORITY)
    if not allowed:
        append_event(conn, "COMPENSATION_REJECTED", original_key,
                     {"comp_key": comp["idempotency_key"], "fingerprint": comp_fp, "reason": reason})
        print(f"COMPENSATION_REJECTED {reason}")
        return 5

    comp_key = comp["idempotency_key"]
    comp_json = json.dumps(comp, sort_keys=True)
    conn.execute("BEGIN IMMEDIATE")
    existing = conn.execute(
        "SELECT fingerprint FROM compensations WHERE comp_key = ?", (comp_key,)
    ).fetchone()
    if existing:
        conn.commit()
        print(f"COMPENSATION_SCHEDULED key={original_key} comp_key={comp_key}")
        return 0

    now = durable_now()
    conn.execute(
        "INSERT INTO compensations (original_key, comp_key, fingerprint, intent, status, "
        "attempts, next_attempt_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (original_key, comp_key, comp_fp, comp_json, "PENDING", 0, now),
    )
    append_event(conn, "PREPARED", comp_key, {
        "fingerprint": comp_fp,
        "principal": AUTHORITY["principal"],
        "session": AUTHORITY["session"],
        "purpose": comp["purpose"],
        "intent": comp,
    }, commit=False)
    conn.execute("""INSERT INTO outbox
        (key, fingerprint, intent, principal, session, purpose, status, attempts, next_attempt_at, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (comp_key, comp_fp, comp_json, AUTHORITY["principal"], AUTHORITY["session"],
         comp["purpose"], "PENDING", 0, now, now))
    append_event(conn, "COMPENSATION_SCHEDULED", original_key, {
        "comp_key": comp_key, "fingerprint": comp_fp
    }, commit=False)
    conn.commit()
    print(f"COMPENSATION_SCHEDULED key={original_key} comp_key={comp['idempotency_key']}")
    return 0


# ------------------------------------------------------------------ state --

def state_main() -> int:
    rt = open_db(RUNTIME_DB)
    init_runtime(rt)
    prov = open_db(PROVIDER_DB)
    init_provider(prov)

    ok, n = chain_ok(rt)
    balances = {a: b for a, b in prov.execute("SELECT account, amount FROM balances ORDER BY account")}
    commits = {k: {"receipt": r, "method": m, "principal": p, "session": s, "purpose": u}
               for k, r, m, p, s, u in rt.execute(
                   "SELECT key, receipt, method, principal, session, purpose FROM commits ORDER BY key")}
    outbox = {k: {"status": st, "attempts": a, "lease_owner": lo}
              for k, st, a, lo in rt.execute(
                  "SELECT key, status, attempts, lease_owner FROM outbox ORDER BY key")}
    leases = {scope: {"intent_key": key, "owner": owner, "expires_at": expires}
              for scope, key, owner, expires in rt.execute(
                  "SELECT scope, intent_key, owner, expires_at FROM leases ORDER BY scope")}
    compensations = {ck: {"original_key": ok, "status": st}
                     for ok, ck, st in rt.execute("SELECT original_key, comp_key, status FROM compensations ORDER BY comp_key")}

    print(json.dumps({
        "balances": balances,
        "commits": commits,
        "outbox": outbox,
        "leases": leases,
        "compensations": compensations,
        "chain_ok": ok,
        "events": n,
        "journal_mode": {"runtime": rt.execute("PRAGMA journal_mode").fetchone()[0],
                         "provider": prov.execute("PRAGMA journal_mode").fetchone()[0]},
        "integrity": {"runtime": rt.execute("PRAGMA integrity_check").fetchone()[0],
                      "provider": prov.execute("PRAGMA integrity_check").fetchone()[0]},
    }, sort_keys=True))
    return 0


# ------------------------------------------------------------ orchestrator --

def run_submit(intent: dict) -> subprocess.CompletedProcess:
    return subprocess.run(self_cmd("submit", json.dumps(intent, sort_keys=True)),
                          capture_output=True, text=True, timeout=30)


def run_dispatcher(owner: str) -> subprocess.CompletedProcess:
    return subprocess.run(self_cmd("dispatcher", owner), capture_output=True, text=True, timeout=60)


def run_dispatch_one(key: str, owner: str, crash_point: str = "none") -> subprocess.CompletedProcess:
    return subprocess.run(self_cmd("dispatch-one", key, owner, crash_point),
                          capture_output=True, text=True, timeout=30)


def run_recover() -> subprocess.CompletedProcess:
    return subprocess.run(self_cmd("recover"), capture_output=True, text=True, timeout=60)


def run_compensate(original_key: str) -> subprocess.CompletedProcess:
    return subprocess.run(self_cmd("compensate", original_key), capture_output=True, text=True, timeout=30)


def run_provider_apply(target: dict, epoch: int) -> subprocess.CompletedProcess:
    return subprocess.run(
        self_cmd(
            "provider",
            "apply",
            target["idempotency_key"],
            fingerprint(target),
            target["source"],
            target["destination"],
            str(target["amount"]),
            str(epoch),
            "none",
        ),
        capture_output=True,
        text=True,
        timeout=30,
    )


def read_state() -> dict:
    out = subprocess.run(self_cmd("state"), capture_output=True, text=True, timeout=30)
    return json.loads(out.stdout)


def intent(key: str, amount: int = 25, purpose: str = "invoice-payment",
           source: str = "research", destination: str = "vendor") -> dict:
    return {"operation": "transfer", "source": source, "destination": destination,
            "amount": amount, "purpose": purpose, "idempotency_key": key}


def wait_for_lease_release(scope: str, timeout: float = 10.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        conn = open_db(RUNTIME_DB)
        row = conn.execute("SELECT 1 FROM leases WHERE scope = ?", (scope,)).fetchone()
        conn.close()
        if row is None:
            return True
        time.sleep(0.05)
    return False


def wait_for_next_attempt(key: str, timeout: float = 10.0) -> bool:
    """Wait until an outbox item's next_attempt_at has elapsed."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        conn = open_db(RUNTIME_DB)
        row = conn.execute("SELECT next_attempt_at FROM outbox WHERE key = ?", (key,)).fetchone()
        conn.close()
        if row is None or row[0] is None or row[0] <= durable_now():
            return True
        time.sleep(0.05)
    return False


def main() -> None:
    if os.path.exists(RUNTIME_DB) or os.path.exists(PROVIDER_DB):
        raise SystemExit("run in an EMPTY directory: runtime.db/provider.db already exist")
    print("=" * 78)
    print("Agent runtime L2 — lease+fencing / outbox / backoff / compensation / control-plane binding")
    print("=" * 78)
    print("toy: stdlib only (sqlite3/subprocess) | provider = separate process")
    print("protocol inherits L1: typed intent -> authorize -> PREPARED -> provider -> COMMITTED")
    print("new in L2: outbox + lease/fencing + recovery + backoff + compensation + trusted context")
    metrics: dict = {}

    print("\n[1] Happy path: submit -> dispatcher -> commit exactly once")
    run_submit(intent("inv-1"))
    run_dispatcher("disp-1")
    st = read_state()
    print(f"    balances={st['balances']} outbox={st['outbox']}")
    metrics["s1_committed"] = "inv-1" in st["commits"]
    metrics["s1_outbox_done"] = st["outbox"].get("inv-1", {}).get("status") == "DONE"
    metrics["s1_context_bound"] = st["commits"].get("inv-1", {}) == {
        "receipt": "prov-receipt-1",
        "method": "outbox",
        "principal": AUTHORITY["principal"],
        "session": AUTHORITY["session"],
        "purpose": "invoice-payment",
    }

    print("\n[2] Dispatcher dies holding lease; recovery takes over and finishes")
    run_submit(intent("inv-2"))
    p = run_dispatch_one("inv-2", f"disp-dead-{os.getpid()}", "dispatcher:after-provider")
    print(f"    dispatcher killed rc={p.returncode} (SIGKILL={-signal.SIGKILL})")
    # Lease is held by a dead process; recovery must detect dead owner.
    r = run_recover()
    print(r.stdout)
    st = read_state()
    print(f"    balances={st['balances']} commits={st['commits'].get('inv-2', {})}")
    metrics["s2_dispatcher_killed"] = p.returncode == -signal.SIGKILL
    metrics["s2_recovered"] = "inv-2" in st["commits"]
    metrics["s2_no_duplicate_receipt"] = st["commits"].get("inv-2", {}).get("receipt") == "prov-receipt-2"

    print("\n[3] Provider transient failures: retry with exponential backoff, then succeed")
    run_submit(intent("inv-3"))
    # Simulate two provider failures by crashing provider before commit twice.
    run_dispatch_one("inv-3", "disp-3a", "provider:before-commit")
    run_dispatch_one("inv-3", "disp-3b", "provider:before-commit")
    st = read_state()
    attempts = st["outbox"].get("inv-3", {}).get("attempts", 0)
    print(f"    after 2 provider kills: attempts={attempts} status={st['outbox'].get('inv-3', {}).get('status')}")
    wait_for_next_attempt("inv-3")
    run_dispatcher("disp-3c")
    st = read_state()
    print(f"    after successful retry: attempts={st['outbox'].get('inv-3', {}).get('attempts')} "
          f"status={st['outbox'].get('inv-3', {}).get('status')} receipt={st['commits'].get('inv-3', {}).get('receipt')}")
    metrics["s3_backoff_attempts"] = attempts >= 2
    metrics["s3_eventually_committed"] = "inv-3" in st["commits"]

    print("\n[4] Concurrent duplicate submission: lease + outbox unique key dedupes")
    target = intent("inv-4")
    pa = subprocess.Popen(self_cmd("submit", json.dumps(target, sort_keys=True)), stdout=subprocess.PIPE, text=True)
    pb = subprocess.Popen(self_cmd("submit", json.dumps(target, sort_keys=True)), stdout=subprocess.PIPE, text=True)
    oa, ob = pa.communicate(timeout=30)[0].strip(), pb.communicate(timeout=30)[0].strip()
    print(f"    submit outputs: {oa!r} / {ob!r}")
    mismatch = run_submit(intent("inv-4", amount=26))
    print(f"    same key / changed payload: rc={mismatch.returncode} {mismatch.stdout.strip()}")
    run_dispatcher("disp-4")
    st = read_state()
    conn = open_db(PROVIDER_DB)
    init_provider(conn)
    prov_rows = conn.execute("SELECT COUNT(*) FROM receipts WHERE key = 'inv-4'").fetchone()[0]
    conn.close()
    print(f"    balances={st['balances']} provider_receipts_for_key={prov_rows}")
    metrics["s4_one_provider_receipt"] = prov_rows == 1
    metrics["s4_one_commit_row"] = "inv-4" in st["commits"]
    metrics["s4_payload_mismatch_rejected"] = mismatch.returncode == 6

    print("\n[5] Compensation: commit a transfer, then undo it with a separately authorized refund")
    pre_inv5_balances = read_state()["balances"]
    run_submit(intent("inv-5"))
    run_dispatcher("disp-5")
    st = read_state()
    before_balances = dict(st["balances"])
    run_compensate("inv-5")
    run_dispatcher("disp-5c")
    st = read_state()
    after_balances = st["balances"]
    print(f"    before_comp={before_balances} after_comp={after_balances} baseline_before_inv5={pre_inv5_balances}")
    print(f"    compensations={st['compensations']}")
    metrics["s5_original_committed"] = "inv-5" in st["commits"]
    metrics["s5_compensation_committed"] = st["outbox"].get("comp-inv-5", {}).get("status") == "DONE"
    metrics["s5_balances_restored"] = after_balances == pre_inv5_balances

    print("\n[6] Control-plane binding: model purpose is scoped; identity is runtime-owned")
    bad_purpose = intent("inv-6", purpose="espionage")
    p = run_submit(bad_purpose)
    bad_scope = intent("inv-7", source="attacker")
    q = run_submit(bad_scope)
    print(f"    bad purpose rc={p.returncode} {p.stdout.strip()}")
    print(f"    bad source rc={q.returncode} {q.stdout.strip()}")
    st = read_state()
    metrics["s6_bad_purpose_rejected"] = p.returncode == 5
    metrics["s6_bad_scope_rejected"] = q.returncode == 5
    metrics["s6_no_commits"] = "inv-6" not in st["commits"] and "inv-7" not in st["commits"]

    print("\n[7] Max attempts exceeded -> NEEDS_HUMAN (simulated by provider crash every time)")
    run_submit(intent("inv-8"))
    for _ in range(MAX_ATTEMPTS):
        run_dispatch_one("inv-8", "disp-8", "provider:before-commit")
    st = read_state()
    print(f"    outbox={st['outbox'].get('inv-8', {})}")
    metrics["s7_needs_human"] = st["outbox"].get("inv-8", {}).get("status") == "NEEDS_HUMAN"
    metrics["s7_no_commit"] = "inv-8" not in st["commits"]

    print("\n[8] Expired owner is rejected by a provider-side fencing epoch")
    target = intent("inv-9", amount=1)
    balances_before_fence = read_state()["balances"]
    run_submit(target)
    conn = open_db(RUNTIME_DB)
    init_runtime(conn)
    epoch_a = acquire_lease(conn, "inv-9", "stale-owner", target["source"])
    blocked_epoch = acquire_lease(
        conn, "inv-9-other", "competing-owner", target["source"]
    )
    conn.execute("UPDATE leases SET expires_at = 0 WHERE scope = ?", (target["source"],))
    conn.commit()
    epoch_b = acquire_lease(conn, "inv-9", "fresh-owner", target["source"])
    assert epoch_a is not None and epoch_b is not None
    fresh = run_provider_apply(target, epoch_b)
    # Use a different idempotency key for the stale request.  Rejection must
    # therefore come from the resource-scoped epoch, not key deduplication.
    stale_target = intent("inv-9-stale", amount=2)
    stale = run_provider_apply(stale_target, epoch_a)
    release_lease(conn, target["source"], "fresh-owner")
    conn.close()
    run_dispatcher("disp-9-reconcile")
    st = read_state()
    provider = open_db(PROVIDER_DB)
    init_provider(provider)
    fence_max = provider.execute(
        "SELECT max_epoch FROM fences WHERE scope = ?", (target["source"],)
    ).fetchone()[0]
    fence_receipts = provider.execute(
        "SELECT COUNT(*) FROM receipts WHERE key = 'inv-9'"
    ).fetchone()[0]
    stale_receipts = provider.execute(
        "SELECT COUNT(*) FROM receipts WHERE key = 'inv-9-stale'"
    ).fetchone()[0]
    provider.close()
    expected_balances = dict(balances_before_fence)
    expected_balances["research"] -= 1
    expected_balances["vendor"] += 1
    print(
        f"    epochs: stale={epoch_a} fresh={epoch_b} reconciled={fence_max}; "
        f"fresh_rc={fresh.returncode} stale_rc={stale.returncode} "
        f"stale_reply={stale.stdout.strip()} receipts={fence_receipts} "
        f"stale_key_receipts={stale_receipts} same_scope_blocked={blocked_epoch is None}"
    )
    print(f"    balances={st['balances']} outbox={st['outbox'].get('inv-9', {})}")
    metrics["s8_epoch_monotonic"] = epoch_a < epoch_b < fence_max
    metrics["s8_scope_exclusion"] = blocked_epoch is None
    metrics["s8_stale_rejected"] = (
        stale.returncode == 3
        and stale.stdout.strip() == "STALE_FENCE"
        and stale_receipts == 0
    )
    metrics["s8_one_effect"] = (
        fresh.returncode == 0
        and fence_receipts == 1
        and st["balances"] == expected_balances
        and st["commits"].get("inv-9", {}).get("receipt") == fresh.stdout.strip()
    )

    print("\n[9] Durability evidence + self-check")
    st = read_state()
    print(f"    journal_mode: runtime={st['journal_mode']['runtime']} provider={st['journal_mode']['provider']}")
    print(f"    integrity_check: runtime={st['integrity']['runtime']} provider={st['integrity']['provider']}")
    print(f"    event hash chain verifies={st['chain_ok']} ({st['events']} events)")
    print(f"    leases={st['leases']} outbox={st['outbox']}")

    checks = (
        (metrics["s1_committed"] and metrics["s1_outbox_done"] and metrics["s1_context_bound"],
         "[1] commit is DONE and bound to trusted principal/session/purpose context"),
        (metrics["s2_dispatcher_killed"] and metrics["s2_recovered"] and metrics["s2_no_duplicate_receipt"],
         "[2] dead dispatcher's lease taken over and intent committed exactly once"),
        (metrics["s3_backoff_attempts"] and metrics["s3_eventually_committed"],
         "[3] provider failures retried with backoff, then committed"),
        (metrics["s4_one_provider_receipt"] and metrics["s4_one_commit_row"]
         and metrics["s4_payload_mismatch_rejected"],
         "[4] duplicate payload deduped; same key with changed payload rejected"),
        (metrics["s5_original_committed"] and metrics["s5_compensation_committed"]
         and metrics["s5_balances_restored"],
         "[5] compensation reversed the committed transfer, balances restored"),
        (metrics["s6_bad_purpose_rejected"] and metrics["s6_bad_scope_rejected"] and metrics["s6_no_commits"],
         "[6] wrong purpose and out-of-scope source rejected without commits"),
        (metrics["s7_needs_human"] and metrics["s7_no_commit"],
         "[7] max attempts exceeded escalates to NEEDS_HUMAN and never commits"),
        (metrics["s8_epoch_monotonic"] and metrics["s8_scope_exclusion"]
         and metrics["s8_stale_rejected"] and metrics["s8_one_effect"],
         "[8] provider rejects stale fencing epoch; recovery converges to one effect"),
        (st["journal_mode"]["runtime"] == "wal" and st["journal_mode"]["provider"] == "wal",
         "[9] WAL enabled on both stores"),
        (st["integrity"]["runtime"] == "ok" and st["integrity"]["provider"] == "ok",
         "[9] integrity_check ok after all kills"),
        (st["chain_ok"], "[9] event hash chain verifies end-to-end"),
    )
    for ok, name in checks:
        print(f"    {'PASS' if ok else 'FAIL'} | {name}")
    failed = [name for ok, name in checks if not ok]
    if failed:
        raise AssertionError(f"self-check failed: {failed}")
    print(f"\nSELF-CHECK: {len(checks)}/{len(checks)} PASS")
    canonical_metrics = json.dumps(metrics, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(canonical_metrics.encode()).hexdigest()[:16]
    print(f"digest(sha256 of metrics) = {digest}")
    print("takeaway: leases coordinate workers; a monotonic provider-checked epoch fences stale "
          "owners; payload-bound idempotency makes retries converge to one effect. The outbox "
          "makes work durable, backoff absorbs transient failures, compensation undoes committed "
          "effects, and trusted context binds principal/session/purpose.")
    result = {
        "checks": {"passed": len(checks), "total": len(checks)},
        "digest": digest,
        "evidence_boundary": (
            "Real local processes and SQLite/WAL; toy single-host provider, wall-clock leases, "
            "provider-side fencing simulated in SQLite, and no cryptographic token verification "
            "or distributed fencing service."
        ),
        "metrics": metrics,
        "module": "nano_agent_runtime_l2",
        "schema_version": 1,
    }
    print("RESULT_JSON=" + json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    if mode == "provider":
        raise SystemExit(provider_main(sys.argv[2:]))
    elif mode == "submit":
        raise SystemExit(submit_main(sys.argv[2]))
    elif mode == "dispatcher":
        raise SystemExit(dispatcher_main(sys.argv[2]))
    elif mode == "dispatch-one":
        raise SystemExit(dispatch_one_main(sys.argv[2], sys.argv[3], sys.argv[4]))
    elif mode == "recover":
        raise SystemExit(recover_main())
    elif mode == "compensate":
        raise SystemExit(compensate_main(sys.argv[2]))
    elif mode == "state":
        raise SystemExit(state_main())
    elif mode == "":
        main()
    else:
        raise SystemExit(f"unknown mode: {mode}")
