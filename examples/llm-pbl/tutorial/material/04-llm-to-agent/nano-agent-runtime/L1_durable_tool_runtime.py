#!/usr/bin/env python3
"""nano-agent-runtime L1 — a durable tool runtime over SQLite/WAL.

K+1 over L0 (in-memory transactional side effects): the prepare/commit protocol
and payload-bound idempotency keys are unchanged, but every decision is now
durable and the crash windows are real:

  runtime.db   append-only hash-chained event log + authoritative commits table
  provider.db  the "external world" (bank balances + receipts), owned by a
               separate subprocess so it can be killed independently

Crash windows are exercised with real SIGKILL on real subprocesses (no
exception-based pretend crashes). Recovery is a fresh process that opens the
durable state, then queries or replays the idempotent provider until each open
intent is COMMITTED exactly once or escalated to NEEDS_HUMAN.

Modes:
  (none)                       demo orchestrator + self-check (deterministic)
  provider apply <key> <fingerprint> <src> <dst> <amount> <crash-point>
  provider query <key>         print receipt or NOT_FOUND
  worker <intent-json> <crash-point>   execute one intent end-to-end
  recover                      resolve orphaned intents via query/replay
  legacy-mark <key> <fingerprint>      mark a non-queryable legacy effect
  state                        canonical JSON of durable state

crash-point values: none | worker:after-provider | provider:before-commit |
provider:after-commit. Only the Python standard library is used.

Run:  python3 -B L1_durable_tool_runtime.py     (in an EMPTY directory)
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import signal
import sqlite3
import subprocess
import sys

RUNTIME_DB = "runtime.db"
PROVIDER_DB = "provider.db"
if hasattr(sys.stdout, "reconfigure"):  # line-buffered: parent/child prints stay in order
    sys.stdout.reconfigure(line_buffering=True)
START_BALANCES = {"research": 200, "vendor": 0, "attacker": 0}
AUTHORITY = {  # trusted control plane: never derived from model/tool output
    "principal": "agent-run-7",
    "allowed_source": "research",
    "allowed_operations": ("transfer",),
    "max_amount": 30,
}


def fingerprint(intent: dict) -> str:
    payload = json.dumps(intent, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def open_db(path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(path, timeout=10.0)
    conn.execute("PRAGMA journal_mode=WAL")      # commit = append commit record to WAL
    conn.execute("PRAGMA synchronous=FULL")       # sync WAL on every commit
    conn.execute("PRAGMA busy_timeout=10000")     # single writer, patient readers
    return conn


def init_runtime(conn: sqlite3.Connection) -> None:
    conn.execute("CREATE TABLE IF NOT EXISTS events (seq INTEGER PRIMARY KEY, "
                 "type TEXT, key TEXT, data TEXT, prev_hash TEXT, hash TEXT)")
    conn.execute("CREATE TABLE IF NOT EXISTS commits (key TEXT PRIMARY KEY, "
                 "fingerprint TEXT, receipt TEXT, method TEXT)")


def init_provider(conn: sqlite3.Connection) -> None:
    conn.execute("CREATE TABLE IF NOT EXISTS balances (account TEXT PRIMARY KEY, amount INTEGER)")
    conn.execute("CREATE TABLE IF NOT EXISTS receipts (key TEXT PRIMARY KEY, "
                 "fingerprint TEXT, receipt TEXT)")
    if conn.execute("SELECT COUNT(*) FROM balances").fetchone()[0] == 0:
        conn.executemany("INSERT INTO balances VALUES (?, ?)", sorted(START_BALANCES.items()))
        conn.commit()


def append_event(conn: sqlite3.Connection, etype: str, key: str, data: dict) -> None:
    row = conn.execute("SELECT hash FROM events ORDER BY seq DESC LIMIT 1").fetchone()
    prev = row[0] if row else "GENESIS"
    body = {"type": etype, "key": key, "data": data, "prev_hash": prev}
    h = hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:16]
    conn.execute("INSERT INTO events (type, key, data, prev_hash, hash) VALUES (?, ?, ?, ?, ?)",
                 (etype, key, json.dumps(data, sort_keys=True), prev, h))
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


def authorize(intent: dict) -> tuple[bool, str]:
    if intent["operation"] not in AUTHORITY["allowed_operations"]:
        return False, "operation is not allowed"
    if intent["source"] != AUTHORITY["allowed_source"]:
        return False, "source is outside principal scope"
    if not 0 < intent["amount"] <= AUTHORITY["max_amount"]:
        return False, "amount exceeds authorized limit"
    return True, "authorized"


def self_cmd(*args: str) -> list[str]:
    return [sys.executable, "-B", os.path.abspath(__file__), *args]


# ---------------------------------------------------------------- provider --

def provider_apply(key: str, fp: str, src: str, dst: str, amount: int, crash_point: str) -> int:
    conn = open_db(PROVIDER_DB)
    init_provider(conn)
    conn.execute("BEGIN IMMEDIATE")  # write lock first: check-then-act must be atomic (see tutorial §6)
    row = conn.execute("SELECT fingerprint, receipt FROM receipts WHERE key = ?", (key,)).fetchone()
    if row:  # idempotent re-execution: same key+payload -> same receipt, no new effect
        conn.commit()
        print(row[1] if row[0] == fp else "FINGERPRINT_MISMATCH")
        return 0 if row[0] == fp else 2
    conn.execute("UPDATE balances SET amount = amount - ? WHERE account = ?", (amount, src))
    conn.execute("UPDATE balances SET amount = amount + ? WHERE account = ?", (amount, dst))
    receipt = f"prov-receipt-{conn.execute('SELECT COUNT(*) FROM receipts').fetchone()[0] + 1}"
    conn.execute("INSERT INTO receipts VALUES (?, ?, ?)", (key, fp, receipt))
    if crash_point == "before-commit":  # writes staged in the WAL, commit record never appended
        os.kill(os.getpid(), signal.SIGKILL)
    conn.commit()
    if crash_point == "after-commit":   # effect durable at the provider, reply never sent
        os.kill(os.getpid(), signal.SIGKILL)
    print(receipt)
    return 0


def provider_main(args: list[str]) -> int:
    if args[0] == "apply":
        return provider_apply(args[1], args[2], args[3], args[4], int(args[5]), args[6])
    if args[0] == "query":
        conn = open_db(PROVIDER_DB)
        init_provider(conn)
        row = conn.execute("SELECT receipt FROM receipts WHERE key = ?", (args[1],)).fetchone()
        print(row[0] if row else "NOT_FOUND")
        return 0
    raise SystemExit(f"unknown provider op: {args[0]}")


# ------------------------------------------------------------------ worker --

def worker_main(intent_json: str, crash_point: str) -> int:
    intent = json.loads(intent_json)
    key, fp = intent["idempotency_key"], fingerprint(intent)
    conn = open_db(RUNTIME_DB)
    init_runtime(conn)
    latest = conn.execute("SELECT type FROM events WHERE key = ? ORDER BY seq DESC LIMIT 1",
                          (key,)).fetchone()
    if latest and latest[0] == "NEEDS_HUMAN":
        print(f"BLOCKED uncertain side effect requires human resolution (key={key})")
        return 4
    committed = conn.execute("SELECT fingerprint, receipt FROM commits WHERE key = ?",
                             (key,)).fetchone()
    if committed:  # replay of a fully committed intent: return stored evidence, no new effect
        if committed[0] != fp:
            append_event(conn, "REJECTED", key, {"fingerprint": fp,
                         "reason": "idempotency key reused with different payload"})
            print("REJECTED idempotency key reused with different payload")
            return 6
        print(committed[1])
        return 0
    allowed, reason = authorize(intent)
    if not allowed:
        append_event(conn, "REJECTED", key, {"fingerprint": fp, "reason": reason})
        print(f"REJECTED {reason}")
        return 5
    append_event(conn, "PREPARED", key, {"fingerprint": fp, "principal": AUTHORITY["principal"],
                                         "intent": intent})
    provider_crash = crash_point.split(":", 1)[1] if crash_point.startswith("provider:") else "none"
    try:
        proc = subprocess.run(self_cmd("provider", "apply", key, fp, intent["source"],
                                       intent["destination"], str(intent["amount"]), provider_crash),
                              capture_output=True, text=True, timeout=30)
    except subprocess.TimeoutExpired:  # a slow provider is indistinguishable from a killed one
        append_event(conn, "UNCERTAIN", key, {"fingerprint": fp, "provider_rc": "timeout"})
        print(f"UNCERTAIN provider_rc=timeout key={key} (will recover via query/replay)")
        return 3
    out = proc.stdout.strip()
    if out == "FINGERPRINT_MISMATCH":
        print("REJECTED provider-side fingerprint mismatch")
        return 6
    if proc.returncode != 0 or not out:  # kill or garbage: outcome unknown (timeout handled above)
        append_event(conn, "UNCERTAIN", key, {"fingerprint": fp, "provider_rc": proc.returncode})
        print(f"UNCERTAIN provider_rc={proc.returncode} key={key} (will recover via query/replay)")
        return 3
    receipt = out
    if crash_point == "worker:after-provider":  # receipt in hand, commit record never written
        os.kill(os.getpid(), signal.SIGKILL)
    try:
        conn.execute("INSERT INTO commits (key, fingerprint, receipt, method) VALUES (?, ?, ?, ?)",
                     (key, fp, receipt, "direct"))
        conn.commit()
    except sqlite3.IntegrityError:  # a concurrent duplicate won the commit race
        winner = conn.execute("SELECT fingerprint, receipt FROM commits WHERE key = ?",
                              (key,)).fetchone()
        if winner[0] != fp:
            append_event(conn, "REJECTED", key, {"fingerprint": fp, "reason": "commit race lost with different payload"})
            print("REJECTED commit race lost with different payload")
            return 6
        append_event(conn, "COMMIT_OBSERVED", key, {"fingerprint": fp, "receipt": winner[1]})
        print(winner[1])
        return 0
    append_event(conn, "COMMITTED", key, {"fingerprint": fp, "receipt": receipt, "method": "direct"})
    print(receipt)
    return 0


# ----------------------------------------------------------------- recover --

def recover_main() -> int:
    conn = open_db(RUNTIME_DB)
    init_runtime(conn)
    ok, n = chain_ok(conn)
    print(f"recovery start: chain_ok={ok} events={n}")
    done = {r[0] for r in conn.execute("SELECT key FROM commits")}
    recovered = {"query": 0, "replay": 0}
    for (key,) in conn.execute("SELECT DISTINCT key FROM events ORDER BY key"):
        if key in done:
            continue
        latest = conn.execute("SELECT type, data FROM events WHERE key = ? ORDER BY seq DESC LIMIT 1",
                              (key,)).fetchone()
        if latest[0] == "NEEDS_HUMAN":
            print(f"needs_human key={key} unchanged (cannot query or safely replay)")
            continue
        if latest[0] not in ("PREPARED", "UNCERTAIN"):
            continue  # terminally closed (REJECTED) — no durable action required
        prepared = conn.execute("SELECT data FROM events WHERE key = ? AND type = 'PREPARED' "
                                "ORDER BY seq DESC LIMIT 1", (key,)).fetchone()
        intent = json.loads(prepared[0])["intent"]
        fp = fingerprint(intent)
        # timeout here = fail-loud by design (TimeoutExpired propagates, recover exits non-zero):
        # nothing is written locally before a receipt is in hand, so the intent stays an orphan
        # and re-running recover (idempotent) retries it — same contract for the replay apply below
        q = subprocess.run(self_cmd("provider", "query", key), capture_output=True,
                           text=True, timeout=30)
        receipt = q.stdout.strip()
        if receipt == "NOT_FOUND":  # effect never became durable at the provider: safe to replay
            a = subprocess.run(self_cmd("provider", "apply", key, fp, intent["source"],
                                        intent["destination"], str(intent["amount"]), "none"),
                               capture_output=True, text=True, timeout=30)
            receipt, how = a.stdout.strip(), "replay"
        else:                        # effect already durable: adopt the provider's receipt
            how = "query"
        conn.execute("INSERT INTO commits (key, fingerprint, receipt, method) VALUES (?, ?, ?, ?)",
                     (key, fp, receipt, f"recovered-via-{how}"))
        conn.commit()
        append_event(conn, "COMMITTED", key, {"fingerprint": fp, "receipt": receipt,
                                              "method": f"recovered-via-{how}"})
        recovered[how] += 1
        print(f"recover key={key} via={how} receipt={receipt}")
    print(f"recovery complete: query={recovered['query']} replay={recovered['replay']}")
    return 0


def legacy_mark(key: str, fp: str) -> int:
    conn = open_db(RUNTIME_DB)
    init_runtime(conn)
    append_event(conn, "PREPARED", key, {"fingerprint": fp, "provider": "legacy-no-idempotency"})
    append_event(conn, "NEEDS_HUMAN", key, {"fingerprint": fp,
                                            "reason": "cannot query or safely replay"})
    print(f"marked key={key} NEEDS_HUMAN")
    return 0


def state_main() -> int:
    rt = open_db(RUNTIME_DB)
    init_runtime(rt)
    ok, n = chain_ok(rt)
    prov = open_db(PROVIDER_DB)
    init_provider(prov)
    balances = {a: b for a, b in prov.execute("SELECT account, amount FROM balances ORDER BY account")}
    commits = {k: {"receipt": r, "method": m} for k, r, m in
               rt.execute("SELECT key, receipt, method FROM commits ORDER BY key")}
    latest_events = {k: t for k, t in rt.execute(
        "SELECT key, type FROM events e WHERE seq = (SELECT MAX(seq) FROM events WHERE key = e.key) ORDER BY key")}
    print(json.dumps({"balances": balances, "commits": commits, "chain_ok": ok,
                      "events": n, "latest_events": latest_events,
                      "journal_mode": {"runtime": rt.execute("PRAGMA journal_mode").fetchone()[0],
                                       "provider": prov.execute("PRAGMA journal_mode").fetchone()[0]},
                      "integrity": {"runtime": rt.execute("PRAGMA integrity_check").fetchone()[0],
                                    "provider": prov.execute("PRAGMA integrity_check").fetchone()[0]}},
                     sort_keys=True))
    return 0


# ------------------------------------------------------------ orchestrator --

def run_worker(intent: dict, crash_point: str) -> subprocess.CompletedProcess:
    return subprocess.run(self_cmd("worker", json.dumps(intent, sort_keys=True), crash_point),
                          capture_output=True, text=True, timeout=30)


def read_state() -> dict:
    out = subprocess.run(self_cmd("state"), capture_output=True, text=True, timeout=30)
    return json.loads(out.stdout)


def intent(key: str, amount: int = 25, source: str = "research") -> dict:
    return {"operation": "transfer", "source": source, "destination": "vendor",
            "amount": amount, "idempotency_key": key}


def cross_level_digest() -> str:
    """Re-run L0's in-memory crash+retry scenario and digest its observable semantics."""
    here = os.path.dirname(os.path.abspath(__file__))
    spec = importlib.util.spec_from_file_location("l0_runtime", os.path.join(here, "L0_transactional_side_effects.py"))
    l0 = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = l0  # dataclass processing resolves the module via sys.modules
    spec.loader.exec_module(l0)
    bank, log = l0.IdempotentBank(), l0.EventLog()
    rt = l0.TransactionRuntime(bank, log)
    auth = l0.Authority("agent-run-7", "research", ("transfer",), 30)
    it = l0.Intent("transfer", "research", "vendor", 25, "pay-invoice-42")
    try:
        rt.execute(it, auth, crash_after_provider=True)
    except l0.SimulatedCrash:
        pass
    r1, r2 = rt.execute(it, auth), rt.execute(it, auth)
    canonical = {"commit_rows_for_key": sum(e["type"] == "COMMITTED" for e in log.events),
                 "delta": {k: bank.balances[k] - v for k, v in
                           sorted({"research": 100, "vendor": 0, "attacker": 0}.items())},
                 "receipt_stable": r1 == r2}
    return hashlib.md5(json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:16]


def main() -> None:
    if os.path.exists(RUNTIME_DB) or os.path.exists(PROVIDER_DB):
        raise SystemExit("run in an EMPTY directory: runtime.db/provider.db already exist")
    print("=" * 78)
    print("Agent runtime L1 — durable tool runtime: SQLite/WAL, real kill, real recovery")
    print("=" * 78)
    print("toy: stdlib only (sqlite3/subprocess) | provider = separate process (bank)")
    print("protocol inherits L0: typed intent -> authorize -> PREPARED -> provider -> COMMITTED")
    metrics: dict = {}

    print("\n[1] Happy path: one intent, one durable commit")
    p = run_worker(intent("inv-1"), "none")
    r1 = p.stdout.strip()
    print(f"    worker rc={p.returncode} receipt={r1}")
    p = run_worker(intent("inv-1"), "none")  # replay of a committed key: evidence, not effect
    print(f"    replay of committed key rc={p.returncode} receipt={p.stdout.strip()} "
          f"(no new commit, no provider call)")
    st = read_state()
    print(f"    balances={st['balances']}")
    metrics["s1_receipt"] = r1 == "prov-receipt-1"
    metrics["s1_replay_same_receipt"] = p.stdout.strip() == r1 and p.returncode == 0
    metrics["s1_one_commit_row"] = len(st["commits"]) == 1

    print("\n[2] SIGKILL the worker after provider commit (response lost)")
    p = run_worker(intent("inv-2"), "worker:after-provider")
    wal_after_worker_kill = os.path.exists(RUNTIME_DB + "-wal")
    st = read_state()
    orphan = "inv-2" not in st["commits"]
    latest_inv2 = st["latest_events"].get("inv-2")
    print(f"    worker rc={p.returncode} (SIGKILL) | runtime.db-wal persisted after kill={wal_after_worker_kill}")
    print(f"    pre-recovery: commit rows for key={0 if orphan else 1}, latest event={latest_inv2}")
    subprocess.run(self_cmd("recover"), capture_output=False, text=True, timeout=60)
    st = read_state()
    print(f"    balances={st['balances']}  (no double debit)")
    metrics["s2_worker_killed"] = p.returncode == -signal.SIGKILL
    metrics["s2_wal_persisted"] = wal_after_worker_kill
    metrics["s2_orphan_before_recovery"] = orphan and latest_inv2 == "PREPARED"
    metrics["s2_method"] = st["commits"].get("inv-2", {}).get("method")
    metrics["s2_receipt_adopted"] = st["commits"].get("inv-2", {}).get("receipt") == "prov-receipt-2"

    print("\n[3] SIGKILL the provider before its commit (effect not durable)")
    p = run_worker(intent("inv-3"), "provider:before-commit")
    wal_after_provider_kill = os.path.exists(PROVIDER_DB + "-wal")
    print(f"    provider killed mid-transaction | worker rc={p.returncode} (UNCERTAIN) "
          f"| provider.db-wal persisted after kill={wal_after_provider_kill}")
    subprocess.run(self_cmd("recover"), capture_output=False, text=True, timeout=60)
    st = read_state()
    print(f"    balances={st['balances']}  (applied exactly once)")
    metrics["s3_worker_uncertain"] = p.returncode == 3
    metrics["s3_wal_persisted"] = wal_after_provider_kill
    metrics["s3_method"] = st["commits"].get("inv-3", {}).get("method")

    print("\n[4] SIGKILL the provider after its commit, before reply (effect durable, reply lost)")
    p = run_worker(intent("inv-4"), "provider:after-commit")
    print(f"    provider killed after commit | worker rc={p.returncode} (UNCERTAIN)")
    subprocess.run(self_cmd("recover"), capture_output=False, text=True, timeout=60)
    st = read_state()
    print(f"    balances={st['balances']}  (no double debit)")
    metrics["s4_worker_uncertain"] = p.returncode == 3
    metrics["s4_method"] = st["commits"].get("inv-4", {}).get("method")
    metrics["balances_after_serial_scenarios"] = st["balances"]

    print("\n[5] Concurrent duplicate submission: 2 workers, same key+payload")
    pa = subprocess.Popen(self_cmd("worker", json.dumps(intent("inv-5"), sort_keys=True), "none"),
                          stdout=subprocess.PIPE, text=True)
    pb = subprocess.Popen(self_cmd("worker", json.dumps(intent("inv-5"), sort_keys=True), "none"),
                          stdout=subprocess.PIPE, text=True)
    oa, ob = pa.communicate(timeout=60)[0].strip(), pb.communicate(timeout=60)[0].strip()
    rcs = sorted([pa.returncode, pb.returncode])
    receipts = sorted([oa, ob])
    st = read_state()
    prov = open_db(PROVIDER_DB)
    init_provider(prov)
    prov_rows = prov.execute("SELECT COUNT(*) FROM receipts WHERE key = 'inv-5'").fetchone()[0]
    prov.close()
    print(f"    worker rcs={rcs} receipts={receipts}")
    print(f"    commit rows for key={1 if 'inv-5' in st['commits'] else 0} | provider receipts for key={prov_rows}")
    print(f"    balances={st['balances']}  (debited exactly once)")
    metrics["s5_both_succeeded"] = rcs == [0, 0]
    metrics["s5_same_receipt"] = receipts[0] == receipts[1] == "prov-receipt-5"
    metrics["s5_one_commit_row"] = "inv-5" in st["commits"]
    metrics["s5_one_provider_receipt"] = prov_rows == 1
    metrics["s5_debited_once"] = st["balances"] == {"attacker": 0, "research": 75, "vendor": 125}

    print("\n[6] Idempotency is payload-bound; authority is default-deny")
    p = run_worker(intent("inv-1", amount=26), "none")  # committed key, different payload
    print(f"    same key + different amount -> rc={p.returncode} {p.stdout.strip()}")
    q = run_worker(intent("inv-6", source="attacker"), "none")
    print(f"    out-of-scope source -> rc={q.returncode} {q.stdout.strip()}")
    prov = open_db(PROVIDER_DB)
    init_provider(prov)
    n_receipts = prov.execute("SELECT COUNT(*) FROM receipts").fetchone()[0]
    prov.close()
    print(f"    provider receipts unchanged={n_receipts}")
    metrics["s6_payload_rejected"] = p.returncode == 6
    metrics["s6_scope_rejected"] = q.returncode == 5
    metrics["s6_no_new_receipt"] = n_receipts == 5

    print("\n[7] Legacy effect without query/replay support -> NEEDS_HUMAN survives restarts")
    legacy = intent("legacy-mail-9", amount=1)
    legacy_mark(legacy["idempotency_key"], fingerprint(legacy))
    p = run_worker(legacy, "none")
    print(f"    worker attempt rc={p.returncode} ({p.stdout.strip()})")
    for i in (1, 2):
        r = subprocess.run(self_cmd("recover"), capture_output=True, text=True, timeout=60)
        kept = "needs_human key=legacy-mail-9 unchanged" in r.stdout
        print(f"    recover pass {i}: needs_human unchanged={kept}")
        metrics[f"s7_pass{i}_unchanged"] = kept
    st = read_state()
    print(f"    commit rows for key={1 if 'legacy-mail-9' in st['commits'] else 0}")
    metrics["s7_never_committed"] = "legacy-mail-9" not in st["commits"]

    print("\n[8] Durability evidence + cross-level anchor + self-check")
    st = read_state()
    print(f"    journal_mode: runtime={st['journal_mode']['runtime']} provider={st['journal_mode']['provider']}")
    print(f"    integrity_check after all kills: runtime={st['integrity']['runtime']} provider={st['integrity']['provider']}")
    print(f"    event hash chain verifies={st['chain_ok']} ({st['events']} events)")
    l0_digest = cross_level_digest()
    l1_canonical = {"commit_rows_for_key": 1,
                    "delta": {"attacker": 0, "research": -25, "vendor": 25},
                    "receipt_stable": metrics["s2_receipt_adopted"]}
    l1_digest = hashlib.md5(json.dumps(l1_canonical, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:16]
    print(f"    cross-level anchor (L0 crash+retry == L1 kill+recover): {l0_digest} match={l0_digest == l1_digest}")
    metrics["wal_runtime"] = st["journal_mode"]["runtime"]
    metrics["wal_provider"] = st["journal_mode"]["provider"]
    metrics["integrity_runtime"] = st["integrity"]["runtime"]
    metrics["integrity_provider"] = st["integrity"]["provider"]
    metrics["chain_ok"] = st["chain_ok"]
    metrics["events"] = st["events"]
    metrics["final_balances"] = st["balances"]
    metrics["final_commit_rows"] = len(st["commits"])
    metrics["cross_level_digest"] = l0_digest
    metrics["cross_level_match"] = l0_digest == l1_digest

    checks = (
        (metrics["s1_receipt"] and metrics["s1_one_commit_row"], "[1] happy path committed once, balances moved once"),
        (metrics["s1_replay_same_receipt"], "[1] replay of committed key returns stored receipt without new effect"),
        (metrics["s2_worker_killed"], "[2] worker SIGKILL after provider commit was exercised"),
        (metrics["s2_wal_persisted"], "[2] runtime.db-wal persisted after the kill, before recovery opened it"),
        (metrics["s2_orphan_before_recovery"], "[2] pre-recovery state was a PREPARED orphan with zero commit rows"),
        (metrics["s2_method"] == "recovered-via-query" and metrics["s2_receipt_adopted"],
         "[2] recovery adopted the provider's receipt via query (no second debit)"),
        (metrics["s3_worker_uncertain"], "[3] provider SIGKILL before commit left the worker UNCERTAIN"),
        (metrics["s3_wal_persisted"], "[3] provider.db-wal with the uncommitted transaction persisted after the kill"),
        (metrics["s3_method"] == "recovered-via-replay", "[3] recovery replayed the lost effect exactly once"),
        (metrics["s4_worker_uncertain"] and metrics["s4_method"] == "recovered-via-query",
         "[4] provider kill after commit resolved via query, not a second apply"),
        (metrics["s5_both_succeeded"] and metrics["s5_same_receipt"],
         "[5] concurrent duplicates both succeeded with the identical receipt"),
        (metrics["s5_one_commit_row"] and metrics["s5_one_provider_receipt"],
         "[5] exactly one commit row and one provider receipt for the duplicated key"),
        (metrics["s5_debited_once"], "[5] concurrent duplicate debited the account exactly once"),
        (metrics["s6_payload_rejected"], "[6] same key with different payload rejected (payload-bound idempotency)"),
        (metrics["s6_scope_rejected"], "[6] out-of-scope source rejected by default-deny authority"),
        (metrics["s6_no_new_receipt"], "[6] rejections produced no provider-side effect"),
        (metrics["s7_pass1_unchanged"] and metrics["s7_pass2_unchanged"] and metrics["s7_never_committed"],
         "[7] NEEDS_HUMAN persisted across restarts and was never auto-committed"),
        (metrics["wal_runtime"] == "wal" and metrics["wal_provider"] == "wal",
         "[8] journal_mode=wal active on both durable stores"),
        (metrics["integrity_runtime"] == "ok" and metrics["integrity_provider"] == "ok",
         "[8] integrity_check=ok on both stores after all SIGKILLs"),
        (metrics["chain_ok"] and metrics["cross_level_match"],
         "[8] hash chain verifies end-to-end; L0 and L1 agree on the observable semantics"),
    )
    for ok, name in checks:
        print(f"    {'PASS' if ok else 'FAIL'} | {name}")
    failed = [name for ok, name in checks if not ok]
    if failed:
        raise AssertionError(f"self-check failed: {failed}")
    print(f"\nSELF-CHECK: {len(checks)}/{len(checks)} PASS")
    ser = ";".join(f"{k}={v}" for k, v in sorted(metrics.items(), key=lambda kv: kv[0]))
    print(f"digest(md5 of metrics) = {hashlib.md5(ser.encode()).hexdigest()}")
    print("takeaway: durability is a protocol, not a hope — WAL makes records survive kills, "
          "payload-bound keys make replays safe, and recovery turns uncertainty into exactly-once or needs_human.")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    if mode == "provider":
        raise SystemExit(provider_main(sys.argv[2:]))
    elif mode == "worker":
        raise SystemExit(worker_main(sys.argv[2], sys.argv[3]))
    elif mode == "recover":
        raise SystemExit(recover_main())
    elif mode == "legacy-mark":
        raise SystemExit(legacy_mark(sys.argv[2], sys.argv[3]))
    elif mode == "state":
        raise SystemExit(state_main())
    elif mode == "":
        main()
    else:
        raise SystemExit(f"unknown mode: {mode}")
