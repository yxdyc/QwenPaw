#!/usr/bin/env python3
"""L3a: durable outbox delivery to an external, generation-guarded router."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json
from pathlib import Path
import sqlite3
import tempfile
from typing import Any


def canonical(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def digest(payload: Any) -> str:
    return hashlib.sha256(canonical(payload).encode()).hexdigest()


@dataclass(frozen=True)
class RouteCommand:
    command_id: str
    idempotency_key: str
    decision_id: str
    kind: str
    from_model: str
    to_model: str
    expected_generation: int
    payload_hash: str


@dataclass(frozen=True)
class RouteReceipt:
    receipt_id: str
    command_id: str
    idempotency_key: str
    payload_hash: str
    from_model: str
    to_model: str
    applied_generation: int


@dataclass(frozen=True)
class LeaseClaim:
    command_id: str
    worker_id: str
    fencing_token: int
    lease_until: int


def make_command(
    decision_id: str, kind: str, from_model: str, to_model: str,
    expected_generation: int,
) -> RouteCommand:
    if kind not in ("ACTIVATE", "ROLLBACK"):
        raise ValueError("route command must be ACTIVATE or ROLLBACK")
    if not decision_id or not from_model or not to_model or from_model == to_model:
        raise ValueError("route command identity is incomplete")
    if expected_generation < 0:
        raise ValueError("expected generation must be non-negative")
    key = f"{kind.lower()}:{decision_id}"
    core = {
        "idempotency_key": key,
        "decision_id": decision_id,
        "kind": kind,
        "from_model": from_model,
        "to_model": to_model,
        "expected_generation": expected_generation,
    }
    payload_hash = digest(core)
    return RouteCommand(
        command_id=f"route-{payload_hash[:12]}",
        payload_hash=payload_hash,
        **core,
    )


def validate_command(command: RouteCommand) -> None:
    expected = make_command(
        command.decision_id, command.kind, command.from_model,
        command.to_model, command.expected_generation,
    )
    if command != expected:
        raise RuntimeError("route command identity does not recompute from its payload")


def restore_command(payload_json: str) -> RouteCommand:
    return RouteCommand(**json.loads(payload_json))


def restore_receipt(payload_json: str) -> RouteReceipt:
    return RouteReceipt(**json.loads(payload_json))


class ExternalRouter:
    """A separate authority that owns the serving route and idempotency receipts."""

    def __init__(self, path: Path, initial_model: str) -> None:
        self.path = path
        self.conn = sqlite3.connect(path)
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.conn.execute("PRAGMA journal_mode = WAL")
        self._create_schema()
        if self.conn.execute("SELECT 1 FROM router_state").fetchone() is None:
            self._initialize(initial_model)

    def _create_schema(self) -> None:
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS router_state(
                singleton INTEGER PRIMARY KEY CHECK(singleton=1),
                active_model TEXT NOT NULL,
                generation INTEGER NOT NULL CHECK(generation >= 0)
            );
            CREATE TABLE IF NOT EXISTS receipts(
                idempotency_key TEXT PRIMARY KEY,
                payload_hash TEXT NOT NULL CHECK(length(payload_hash)=64),
                receipt_json TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS router_events(
                seq INTEGER PRIMARY KEY,
                kind TEXT NOT NULL CHECK(kind IN ('INITIALIZE','APPLY','ADMIN_OVERRIDE')),
                payload_json TEXT NOT NULL,
                prev_hash TEXT NOT NULL,
                event_hash TEXT NOT NULL UNIQUE
            );
            """
        )
        for table in ("receipts", "router_events"):
            for action in ("UPDATE", "DELETE"):
                self.conn.execute(
                    f"CREATE TRIGGER IF NOT EXISTS no_{action.lower()}_{table} "
                    f"BEFORE {action} ON {table} BEGIN "
                    f"SELECT RAISE(ABORT, '{table} is append-only'); END"
                )
        self.conn.commit()

    def _begin(self) -> None:
        self.conn.execute("BEGIN IMMEDIATE")

    def _append_event(self, kind: str, details: dict[str, Any]) -> str:
        row = self.conn.execute(
            "SELECT seq,event_hash FROM router_events ORDER BY seq DESC LIMIT 1"
        ).fetchone()
        seq = 1 if row is None else int(row[0]) + 1
        prev_hash = "GENESIS" if row is None else str(row[1])
        payload = {"seq": seq, "kind": kind, "details": details}
        event_hash = digest({"prev_hash": prev_hash, "payload": payload})
        self.conn.execute(
            "INSERT INTO router_events VALUES (?,?,?,?,?)",
            (seq, kind, canonical(payload), prev_hash, event_hash),
        )
        return event_hash

    def _initialize(self, initial_model: str) -> None:
        if not initial_model:
            raise ValueError("initial model is required")
        self._begin()
        try:
            self.conn.execute("INSERT INTO router_state VALUES (1,?,0)", (initial_model,))
            self._append_event(
                "INITIALIZE", {"active_model": initial_model, "generation": 0}
            )
            self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise

    @property
    def state(self) -> tuple[str, int]:
        row = self.conn.execute(
            "SELECT active_model,generation FROM router_state WHERE singleton=1"
        ).fetchone()
        if row is None:
            raise RuntimeError("router is not initialized")
        return str(row[0]), int(row[1])

    def lookup(self, idempotency_key: str) -> RouteReceipt | None:
        row = self.conn.execute(
            "SELECT receipt_json FROM receipts WHERE idempotency_key=?",
            (idempotency_key,),
        ).fetchone()
        return None if row is None else restore_receipt(str(row[0]))

    def apply(self, command: RouteCommand, fencing_token: int) -> RouteReceipt:
        if fencing_token < 1:
            raise ValueError("router requires a positive fencing token")
        validate_command(command)
        self._begin()
        try:
            existing = self.conn.execute(
                "SELECT payload_hash,receipt_json FROM receipts WHERE idempotency_key=?",
                (command.idempotency_key,),
            ).fetchone()
            if existing is not None:
                if str(existing[0]) != command.payload_hash:
                    raise RuntimeError("idempotency key reused with a different payload")
                receipt = restore_receipt(str(existing[1]))
                self.conn.commit()
                return receipt

            active_model, generation = self.state
            if (active_model, generation) != (
                command.from_model, command.expected_generation,
            ):
                raise RuntimeError("router generation/from-model compare-and-swap failed")
            receipt_core = {
                "command_id": command.command_id,
                "idempotency_key": command.idempotency_key,
                "payload_hash": command.payload_hash,
                "from_model": command.from_model,
                "to_model": command.to_model,
                "applied_generation": generation + 1,
            }
            receipt = RouteReceipt(
                receipt_id=f"receipt-{digest(receipt_core)[:12]}", **receipt_core
            )
            self.conn.execute(
                "INSERT INTO receipts VALUES (?,?,?)",
                (command.idempotency_key, command.payload_hash, canonical(asdict(receipt))),
            )
            self._append_event(
                "APPLY",
                {
                    "command": asdict(command),
                    "receipt": asdict(receipt),
                    "fencing_token": fencing_token,
                },
            )
            updated = self.conn.execute(
                "UPDATE router_state SET active_model=?,generation=? "
                "WHERE singleton=1 AND active_model=? AND generation=?",
                (
                    command.to_model, generation + 1,
                    command.from_model, command.expected_generation,
                ),
            )
            if updated.rowcount != 1:
                raise RuntimeError("router state changed during apply")
            self.conn.commit()
            return receipt
        except Exception:
            self.conn.rollback()
            raise

    def admin_override(self, to_model: str, ticket_id: str) -> int:
        """Auditable external change that the controller did not authorize."""
        if not to_model or not ticket_id:
            raise ValueError("admin override requires a model and ticket")
        self._begin()
        try:
            from_model, generation = self.state
            new_generation = generation + 1
            self._append_event(
                "ADMIN_OVERRIDE",
                {
                    "from_model": from_model,
                    "to_model": to_model,
                    "old_generation": generation,
                    "new_generation": new_generation,
                    "ticket_id": ticket_id,
                },
            )
            self.conn.execute(
                "UPDATE router_state SET active_model=?,generation=? WHERE singleton=1",
                (to_model, new_generation),
            )
            self.conn.commit()
            return new_generation
        except Exception:
            self.conn.rollback()
            raise

    def receipt_count(self) -> int:
        return int(self.conn.execute("SELECT COUNT(*) FROM receipts").fetchone()[0])

    def verify_chain(self) -> bool:
        expected_seq = 1
        expected_prev = "GENESIS"
        for seq, payload_json, prev_hash, event_hash in self.conn.execute(
            "SELECT seq,payload_json,prev_hash,event_hash FROM router_events ORDER BY seq"
        ):
            if int(seq) != expected_seq or str(prev_hash) != expected_prev:
                return False
            calculated = digest(
                {"prev_hash": str(prev_hash), "payload": json.loads(payload_json)}
            )
            if calculated != str(event_hash):
                return False
            expected_seq += 1
            expected_prev = str(event_hash)
        return expected_seq > 1

    def verify_projection(self) -> bool:
        active_model: str | None = None
        generation: int | None = None
        apply_count = 0
        for (payload_json,) in self.conn.execute(
            "SELECT payload_json FROM router_events ORDER BY seq"
        ):
            event = json.loads(payload_json)
            details = event["details"]
            if event["kind"] == "INITIALIZE":
                if active_model is not None:
                    return False
                active_model = details["active_model"]
                generation = int(details["generation"])
            elif event["kind"] == "APPLY":
                command = details["command"]
                receipt = details["receipt"]
                if (command["from_model"], command["expected_generation"]) != (
                    active_model, generation,
                ):
                    return False
                active_model = command["to_model"]
                generation = int(receipt["applied_generation"])
                apply_count += 1
            elif event["kind"] == "ADMIN_OVERRIDE":
                if (details["from_model"], details["old_generation"]) != (
                    active_model, generation,
                ):
                    return False
                active_model = details["to_model"]
                generation = int(details["new_generation"])
        return self.state == (active_model, generation) and self.receipt_count() == apply_count

    def close(self) -> None:
        self.conn.close()


class PublicationController:
    """Owns immutable route intents and delivery state, but not the actual route."""

    def __init__(self, path: Path, initial_model: str) -> None:
        self.path = path
        self.conn = sqlite3.connect(path)
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.conn.execute("PRAGMA journal_mode = WAL")
        self._create_schema()
        if self.conn.execute("SELECT 1 FROM control_state").fetchone() is None:
            self._initialize(initial_model)

    def _create_schema(self) -> None:
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS commands(
                command_id TEXT PRIMARY KEY,
                idempotency_key TEXT NOT NULL UNIQUE,
                payload_hash TEXT NOT NULL CHECK(length(payload_hash)=64),
                command_json TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS delivery_state(
                command_id TEXT PRIMARY KEY REFERENCES commands(command_id),
                status TEXT NOT NULL CHECK(status IN ('PENDING','ACKED','FAILED')),
                lease_owner TEXT,
                lease_until INTEGER,
                fencing_token INTEGER NOT NULL CHECK(fencing_token >= 0),
                receipt_json TEXT,
                error TEXT
            );
            CREATE TABLE IF NOT EXISTS controller_events(
                seq INTEGER PRIMARY KEY,
                kind TEXT NOT NULL,
                command_id TEXT,
                payload_json TEXT NOT NULL,
                prev_hash TEXT NOT NULL,
                event_hash TEXT NOT NULL UNIQUE
            );
            CREATE TABLE IF NOT EXISTS control_state(
                singleton INTEGER PRIMARY KEY CHECK(singleton=1),
                expected_model TEXT NOT NULL,
                expected_generation INTEGER NOT NULL CHECK(expected_generation >= 0),
                frozen INTEGER NOT NULL CHECK(frozen IN (0,1))
            );
            """
        )
        for table in ("commands", "controller_events"):
            for action in ("UPDATE", "DELETE"):
                self.conn.execute(
                    f"CREATE TRIGGER IF NOT EXISTS no_{action.lower()}_{table} "
                    f"BEFORE {action} ON {table} BEGIN "
                    f"SELECT RAISE(ABORT, '{table} is append-only'); END"
                )
        self.conn.commit()

    def _begin(self) -> None:
        self.conn.execute("BEGIN IMMEDIATE")

    def _append_event(
        self, kind: str, command_id: str | None, details: dict[str, Any]
    ) -> str:
        row = self.conn.execute(
            "SELECT seq,event_hash FROM controller_events ORDER BY seq DESC LIMIT 1"
        ).fetchone()
        seq = 1 if row is None else int(row[0]) + 1
        prev_hash = "GENESIS" if row is None else str(row[1])
        payload = {
            "seq": seq, "kind": kind, "command_id": command_id, "details": details,
        }
        event_hash = digest({"prev_hash": prev_hash, "payload": payload})
        self.conn.execute(
            "INSERT INTO controller_events VALUES (?,?,?,?,?,?)",
            (seq, kind, command_id, canonical(payload), prev_hash, event_hash),
        )
        return event_hash

    def _initialize(self, initial_model: str) -> None:
        if not initial_model:
            raise ValueError("initial model is required")
        self._begin()
        try:
            self.conn.execute(
                "INSERT INTO control_state VALUES (1,?,0,0)", (initial_model,)
            )
            self._append_event(
                "INITIALIZE", None,
                {"expected_model": initial_model, "expected_generation": 0},
            )
            self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise

    @property
    def state(self) -> tuple[str, int, bool]:
        row = self.conn.execute(
            "SELECT expected_model,expected_generation,frozen "
            "FROM control_state WHERE singleton=1"
        ).fetchone()
        if row is None:
            raise RuntimeError("controller is not initialized")
        return str(row[0]), int(row[1]), bool(row[2])

    def _load_command(self, command_id: str) -> RouteCommand:
        row = self.conn.execute(
            "SELECT command_json FROM commands WHERE command_id=?", (command_id,)
        ).fetchone()
        if row is None:
            raise RuntimeError("unknown route command")
        return restore_command(str(row[0]))

    def enqueue(self, command: RouteCommand) -> None:
        validate_command(command)
        expected_model, expected_generation, frozen = self.state
        if frozen:
            raise RuntimeError("publication is frozen after router drift")
        if (command.from_model, command.expected_generation) != (
            expected_model, expected_generation,
        ):
            raise RuntimeError("command is stale relative to controller projection")
        if self.conn.execute(
            "SELECT 1 FROM delivery_state WHERE status='PENDING'"
        ).fetchone() is not None:
            raise RuntimeError("only one unresolved route command is allowed in this toy")
        self._begin()
        try:
            self.conn.execute(
                "INSERT INTO commands VALUES (?,?,?,?)",
                (
                    command.command_id, command.idempotency_key,
                    command.payload_hash, canonical(asdict(command)),
                ),
            )
            self.conn.execute(
                "INSERT INTO delivery_state VALUES (?,'PENDING',NULL,NULL,0,NULL,NULL)",
                (command.command_id,),
            )
            self._append_event(
                "ENQUEUE", command.command_id, {"command": asdict(command)}
            )
            self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise

    def claim(
        self, command_id: str, worker_id: str, now: int, ttl: int
    ) -> LeaseClaim:
        if not worker_id or ttl < 1:
            raise ValueError("worker id and positive lease ttl are required")
        self._begin()
        try:
            row = self.conn.execute(
                "SELECT status,lease_owner,lease_until,fencing_token "
                "FROM delivery_state WHERE command_id=?", (command_id,)
            ).fetchone()
            if row is None or str(row[0]) != "PENDING":
                raise RuntimeError("route command is not pending")
            owner = None if row[1] is None else str(row[1])
            lease_until = None if row[2] is None else int(row[2])
            token = int(row[3])
            if owner == worker_id and lease_until is not None and now < lease_until:
                self.conn.commit()
                return LeaseClaim(command_id, worker_id, token, lease_until)
            if lease_until is not None and now < lease_until:
                raise RuntimeError("route command has an active lease")
            token += 1
            lease_until = now + ttl
            self.conn.execute(
                "UPDATE delivery_state SET lease_owner=?,lease_until=?,fencing_token=? "
                "WHERE command_id=?",
                (worker_id, lease_until, token, command_id),
            )
            self._append_event(
                "CLAIM", command_id,
                {
                    "worker_id": worker_id,
                    "fencing_token": token,
                    "lease_until": lease_until,
                },
            )
            self.conn.commit()
            return LeaseClaim(command_id, worker_id, token, lease_until)
        except Exception:
            self.conn.rollback()
            raise

    @staticmethod
    def _validate_receipt(command: RouteCommand, receipt: RouteReceipt) -> None:
        expected = (
            command.command_id, command.idempotency_key, command.payload_hash,
            command.from_model, command.to_model, command.expected_generation + 1,
        )
        observed = (
            receipt.command_id, receipt.idempotency_key, receipt.payload_hash,
            receipt.from_model, receipt.to_model, receipt.applied_generation,
        )
        if observed != expected:
            raise RuntimeError("router receipt does not match the durable command")

    def acknowledge(
        self, claim: LeaseClaim, receipt: RouteReceipt, now: int
    ) -> None:
        command = self._load_command(claim.command_id)
        self._validate_receipt(command, receipt)
        self._begin()
        try:
            row = self.conn.execute(
                "SELECT status,lease_owner,lease_until,fencing_token "
                "FROM delivery_state WHERE command_id=?", (claim.command_id,)
            ).fetchone()
            if row is None or str(row[0]) != "PENDING":
                raise RuntimeError("route command is not awaiting acknowledgement")
            if (
                str(row[1]), int(row[2]), int(row[3])
            ) != (
                claim.worker_id, claim.lease_until, claim.fencing_token,
            ) or now > claim.lease_until:
                raise RuntimeError("stale worker fencing token cannot acknowledge")
            expected_model, expected_generation, frozen = self.state
            if frozen or (expected_model, expected_generation) != (
                command.from_model, command.expected_generation,
            ):
                raise RuntimeError("controller projection changed before acknowledgement")
            self._append_event(
                "ACK", command.command_id,
                {
                    "worker_id": claim.worker_id,
                    "fencing_token": claim.fencing_token,
                    "receipt": asdict(receipt),
                },
            )
            self.conn.execute(
                "UPDATE delivery_state SET status='ACKED',receipt_json=?,error=NULL "
                "WHERE command_id=?",
                (canonical(asdict(receipt)), command.command_id),
            )
            updated = self.conn.execute(
                "UPDATE control_state SET expected_model=?,expected_generation=? "
                "WHERE singleton=1 AND expected_model=? AND expected_generation=? AND frozen=0",
                (
                    command.to_model, receipt.applied_generation,
                    command.from_model, command.expected_generation,
                ),
            )
            if updated.rowcount != 1:
                raise RuntimeError("controller acknowledgement CAS failed")
            self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise

    def _recover_receipt(self, command: RouteCommand, receipt: RouteReceipt) -> None:
        self._validate_receipt(command, receipt)
        self._begin()
        try:
            row = self.conn.execute(
                "SELECT status FROM delivery_state WHERE command_id=?",
                (command.command_id,),
            ).fetchone()
            if row is None or str(row[0]) != "PENDING":
                raise RuntimeError("only a pending command can recover a receipt")
            expected_model, expected_generation, frozen = self.state
            if frozen or (expected_model, expected_generation) != (
                command.from_model, command.expected_generation,
            ):
                raise RuntimeError("receipt cannot advance the current controller projection")
            self._append_event(
                "RECONCILE_ACK", command.command_id, {"receipt": asdict(receipt)}
            )
            self.conn.execute(
                "UPDATE delivery_state SET status='ACKED',receipt_json=?,error=NULL "
                "WHERE command_id=?",
                (canonical(asdict(receipt)), command.command_id),
            )
            self.conn.execute(
                "UPDATE control_state SET expected_model=?,expected_generation=? "
                "WHERE singleton=1",
                (command.to_model, receipt.applied_generation),
            )
            self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise

    def reconcile(self, router: ExternalRouter) -> dict[str, Any]:
        recovered: list[str] = []
        pending = list(self.conn.execute(
            "SELECT c.command_id,c.idempotency_key FROM commands c "
            "JOIN delivery_state d USING(command_id) WHERE d.status='PENDING'"
        ))
        for command_id, idempotency_key in pending:
            receipt = router.lookup(str(idempotency_key))
            if receipt is not None:
                command = self._load_command(str(command_id))
                self._recover_receipt(command, receipt)
                recovered.append(str(command_id))

        expected_model, expected_generation, frozen = self.state
        observed_model, observed_generation = router.state
        consistent = (expected_model, expected_generation) == (
            observed_model, observed_generation,
        )
        if not consistent and not frozen:
            self._begin()
            try:
                self._append_event(
                    "ROUTER_DRIFT", None,
                    {
                        "expected_model": expected_model,
                        "expected_generation": expected_generation,
                        "observed_model": observed_model,
                        "observed_generation": observed_generation,
                    },
                )
                self.conn.execute(
                    "UPDATE control_state SET frozen=1 WHERE singleton=1"
                )
                self.conn.commit()
                frozen = True
            except Exception:
                self.conn.rollback()
                raise
        return {"recovered": tuple(recovered), "consistent": consistent, "frozen": frozen}

    def verify_chain(self) -> bool:
        expected_seq = 1
        expected_prev = "GENESIS"
        for seq, payload_json, prev_hash, event_hash in self.conn.execute(
            "SELECT seq,payload_json,prev_hash,event_hash "
            "FROM controller_events ORDER BY seq"
        ):
            if int(seq) != expected_seq or str(prev_hash) != expected_prev:
                return False
            calculated = digest(
                {"prev_hash": str(prev_hash), "payload": json.loads(payload_json)}
            )
            if calculated != str(event_hash):
                return False
            expected_seq += 1
            expected_prev = str(event_hash)
        return expected_seq > 1

    def verify_projection(self) -> bool:
        expected_model: str | None = None
        expected_generation: int | None = None
        frozen = False
        delivery: dict[str, dict[str, Any]] = {}
        for (payload_json,) in self.conn.execute(
            "SELECT payload_json FROM controller_events ORDER BY seq"
        ):
            event = json.loads(payload_json)
            kind = event["kind"]
            command_id = event["command_id"]
            details = event["details"]
            if kind == "INITIALIZE":
                expected_model = details["expected_model"]
                expected_generation = int(details["expected_generation"])
            elif kind == "ENQUEUE":
                delivery[command_id] = {
                    "status": "PENDING", "lease_owner": None, "lease_until": None,
                    "fencing_token": 0, "receipt_json": None, "error": None,
                }
            elif kind == "CLAIM":
                state = delivery[command_id]
                state.update(
                    lease_owner=details["worker_id"],
                    lease_until=int(details["lease_until"]),
                    fencing_token=int(details["fencing_token"]),
                )
            elif kind in ("ACK", "RECONCILE_ACK"):
                receipt = details["receipt"]
                delivery[command_id].update(
                    status="ACKED", receipt_json=canonical(receipt), error=None
                )
                expected_model = receipt["to_model"]
                expected_generation = int(receipt["applied_generation"])
            elif kind == "ROUTER_DRIFT":
                frozen = True

        if self.state != (expected_model, expected_generation, frozen):
            return False
        rows = self.conn.execute(
            "SELECT command_id,status,lease_owner,lease_until,fencing_token,receipt_json,error "
            "FROM delivery_state"
        )
        observed = {
            str(row[0]): {
                "status": str(row[1]),
                "lease_owner": None if row[2] is None else str(row[2]),
                "lease_until": None if row[3] is None else int(row[3]),
                "fencing_token": int(row[4]),
                "receipt_json": None if row[5] is None else str(row[5]),
                "error": None if row[6] is None else str(row[6]),
            }
            for row in rows
        }
        return observed == delivery

    def close(self) -> None:
        self.conn.close()


def demo() -> None:
    parent = "model-parent-v7"
    candidate = "candidate-robust"
    decision_id = "decision-7534d48c13e17619"

    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        controller = PublicationController(base / "controller.sqlite", parent)
        router = ExternalRouter(base / "router.sqlite", parent)
        separate_authorities = controller.path != router.path

        activate = make_command(decision_id, "ACTIVATE", parent, candidate, 0)
        controller.enqueue(activate)
        claim_a = controller.claim(activate.command_id, "worker-a", now=10, ttl=5)
        activation_receipt = router.apply(activate, claim_a.fencing_token)
        route_after_crash = router.state

        active_lease_rejected = False
        try:
            controller.claim(activate.command_id, "worker-b", now=12, ttl=5)
        except RuntimeError:
            active_lease_rejected = True
        claim_b = controller.claim(activate.command_id, "worker-b", now=16, ttl=5)
        replayed_receipt = router.apply(activate, claim_b.fencing_token)

        stale_ack_rejected = False
        try:
            controller.acknowledge(claim_a, activation_receipt, now=16)
        except RuntimeError:
            stale_ack_rejected = True
        controller.acknowledge(claim_b, replayed_receipt, now=17)
        activation_generation = router.state[1]

        payload_collision_rejected = False
        collision = make_command(
            decision_id, "ACTIVATE", parent, "candidate-altered", 0
        )
        try:
            router.apply(collision, fencing_token=3)
        except RuntimeError:
            payload_collision_rejected = True

        stale_generation_rejected = False
        stale = make_command(
            "decision-stale", "ACTIVATE", parent, "candidate-stale", 0
        )
        try:
            router.apply(stale, fencing_token=3)
        except RuntimeError:
            stale_generation_rejected = True

        forged_hash_rejected = False
        try:
            router.apply(replace(stale, payload_hash="f" * 64), fencing_token=3)
        except RuntimeError:
            forged_hash_rejected = True

        rollback = make_command(decision_id, "ROLLBACK", candidate, parent, 1)
        controller.enqueue(rollback)
        rollback_claim = controller.claim(
            rollback.command_id, "worker-b", now=20, ttl=5
        )
        rollback_receipt = router.apply(rollback, rollback_claim.fencing_token)
        reconciliation = controller.reconcile(router)
        replayed_rollback = router.apply(rollback, rollback_claim.fencing_token)

        router.admin_override("rogue-model", "break-glass-ticket-17")
        drift = controller.reconcile(router)
        frozen_enqueue_rejected = False
        try:
            controller.enqueue(
                make_command("decision-after-drift", "ACTIVATE", parent, candidate, 2)
            )
        except RuntimeError:
            frozen_enqueue_rejected = True

        immutable_controller_rejected = False
        try:
            with controller.conn:
                controller.conn.execute(
                    "UPDATE commands SET payload_hash=? WHERE command_id=?",
                    ("0" * 64, activate.command_id),
                )
        except sqlite3.IntegrityError:
            immutable_controller_rejected = True
        immutable_router_rejected = False
        try:
            with router.conn:
                router.conn.execute(
                    "DELETE FROM receipts WHERE idempotency_key=?",
                    (activate.idempotency_key,),
                )
        except sqlite3.IntegrityError:
            immutable_router_rejected = True

        checks = (
            ("controller and router use separate durable authorities", separate_authorities),
            ("router committed while controller still awaited an ack", route_after_crash == (candidate, 1)),
            ("an unexpired lease cannot be stolen", active_lease_rejected),
            ("lease expiry issues a higher fencing token", claim_b.fencing_token > claim_a.fencing_token),
            ("retry returns the original payload-bound receipt", replayed_receipt == activation_receipt),
            ("activation retry does not increment router generation", activation_generation == 1),
            ("a stale worker token cannot acknowledge", stale_ack_rejected),
            ("same key with a changed payload fails closed", payload_collision_rejected),
            ("router recomputes command identity instead of trusting its hash", forged_hash_rejected),
            ("router rejects a stale generation/from-model command", stale_generation_rejected),
            ("reconcile recovered the lost rollback acknowledgement", reconciliation["recovered"] == (rollback.command_id,)),
            ("reconciled controller and router projections agree", reconciliation["consistent"]),
            ("rollback retry returns one receipt and one effect", replayed_rollback == rollback_receipt and router.receipt_count() == 2),
            ("out-of-band route drift freezes publication", not drift["consistent"] and drift["frozen"]),
            ("frozen publication rejects a new command", frozen_enqueue_rejected),
            ("controller commands reject mutation", immutable_controller_rejected),
            ("router receipts reject deletion", immutable_router_rejected),
            ("controller event hash chain verifies", controller.verify_chain()),
            ("router event hash chain verifies", router.verify_chain()),
            ("controller projection matches event replay", controller.verify_projection()),
            ("router projection matches event replay", router.verify_projection()),
        )

        print("[1] outbox delivery crosses two durable authorities")
        print(
            f"    router_after_crash={route_after_crash[0]}@g{route_after_crash[1]} "
            f"controller_pending=True"
        )
        print("[2] lease expiry and idempotent retry close the lost-ack gap")
        print(
            f"    fence={claim_a.fencing_token}->{claim_b.fencing_token} "
            f"same_receipt={activation_receipt == replayed_receipt} "
            f"router_generation={activation_generation}"
        )
        print("[3] payload and generation guards reject stale publication")
        print(
            f"    payload_collision_rejected={payload_collision_rejected} "
            f"forged_hash_rejected={forged_hash_rejected} "
            f"stale_generation_rejected={stale_generation_rejected}"
        )
        print("[4] reconcile recovers a committed rollback with a lost ack")
        print(
            f"    recovered={len(reconciliation['recovered'])} "
            f"controller={controller.state[0]}@g{controller.state[1]}"
        )
        print("[5] unexpected router drift freezes new publication")
        print(
            f"    router={router.state[0]}@g{router.state[1]} "
            f"frozen={controller.state[2]}"
        )
        print("[6] structural self-check")
        for label, passed in checks:
            print(f"    {'PASS' if passed else 'FAIL'} | {label}")
        passed_count = sum(passed for _, passed in checks)
        print(f"SELF-CHECK: {passed_count}/{len(checks)} PASS")
        print(
            "takeaway: an outbox does not create cross-service ACID; stable keys, "
            "generation guards, durable receipts, and reconciliation make gaps auditable."
        )
        controller.close()
        router.close()


if __name__ == "__main__":
    demo()
