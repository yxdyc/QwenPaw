#!/usr/bin/env python3
"""L1: durable promotion, crash recovery, and executable rollback."""

from __future__ import annotations

from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import sqlite3
import tempfile
from typing import Any

from evaluation_gate_lab import (
    EvalBundle,
    EvalRow,
    GateConfig,
    PromotionRecord,
    run_gate,
    synthetic_bundle,
)


def canonical(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def digest(payload: Any) -> str:
    return hashlib.sha256(canonical(payload).encode()).hexdigest()


def record_payload(record: PromotionRecord) -> dict[str, Any]:
    return asdict(record)


def restore_record(payload: dict[str, Any]) -> PromotionRecord:
    return PromotionRecord(
        payload["record_id"], payload["parent_id"], payload["candidate_id"],
        payload["evidence_id"], payload["decision"], tuple(payload["reasons"]),
        tuple((name, float(value)) for name, value in payload["metrics"]),
        payload["rollback_target"],
    )


def evidence_payload(bundle: EvalBundle, cfg: GateConfig) -> dict[str, Any]:
    return {"bundle": asdict(bundle), "config": asdict(cfg)}


def restore_evidence(payload: dict[str, Any]) -> tuple[EvalBundle, GateConfig]:
    raw = payload["bundle"]
    bundle = EvalBundle(
        raw["parent_id"], raw["candidate_id"], raw["dataset_snapshot"],
        raw["evaluator_version"], raw["environment_version"],
        float(raw["parent_cost"]), float(raw["candidate_cost"]),
        tuple(EvalRow(**row) for row in raw["rows"]),
    )
    return bundle, GateConfig(**payload["config"])


class PromotionStore:
    """One SQLite authority: immutable facts plus one mutable active pointer."""

    def __init__(self, path: Path) -> None:
        self.conn = sqlite3.connect(path)
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.conn.execute("PRAGMA journal_mode = WAL")
        self._create_schema()

    def _create_schema(self) -> None:
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS snapshots(
                snapshot_id TEXT PRIMARY KEY,
                parent_id TEXT NOT NULL,
                manifest_sha256 TEXT NOT NULL CHECK(length(manifest_sha256) = 64)
            );
            CREATE TABLE IF NOT EXISTS evidence(
                evidence_digest TEXT PRIMARY KEY,
                payload_json TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS decisions(
                record_id TEXT PRIMARY KEY,
                evidence_digest TEXT NOT NULL REFERENCES evidence(evidence_digest),
                record_json TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS events(
                seq INTEGER PRIMARY KEY,
                kind TEXT NOT NULL CHECK(kind IN ('PREPARE','ACTIVATE','ROLLBACK')),
                record_id TEXT NOT NULL REFERENCES decisions(record_id),
                from_model TEXT NOT NULL,
                to_model TEXT NOT NULL,
                idempotency_key TEXT NOT NULL UNIQUE,
                payload_json TEXT NOT NULL,
                prev_hash TEXT NOT NULL,
                event_hash TEXT NOT NULL UNIQUE
            );
            CREATE TABLE IF NOT EXISTS control_state(
                singleton INTEGER PRIMARY KEY CHECK(singleton = 1),
                active_model TEXT NOT NULL,
                generation INTEGER NOT NULL
            );
            """
        )
        for table in ("snapshots", "evidence", "decisions", "events"):
            for action in ("UPDATE", "DELETE"):
                self.conn.execute(
                    f"CREATE TRIGGER IF NOT EXISTS no_{action.lower()}_{table} "
                    f"BEFORE {action} ON {table} BEGIN "
                    f"SELECT RAISE(ABORT, '{table} is append-only'); END"
                )
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    @property
    def active_model(self) -> str:
        row = self.conn.execute(
            "SELECT active_model FROM control_state WHERE singleton=1"
        ).fetchone()
        if row is None:
            raise RuntimeError("control state is not initialized")
        return str(row[0])

    def register_snapshot(self, snapshot_id: str, parent_id: str) -> str:
        manifest = digest({"snapshot_id": snapshot_id, "parent_id": parent_id})
        existing = self.conn.execute(
            "SELECT parent_id, manifest_sha256 FROM snapshots WHERE snapshot_id=?",
            (snapshot_id,),
        ).fetchone()
        if existing is not None:
            if existing != (parent_id, manifest):
                raise RuntimeError("snapshot identity collision")
            return manifest
        with self.conn:
            self.conn.execute(
                "INSERT INTO snapshots VALUES (?,?,?)", (snapshot_id, parent_id, manifest)
            )
        return manifest

    def initialize_active(self, snapshot_id: str) -> None:
        if self.conn.execute(
            "SELECT 1 FROM snapshots WHERE snapshot_id=?", (snapshot_id,)
        ).fetchone() is None:
            raise RuntimeError("initial active snapshot is not registered")
        existing = self.conn.execute(
            "SELECT active_model FROM control_state WHERE singleton=1"
        ).fetchone()
        if existing is not None and existing[0] != snapshot_id:
            raise RuntimeError("control state is already initialized")
        with self.conn:
            self.conn.execute(
                "INSERT OR IGNORE INTO control_state(singleton, active_model, generation) "
                "VALUES (1, ?, 0)", (snapshot_id,),
            )

    def persist_decision(
        self, bundle: EvalBundle, cfg: GateConfig, record: PromotionRecord
    ) -> str:
        recomputed = run_gate(bundle, cfg)
        if record_payload(recomputed) != record_payload(record):
            raise RuntimeError("decision does not recompute from raw evidence")
        raw = evidence_payload(bundle, cfg)
        evidence_digest = digest(raw)
        evidence_json = canonical(raw)
        record_json = canonical(record_payload(record))
        existing = self.conn.execute(
            "SELECT evidence_digest, record_json FROM decisions WHERE record_id=?",
            (record.record_id,),
        ).fetchone()
        if existing is not None:
            if existing != (evidence_digest, record_json):
                raise RuntimeError("record identity collision")
            return evidence_digest
        with self.conn:
            self.conn.execute(
                "INSERT OR IGNORE INTO evidence VALUES (?,?)",
                (evidence_digest, evidence_json),
            )
            self.conn.execute(
                "INSERT INTO decisions VALUES (?,?,?)",
                (record.record_id, evidence_digest, record_json),
            )
        return evidence_digest

    def load_record(self, record_id: str) -> PromotionRecord:
        row = self.conn.execute(
            "SELECT record_json FROM decisions WHERE record_id=?", (record_id,)
        ).fetchone()
        if row is None:
            raise RuntimeError("unknown decision")
        return restore_record(json.loads(row[0]))

    def audit_decision(self, record_id: str) -> bool:
        row = self.conn.execute(
            "SELECT d.record_json, e.payload_json FROM decisions d "
            "JOIN evidence e USING(evidence_digest) WHERE d.record_id=?",
            (record_id,),
        ).fetchone()
        if row is None:
            return False
        stored = restore_record(json.loads(row[0]))
        bundle, cfg = restore_evidence(json.loads(row[1]))
        return record_payload(run_gate(bundle, cfg)) == record_payload(stored)

    def _event_for_key(self, key: str) -> tuple[int, str] | None:
        row = self.conn.execute(
            "SELECT seq, event_hash FROM events WHERE idempotency_key=?", (key,)
        ).fetchone()
        return None if row is None else (int(row[0]), str(row[1]))

    def _append_event(
        self, kind: str, record: PromotionRecord, from_model: str, to_model: str,
        key: str, extra: dict[str, Any],
    ) -> tuple[int, str]:
        existing = self._event_for_key(key)
        if existing is not None:
            return existing
        last = self.conn.execute(
            "SELECT seq, event_hash FROM events ORDER BY seq DESC LIMIT 1"
        ).fetchone()
        seq = 1 if last is None else int(last[0]) + 1
        prev_hash = "GENESIS" if last is None else str(last[1])
        payload = {
            "seq": seq, "kind": kind, "record_id": record.record_id,
            "from_model": from_model, "to_model": to_model,
            "idempotency_key": key, "extra": extra,
        }
        event_hash = digest({"prev_hash": prev_hash, "payload": payload})
        self.conn.execute(
            "INSERT INTO events VALUES (?,?,?,?,?,?,?,?,?)",
            (seq, kind, record.record_id, from_model, to_model, key,
             canonical(payload), prev_hash, event_hash),
        )
        return seq, event_hash

    def prepare(self, record_id: str) -> tuple[int, str]:
        key = f"prepare:{record_id}"
        existing = self._event_for_key(key)
        if existing is not None:
            return existing
        record = self.load_record(record_id)
        if record.decision != "PROMOTE":
            raise RuntimeError("only a PROMOTE decision can be prepared")
        if self.active_model != record.parent_id:
            raise RuntimeError("stale parent: active model changed after evaluation")
        if record.rollback_target != record.parent_id:
            raise RuntimeError("rollback target is not the evaluated parent")
        ids = {row[0] for row in self.conn.execute("SELECT snapshot_id FROM snapshots")}
        if record.parent_id not in ids or record.candidate_id not in ids:
            raise RuntimeError("missing immutable snapshot")
        with self.conn:
            return self._append_event(
                "PREPARE", record, record.parent_id, record.candidate_id, key,
                {"rollback_target": record.rollback_target},
            )

    def activate(self, record_id: str) -> tuple[int, str]:
        key = f"activate:{record_id}"
        existing = self._event_for_key(key)
        if existing is not None:
            return existing
        record = self.load_record(record_id)
        if self._event_for_key(f"prepare:{record_id}") is None:
            raise RuntimeError("promotion was not durably prepared")
        if self.active_model != record.parent_id:
            raise RuntimeError("stale parent at activation")
        with self.conn:
            event = self._append_event(
                "ACTIVATE", record, record.parent_id, record.candidate_id, key,
                {"decision": record.decision},
            )
            changed = self.conn.execute(
                "UPDATE control_state SET active_model=?, generation=generation+1 "
                "WHERE singleton=1 AND active_model=?",
                (record.candidate_id, record.parent_id),
            ).rowcount
            if changed != 1:
                raise RuntimeError("compare-and-swap activation failed")
            return event

    def rollback(self, record_id: str, sentinel_reason: str) -> tuple[int, str]:
        key = f"rollback:{record_id}"
        existing = self._event_for_key(key)
        if existing is not None:
            return existing
        record = self.load_record(record_id)
        if self.active_model != record.candidate_id:
            raise RuntimeError("rollback source is not active")
        with self.conn:
            event = self._append_event(
                "ROLLBACK", record, record.candidate_id, record.rollback_target, key,
                {"sentinel_reason": sentinel_reason},
            )
            changed = self.conn.execute(
                "UPDATE control_state SET active_model=?, generation=generation+1 "
                "WHERE singleton=1 AND active_model=?",
                (record.rollback_target, record.candidate_id),
            ).rowcount
            if changed != 1:
                raise RuntimeError("compare-and-swap rollback failed")
            return event

    def event_count(self, kind: str) -> int:
        return int(self.conn.execute(
            "SELECT count(*) FROM events WHERE kind=?", (kind,)
        ).fetchone()[0])

    def verify_event_chain(self) -> bool:
        prev_hash = "GENESIS"
        for seq, payload_json, stored_prev, event_hash in self.conn.execute(
            "SELECT seq, payload_json, prev_hash, event_hash FROM events ORDER BY seq"
        ):
            payload = json.loads(payload_json)
            if payload["seq"] != seq or stored_prev != prev_hash:
                return False
            if digest({"prev_hash": prev_hash, "payload": payload}) != event_hash:
                return False
            prev_hash = event_hash
        return True


def expect_reject(action: Any, phrase: str) -> bool:
    try:
        action()
    except (RuntimeError, sqlite3.IntegrityError) as exc:
        return phrase in str(exc)
    return False


def main() -> None:
    cfg = GateConfig()
    robust = synthetic_bundle(
        "candidate-robust", [0.03, 0.04, 0.05, 0.04, 0.03] * 4,
        [0.02, 0.03, 0.04, 0.025], 1.10,
    )
    stale = replace(robust, candidate_id="candidate-stale")
    headline = synthetic_bundle(
        "candidate-headline", [0.08] * 20, [-0.03, -0.02, -0.18, -0.01], 1.05,
    )
    robust_record, stale_record, reject_record = (
        run_gate(robust, cfg), run_gate(stale, cfg), run_gate(headline, cfg)
    )

    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "promotion.db"
        store = PromotionStore(db_path)
        for snapshot_id, parent_id in (
            (robust.parent_id, "model-parent-v6"),
            (robust.candidate_id, robust.parent_id),
            (stale.candidate_id, stale.parent_id),
            (headline.candidate_id, headline.parent_id),
        ):
            store.register_snapshot(snapshot_id, parent_id)
        store.initialize_active(robust.parent_id)
        for bundle, record in (
            (robust, robust_record), (stale, stale_record), (headline, reject_record)
        ):
            store.persist_decision(bundle, cfg, record)

        print("[1] stored raw evidence -> fresh gate recomputation")
        print(f"    decision={robust_record.record_id} audit_match={store.audit_decision(robust_record.record_id)}")
        reject_prepare = expect_reject(
            lambda: store.prepare(reject_record.record_id), "only a PROMOTE"
        )
        print(f"    REJECT cannot prepare={reject_prepare}")

        print("[2] durable PREPARE, then simulated process crash")
        prepare_event = store.prepare(robust_record.record_id)
        store.close()  # crash boundary: PREPARE committed; ACTIVATE never ran
        store = PromotionStore(db_path)
        parent_survived = store.active_model == robust.parent_id
        print(f"    prepare_seq={prepare_event[0]} active_after_restart={store.active_model}")

        print("[3] recovery activates exactly once")
        first_activation = store.activate(robust_record.record_id)
        retry_activation = store.activate(robust_record.record_id)
        exactly_once = first_activation == retry_activation and store.event_count("ACTIVATE") == 1
        candidate_activated = store.active_model == robust.candidate_id
        print(f"    active={store.active_model} activation_seq={first_activation[0]} retry_same={exactly_once}")

        print("[4] stale parent fails closed")
        stale_rejected = expect_reject(
            lambda: store.prepare(stale_record.record_id), "stale parent"
        )
        print(f"    stale_candidate_prepared={not stale_rejected} events_unchanged={store.event_count('PREPARE') == 1}")

        print("[5] post-activation sentinel executes rollback")
        rollback_event = store.rollback(robust_record.record_id, "hidden-safety canary below floor")
        retry_rollback = store.rollback(robust_record.record_id, "duplicate alert")
        rollback_once = rollback_event == retry_rollback and store.event_count("ROLLBACK") == 1
        print(f"    active={store.active_model} rollback_seq={rollback_event[0]} retry_same={rollback_once}")

        immutable_blocked = expect_reject(
            lambda: store.conn.execute(
                "UPDATE decisions SET record_json='{}' WHERE record_id=?",
                (robust_record.record_id,),
            ),
            "append-only",
        )
        store.conn.rollback()
        decision_rows = int(store.conn.execute("SELECT count(*) FROM decisions").fetchone()[0])
        history_kept = decision_rows == 3 and store.event_count("ACTIVATE") == 1
        chain_ok = store.verify_event_chain()
        active_digest_ok = store.conn.execute(
            "SELECT length(manifest_sha256)=64 FROM snapshots WHERE snapshot_id=?",
            (store.active_model,),
        ).fetchone()[0] == 1

        checks = {
            "raw evidence recomputes the same decision": store.audit_decision(robust_record.record_id),
            "REJECT decision cannot enter deployment": reject_prepare,
            "crash after PREPARE leaves parent active": parent_survived,
            "recovery activates candidate": candidate_activated,
            "activation retry is exactly-once": exactly_once,
            "stale parent fails closed": stale_rejected,
            "stale attempt appends no PREPARE": store.event_count("PREPARE") == 1,
            "rollback returns to recorded parent": store.active_model == robust_record.rollback_target,
            "rollback retry is idempotent": rollback_once,
            "promotion history survives rollback": history_kept,
            "immutable table rejects UPDATE": immutable_blocked,
            "event hash chain verifies": chain_ok,
            "active pointer names a registered snapshot": active_digest_ok,
        }
        print("[6] structural self-check")
        for name, passed in checks.items():
            print(f"    {'PASS' if passed else 'FAIL'} | {name}")
        assert all(checks.values())
        print(f"SELF-CHECK: {sum(checks.values())}/{len(checks)} PASS")
        print("takeaway: promotion is an append-only, recoverable state transition; rollback is a new fact, not erased history.")
        store.close()


if __name__ == "__main__":
    main()
