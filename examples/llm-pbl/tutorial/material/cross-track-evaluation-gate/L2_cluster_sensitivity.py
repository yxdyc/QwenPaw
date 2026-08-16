#!/usr/bin/env python3
"""L2 companion: audit whether a gate conclusion depends on cluster policy."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import tempfile
from typing import Any, Iterable

from L2_evaluator_governance import exact_sign_flip_pvalue


AXES = ("task_id", "source_id", "user_id", "domain_id")


def canonical(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def digest(payload: Any) -> str:
    return hashlib.sha256(canonical(payload).encode()).hexdigest()


@dataclass(frozen=True)
class LineageRow:
    task_id: str
    source_id: str
    user_id: str
    domain_id: str
    traffic_weight: float


@dataclass(frozen=True)
class PairedDelta:
    task_id: str
    delta: float


@dataclass(frozen=True)
class ClusterPolicy:
    name: str
    axis: str
    alpha: float = 0.05


@dataclass(frozen=True)
class PolicyResult:
    policy_id: str
    name: str
    axis: str
    task_count: int
    cluster_count: int
    row_weighted_effect: float
    equal_cluster_effect: float
    p_value: float
    fixed_alpha_pass: bool


@dataclass(frozen=True)
class SensitivityReport:
    report_id: str
    manifest_id: str
    evidence_id: str
    status: str
    results: tuple[PolicyResult, ...]


def validate_manifest(rows: Iterable[LineageRow]) -> tuple[LineageRow, ...]:
    checked = tuple(sorted(rows, key=lambda row: row.task_id))
    task_ids = [row.task_id for row in checked]
    if not checked or len(task_ids) != len(set(task_ids)):
        raise ValueError("manifest requires unique task ids")
    for row in checked:
        fields = (row.task_id, row.source_id, row.user_id, row.domain_id)
        if any(not value.strip() for value in fields):
            raise ValueError("all lineage fields must be non-empty")
        if not math.isfinite(row.traffic_weight) or row.traffic_weight <= 0.0:
            raise ValueError("traffic weights must be finite and positive")
    if not math.isclose(
        sum(row.traffic_weight for row in checked), 1.0, abs_tol=1e-9
    ):
        raise ValueError("traffic weights must sum to one")
    return checked


def validate_evidence(rows: Iterable[PairedDelta]) -> tuple[PairedDelta, ...]:
    checked = tuple(sorted(rows, key=lambda row: row.task_id))
    task_ids = [row.task_id for row in checked]
    if not checked or len(task_ids) != len(set(task_ids)):
        raise ValueError("evidence requires unique task ids")
    if any(
        not row.task_id.strip()
        or not math.isfinite(row.delta)
        or not -1.0 <= row.delta <= 1.0
        for row in checked
    ):
        raise ValueError("evidence rows require valid task ids and deltas")
    return checked


def manifest_id(rows: Iterable[LineageRow]) -> str:
    checked = validate_manifest(rows)
    return f"manifest-{digest([asdict(row) for row in checked])[:12]}"


def evidence_id(rows: Iterable[PairedDelta]) -> str:
    checked = validate_evidence(rows)
    return f"evidence-{digest([asdict(row) for row in checked])[:12]}"


def policy_id(policy: ClusterPolicy, lineage_manifest_id: str) -> str:
    if not policy.name.strip() or policy.axis not in AXES:
        raise ValueError("policy requires a name and an allowed lineage axis")
    if not 0.0 < policy.alpha < 1.0:
        raise ValueError("policy alpha must be in (0, 1)")
    core = {
        "name": policy.name,
        "axis": policy.axis,
        "alpha": policy.alpha,
        "manifest_id": lineage_manifest_id,
        "within_cluster": "equal-row-mean-v1",
        "between_cluster": "equal-cluster-mean-v1",
        "test": "one-sided-exact-sign-flip-v1",
    }
    return f"policy-{digest(core)[:12]}"


def derive_cluster_ids(
    manifest: Iterable[LineageRow], axis: str
) -> dict[str, str]:
    checked = validate_manifest(manifest)
    if axis not in AXES:
        raise ValueError("cluster axis is not in the lineage contract")
    return {row.task_id: str(getattr(row, axis)) for row in checked}


def audit_policy(
    manifest: Iterable[LineageRow],
    evidence: Iterable[PairedDelta],
    policy: ClusterPolicy,
    submitted_cluster_ids: dict[str, str] | None = None,
) -> PolicyResult:
    checked_manifest = validate_manifest(manifest)
    checked_evidence = validate_evidence(evidence)
    manifest_tasks = {row.task_id for row in checked_manifest}
    evidence_by_task = {row.task_id: row.delta for row in checked_evidence}
    if set(evidence_by_task) != manifest_tasks:
        raise ValueError("evidence coverage must exactly match the frozen manifest")

    mid = manifest_id(checked_manifest)
    pid = policy_id(policy, mid)
    derived = derive_cluster_ids(checked_manifest, policy.axis)
    if submitted_cluster_ids is not None:
        normalized = {
            str(task_id).strip(): str(cluster_id).strip()
            for task_id, cluster_id in submitted_cluster_ids.items()
        }
        if normalized != derived:
            raise ValueError("submitted cluster ids disagree with manifest derivation")

    grouped: dict[str, list[float]] = {}
    for row in checked_manifest:
        grouped.setdefault(derived[row.task_id], []).append(
            evidence_by_task[row.task_id]
        )
    cluster_deltas = tuple(
        sum(grouped[cluster_id]) / len(grouped[cluster_id])
        for cluster_id in sorted(grouped)
    )
    p_value = exact_sign_flip_pvalue(cluster_deltas)
    row_effect = sum(
        row.traffic_weight * evidence_by_task[row.task_id]
        for row in checked_manifest
    )
    cluster_effect = sum(cluster_deltas) / len(cluster_deltas)
    if abs(cluster_effect) < 1e-15:
        cluster_effect = 0.0
    return PolicyResult(
        policy_id=pid,
        name=policy.name,
        axis=policy.axis,
        task_count=len(checked_manifest),
        cluster_count=len(cluster_deltas),
        row_weighted_effect=row_effect,
        equal_cluster_effect=cluster_effect,
        p_value=p_value,
        fixed_alpha_pass=cluster_effect > 0.0 and p_value <= policy.alpha,
    )


def audit_sensitivity(
    manifest: Iterable[LineageRow],
    evidence: Iterable[PairedDelta],
    policies: Iterable[ClusterPolicy],
) -> SensitivityReport:
    checked_manifest = validate_manifest(manifest)
    checked_evidence = validate_evidence(evidence)
    checked_policies = tuple(policies)
    axes = [policy.axis for policy in checked_policies]
    if len(axes) < 2 or len(axes) != len(set(axes)):
        raise ValueError("sensitivity audit requires at least two distinct axes")
    results = tuple(
        audit_policy(checked_manifest, checked_evidence, policy)
        for policy in checked_policies
    )
    decisions = {result.fixed_alpha_pass for result in results}
    status = "FREEZE" if len(decisions) > 1 else "CONSISTENT"
    mid = manifest_id(checked_manifest)
    eid = evidence_id(checked_evidence)
    report_core = {
        "manifest_id": mid,
        "evidence_id": eid,
        "status": status,
        "results": [asdict(result) for result in results],
    }
    return SensitivityReport(
        report_id=f"sensitivity-{digest(report_core)[:12]}",
        manifest_id=mid,
        evidence_id=eid,
        status=status,
        results=results,
    )


class PreregistrationLedger:
    """Append-only manifest -> policy -> evidence -> audit registry."""

    def __init__(self, path: Path) -> None:
        self.conn = sqlite3.connect(path)
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.conn.execute("PRAGMA journal_mode = WAL")
        self._create_schema()

    def _create_schema(self) -> None:
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS registry_events(
                seq INTEGER PRIMARY KEY,
                kind TEXT NOT NULL CHECK(kind IN (
                    'REGISTER_MANIFEST','REGISTER_POLICY',
                    'RECORD_EVIDENCE','AUDIT'
                )),
                subject_id TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                prev_hash TEXT NOT NULL,
                event_hash TEXT NOT NULL UNIQUE
            );
            CREATE TABLE IF NOT EXISTS manifests(
                manifest_id TEXT PRIMARY KEY,
                payload_json TEXT NOT NULL,
                event_seq INTEGER NOT NULL UNIQUE REFERENCES registry_events(seq)
            );
            CREATE TABLE IF NOT EXISTS policies(
                policy_id TEXT PRIMARY KEY,
                manifest_id TEXT NOT NULL REFERENCES manifests(manifest_id),
                payload_json TEXT NOT NULL,
                event_seq INTEGER NOT NULL UNIQUE REFERENCES registry_events(seq)
            );
            CREATE TABLE IF NOT EXISTS evidences(
                evidence_id TEXT PRIMARY KEY,
                manifest_id TEXT NOT NULL REFERENCES manifests(manifest_id),
                payload_json TEXT NOT NULL,
                event_seq INTEGER NOT NULL UNIQUE REFERENCES registry_events(seq)
            );
            CREATE TABLE IF NOT EXISTS audit_reports(
                report_id TEXT PRIMARY KEY,
                evidence_id TEXT NOT NULL UNIQUE REFERENCES evidences(evidence_id),
                payload_json TEXT NOT NULL,
                event_seq INTEGER NOT NULL UNIQUE REFERENCES registry_events(seq)
            );
            """
        )
        for table in (
            "registry_events", "manifests", "policies", "evidences",
            "audit_reports",
        ):
            for action in ("UPDATE", "DELETE"):
                self.conn.execute(
                    f"CREATE TRIGGER IF NOT EXISTS no_{action.lower()}_{table} "
                    f"BEFORE {action} ON {table} BEGIN "
                    f"SELECT RAISE(ABORT, '{table} is append-only'); END"
                )
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    def _append_event(
        self, kind: str, subject_id: str, details: dict[str, Any]
    ) -> int:
        row = self.conn.execute(
            "SELECT seq,event_hash FROM registry_events ORDER BY seq DESC LIMIT 1"
        ).fetchone()
        seq = 1 if row is None else int(row[0]) + 1
        prev_hash = "GENESIS" if row is None else str(row[1])
        payload = {
            "seq": seq,
            "kind": kind,
            "subject_id": subject_id,
            "details": details,
        }
        event_hash = digest({"prev_hash": prev_hash, "payload": payload})
        self.conn.execute(
            "INSERT INTO registry_events VALUES (?,?,?,?,?,?)",
            (seq, kind, subject_id, canonical(payload), prev_hash, event_hash),
        )
        return seq

    def register_manifest(self, rows: Iterable[LineageRow]) -> str:
        checked = validate_manifest(rows)
        mid = manifest_id(checked)
        payload_json = canonical([asdict(row) for row in checked])
        existing = self.conn.execute(
            "SELECT payload_json FROM manifests WHERE manifest_id=?", (mid,)
        ).fetchone()
        if existing is not None:
            if str(existing[0]) != payload_json:
                raise RuntimeError("manifest id collision")
            return mid
        with self.conn:
            seq = self._append_event(
                "REGISTER_MANIFEST", mid, {"row_count": len(checked)}
            )
            self.conn.execute(
                "INSERT INTO manifests VALUES (?,?,?)", (mid, payload_json, seq)
            )
        return mid

    def register_policy(self, mid: str, policy: ClusterPolicy) -> str:
        if self.conn.execute(
            "SELECT 1 FROM manifests WHERE manifest_id=?", (mid,)
        ).fetchone() is None:
            raise RuntimeError("policy requires a registered manifest")
        pid = policy_id(policy, mid)
        payload_json = canonical(asdict(policy))
        existing = self.conn.execute(
            "SELECT payload_json FROM policies WHERE policy_id=?", (pid,)
        ).fetchone()
        if existing is not None:
            if str(existing[0]) != payload_json:
                raise RuntimeError("policy id collision")
            return pid
        if self.conn.execute(
            "SELECT 1 FROM evidences WHERE manifest_id=?", (mid,)
        ).fetchone() is not None:
            raise RuntimeError("policy registration is closed after evidence")
        with self.conn:
            seq = self._append_event(
                "REGISTER_POLICY", pid, {"manifest_id": mid, "axis": policy.axis}
            )
            self.conn.execute(
                "INSERT INTO policies VALUES (?,?,?,?)",
                (pid, mid, payload_json, seq),
            )
        return pid

    def _load_manifest(self, mid: str) -> tuple[LineageRow, ...]:
        row = self.conn.execute(
            "SELECT payload_json FROM manifests WHERE manifest_id=?", (mid,)
        ).fetchone()
        if row is None:
            raise RuntimeError("unknown manifest")
        return tuple(LineageRow(**item) for item in json.loads(str(row[0])))

    def _load_evidence(self, eid: str) -> tuple[PairedDelta, ...]:
        row = self.conn.execute(
            "SELECT payload_json FROM evidences WHERE evidence_id=?", (eid,)
        ).fetchone()
        if row is None:
            raise RuntimeError("unknown evidence")
        return tuple(PairedDelta(**item) for item in json.loads(str(row[0])))

    def _load_policies(self, mid: str) -> tuple[ClusterPolicy, ...]:
        rows = self.conn.execute(
            "SELECT payload_json FROM policies WHERE manifest_id=? ORDER BY event_seq",
            (mid,),
        )
        return tuple(ClusterPolicy(**json.loads(str(row[0]))) for row in rows)

    def record_evidence(
        self, mid: str, rows: Iterable[PairedDelta]
    ) -> str:
        manifest = self._load_manifest(mid)
        checked = validate_evidence(rows)
        if {row.task_id for row in manifest} != {row.task_id for row in checked}:
            raise ValueError("evidence coverage must exactly match the manifest")
        policies = self._load_policies(mid)
        axes = [policy.axis for policy in policies]
        if len(axes) < 2 or len(axes) != len(set(axes)):
            raise RuntimeError(
                "evidence requires at least two distinct preregistered axes"
            )
        eid = evidence_id(checked)
        payload_json = canonical([asdict(row) for row in checked])
        existing = self.conn.execute(
            "SELECT manifest_id,payload_json FROM evidences WHERE evidence_id=?",
            (eid,),
        ).fetchone()
        if existing is not None:
            if (str(existing[0]), str(existing[1])) != (mid, payload_json):
                raise RuntimeError("evidence id collision")
            return eid
        registered_ids = tuple(
            str(row[0]) for row in self.conn.execute(
                "SELECT policy_id FROM policies WHERE manifest_id=? ORDER BY event_seq",
                (mid,),
            )
        )
        with self.conn:
            seq = self._append_event(
                "RECORD_EVIDENCE", eid,
                {"manifest_id": mid, "registered_policy_ids": registered_ids},
            )
            self.conn.execute(
                "INSERT INTO evidences VALUES (?,?,?,?)",
                (eid, mid, payload_json, seq),
            )
        return eid

    def run_registered_audit(self, eid: str) -> SensitivityReport:
        row = self.conn.execute(
            "SELECT manifest_id FROM evidences WHERE evidence_id=?", (eid,)
        ).fetchone()
        if row is None:
            raise RuntimeError("audit requires registered evidence")
        mid = str(row[0])
        policies = self._load_policies(mid)
        report = audit_sensitivity(
            self._load_manifest(mid), self._load_evidence(eid), policies
        )
        payload_json = canonical(asdict(report))
        existing = self.conn.execute(
            "SELECT payload_json FROM audit_reports WHERE report_id=?",
            (report.report_id,),
        ).fetchone()
        if existing is not None:
            if str(existing[0]) != payload_json:
                raise RuntimeError("report id collision")
            return report
        policy_ids = tuple(
            policy_id(policy, mid) for policy in policies
        )
        with self.conn:
            seq = self._append_event(
                "AUDIT", report.report_id,
                {
                    "evidence_id": eid,
                    "policy_ids": policy_ids,
                    "status": report.status,
                },
            )
            self.conn.execute(
                "INSERT INTO audit_reports VALUES (?,?,?,?)",
                (report.report_id, eid, payload_json, seq),
            )
        return report

    def load_report(self, report_id: str) -> SensitivityReport:
        row = self.conn.execute(
            "SELECT payload_json FROM audit_reports WHERE report_id=?", (report_id,)
        ).fetchone()
        if row is None:
            raise RuntimeError("unknown audit report")
        payload = json.loads(str(row[0]))
        return SensitivityReport(
            report_id=payload["report_id"],
            manifest_id=payload["manifest_id"],
            evidence_id=payload["evidence_id"],
            status=payload["status"],
            results=tuple(PolicyResult(**item) for item in payload["results"]),
        )

    def counts(self) -> tuple[int, int, int, int, int]:
        return tuple(
            int(self.conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
            for table in (
                "manifests", "policies", "evidences", "audit_reports",
                "registry_events",
            )
        )

    def verify_chain(self) -> bool:
        expected_seq = 1
        expected_prev = "GENESIS"
        rows = self.conn.execute(
            "SELECT seq,kind,subject_id,payload_json,prev_hash,event_hash "
            "FROM registry_events ORDER BY seq"
        )
        for seq, kind, subject_id, payload_json, prev_hash, event_hash in rows:
            if int(seq) != expected_seq or str(prev_hash) != expected_prev:
                return False
            payload = json.loads(str(payload_json))
            if (
                payload.get("seq") != int(seq)
                or payload.get("kind") != str(kind)
                or payload.get("subject_id") != str(subject_id)
            ):
                return False
            calculated = digest(
                {"prev_hash": str(prev_hash), "payload": payload}
            )
            if calculated != str(event_hash):
                return False
            expected_seq += 1
            expected_prev = str(event_hash)
        return expected_seq > 1

    def verify_registration_order(self) -> bool:
        manifests: dict[str, int] = {}
        policies: dict[str, tuple[str, int, str]] = {}
        evidence: dict[str, tuple[str, int]] = {}
        reports: dict[str, int] = {}
        for seq, payload_json in self.conn.execute(
            "SELECT seq,payload_json FROM registry_events ORDER BY seq"
        ):
            payload = json.loads(str(payload_json))
            kind = payload["kind"]
            subject_id = payload["subject_id"]
            details = payload["details"]
            if kind == "REGISTER_MANIFEST":
                if subject_id in manifests:
                    return False
                manifests[subject_id] = int(seq)
            elif kind == "REGISTER_POLICY":
                mid = details["manifest_id"]
                if subject_id in policies or mid not in manifests or any(
                    evidence_mid == mid for evidence_mid, _ in evidence.values()
                ):
                    return False
                policies[subject_id] = (mid, int(seq), details["axis"])
            elif kind == "RECORD_EVIDENCE":
                mid = details["manifest_id"]
                registered = tuple(details["registered_policy_ids"])
                expected = tuple(
                    pid for pid, (policy_mid, _, _) in policies.items()
                    if policy_mid == mid
                )
                if (
                    subject_id in evidence
                    or mid not in manifests
                    or len(registered) < 2
                    or len({policies[pid][2] for pid in expected}) < 2
                ):
                    return False
                if (
                    len(registered) != len(expected)
                    or set(registered) != set(expected)
                ):
                    return False
                if any(
                    pid not in policies
                    or policies[pid][0] != mid
                    or policies[pid][1] >= int(seq)
                    for pid in registered
                ):
                    return False
                evidence[subject_id] = (mid, int(seq))
            elif kind == "AUDIT":
                eid = details["evidence_id"]
                if eid not in evidence or evidence[eid][1] >= int(seq):
                    return False
                mid = evidence[eid][0]
                registered = tuple(details["policy_ids"])
                expected = tuple(
                    pid for pid, (policy_mid, _, _) in policies.items()
                    if policy_mid == mid
                )
                if (
                    subject_id in reports
                    or len(registered) != len(expected)
                    or set(registered) != set(expected)
                    or any(
                        policies[pid][1] >= int(seq) for pid in registered
                    )
                ):
                    return False
                reports[subject_id] = int(seq)
            else:
                return False
        if (
            len(manifests), len(policies), len(evidence), len(reports)
        ) != self.counts()[:4]:
            return False
        projections = (
            ("manifests", "manifest_id", "REGISTER_MANIFEST"),
            ("policies", "policy_id", "REGISTER_POLICY"),
            ("evidences", "evidence_id", "RECORD_EVIDENCE"),
            ("audit_reports", "report_id", "AUDIT"),
        )
        return all(
            int(self.conn.execute(
                f"SELECT COUNT(*) FROM {table} AS object "
                "JOIN registry_events AS event ON object.event_seq=event.seq "
                f"WHERE event.kind != ? OR event.subject_id != object.{id_column}",
                (kind,),
            ).fetchone()[0]) == 0
            for table, id_column, kind in projections
        )


def demo() -> None:
    manifest = tuple(
        LineageRow(
            task_id=f"task-{index:02d}",
            source_id="source-a" if index < 10 else "source-b",
            user_id=(
                "user-1" if index < 4 else
                "user-2" if index < 7 else
                "user-3" if index < 10 else "user-4"
            ),
            domain_id="domain-a" if index < 6 else "domain-b",
            traffic_weight=1.0 / 12.0,
        )
        for index in range(12)
    )
    evidence = tuple(
        PairedDelta(f"task-{index:02d}", 0.01 if index < 10 else -0.01)
        for index in range(12)
    )
    policies = (
        ClusterPolicy("naive-task", "task_id"),
        ClusterPolicy("source-unit", "source_id"),
        ClusterPolicy("user-unit", "user_id"),
        ClusterPolicy("domain-unit", "domain_id"),
    )
    with tempfile.TemporaryDirectory() as directory:
        registry_path = Path(directory) / "policy-registry.sqlite"
        ledger = PreregistrationLedger(registry_path)
        registered_manifest_id = ledger.register_manifest(manifest)
        registered_policy_ids = tuple(
            ledger.register_policy(registered_manifest_id, policy)
            for policy in policies
        )
        registered_evidence_id = ledger.record_evidence(
            registered_manifest_id, evidence
        )
        report = ledger.run_registered_audit(registered_evidence_id)
        counts_before_retry = ledger.counts()

        policy_retry_after_evidence = (
            ledger.register_policy(registered_manifest_id, policies[0])
            == registered_policy_ids[0]
        )
        evidence_retry = (
            ledger.record_evidence(registered_manifest_id, evidence)
            == registered_evidence_id
        )
        audit_retry = (
            ledger.run_registered_audit(registered_evidence_id).report_id
            == report.report_id
        )
        idempotent_retries = (
            policy_retry_after_evidence
            and evidence_retry
            and audit_retry
            and ledger.counts() == counts_before_retry
        )

        late_policy_rejected = False
        try:
            ledger.register_policy(
                registered_manifest_id,
                ClusterPolicy("late-source", "source_id", alpha=0.025),
            )
        except RuntimeError:
            late_policy_rejected = True

        immutable_update_rejected = False
        try:
            with ledger.conn:
                ledger.conn.execute(
                    "UPDATE policies SET payload_json=payload_json"
                )
        except sqlite3.IntegrityError:
            immutable_update_rejected = True
        immutable_delete_rejected = False
        try:
            with ledger.conn:
                ledger.conn.execute(
                    "DELETE FROM policies WHERE policy_id=?",
                    (registered_policy_ids[0],),
                )
        except sqlite3.IntegrityError:
            immutable_delete_rejected = True

        chain_verified = ledger.verify_chain()
        order_verified = ledger.verify_registration_order()
        stored_report_matches = asdict(ledger.load_report(report.report_id)) == asdict(
            report
        )
        ledger.close()

        reopened = PreregistrationLedger(registry_path)
        reopened_counts_preserved = reopened.counts() == counts_before_retry
        reopened_report_preserved = asdict(
            reopened.load_report(report.report_id)
        ) == asdict(report)
        reopened_chain_verified = reopened.verify_chain()
        reopened_order_verified = reopened.verify_registration_order()
        reopened.close()

        empty_registry = PreregistrationLedger(
            Path(directory) / "empty-policy-registry.sqlite"
        )
        empty_manifest_id = empty_registry.register_manifest(manifest)
        evidence_without_policy_rejected = False
        try:
            empty_registry.record_evidence(empty_manifest_id, evidence)
        except RuntimeError:
            evidence_without_policy_rejected = True
        empty_registry.close()

    by_axis = {result.axis: result for result in report.results}

    wrong_assignment_rejected = False
    wrong = derive_cluster_ids(manifest, "source_id")
    wrong["task-00"] = "source-b"
    try:
        audit_policy(manifest, evidence, policies[1], wrong)
    except ValueError:
        wrong_assignment_rejected = True

    duplicate_task_rejected = False
    try:
        validate_manifest(manifest + (manifest[0],))
    except ValueError:
        duplicate_task_rejected = True

    missing_lineage_rejected = False
    try:
        validate_manifest((replace(manifest[0], source_id=" "),) + manifest[1:])
    except ValueError:
        missing_lineage_rejected = True

    invalid_weights_rejected = False
    try:
        validate_manifest(tuple(replace(row, traffic_weight=0.1) for row in manifest))
    except ValueError:
        invalid_weights_rejected = True

    incomplete_evidence_rejected = False
    try:
        audit_policy(manifest, evidence[:-1], policies[1])
    except ValueError:
        incomplete_evidence_rejected = True

    reweighted_manifest = tuple(
        replace(row, traffic_weight=0.05 if index < 10 else 0.25)
        for index, row in enumerate(manifest)
    )
    original_policy_id = policy_id(policies[1], manifest_id(manifest))
    reweighted_policy_id = policy_id(
        policies[1], manifest_id(reweighted_manifest)
    )

    task = by_axis["task_id"]
    source = by_axis["source_id"]
    user = by_axis["user_id"]
    domain = by_axis["domain_id"]
    checks = (
        (
            "manifest identity ignores row ordering",
            manifest_id(manifest) == manifest_id(reversed(manifest)),
        ),
        ("task-level exact p-value is 79/4096", abs(task.p_value - 79 / 4096) < 1e-12),
        ("naive task policy passes fixed 0.05", task.fixed_alpha_pass),
        ("source lineage leaves only two clusters", source.cluster_count == 2),
        ("equal-source estimand cancels to zero", abs(source.equal_cluster_effect) < 1e-12),
        ("source-level exact p-value is 0.75", abs(source.p_value - 0.75) < 1e-12),
        (
            "user-level evidence does not pass",
            user.cluster_count == 4 and not user.fixed_alpha_pass,
        ),
        (
            "domain-level evidence does not pass",
            domain.cluster_count == 2 and not domain.fixed_alpha_pass,
        ),
        (
            "row-weighted effect is stable across policies",
            len(
                {
                    round(result.row_weighted_effect, 12)
                    for result in report.results
                }
            ) == 1,
        ),
        ("policy disagreement freezes selection", report.status == "FREEZE"),
        ("lineage-derived ids reject caller relabeling", wrong_assignment_rejected),
        ("duplicate task ids fail closed", duplicate_task_rejected),
        ("missing lineage fails closed", missing_lineage_rejected),
        ("invalid traffic weights fail closed", invalid_weights_rejected),
        ("incomplete evidence coverage fails closed", incomplete_evidence_rejected),
        (
            "changing frozen weights changes policy identity",
            original_policy_id != reweighted_policy_id,
        ),
        (
            "registry stores one manifest, four policies, evidence, report, and events",
            counts_before_retry == (1, 4, 1, 1, 7),
        ),
        (
            "evidence requires preregistered sensitivity axes",
            evidence_without_policy_rejected,
        ),
        ("new policy is rejected after evidence", late_policy_rejected),
        (
            "payload-identical policy retry remains idempotent after closure",
            policy_retry_after_evidence,
        ),
        ("retries do not append duplicate events", idempotent_retries),
        (
            "stored rows reject in-place UPDATE and DELETE",
            immutable_update_rejected and immutable_delete_rejected,
        ),
        ("event hash chain verifies", chain_verified),
        ("event replay verifies registration order", order_verified),
        ("stored report round-trips", stored_report_matches),
        (
            "close and reopen preserves counts, report, chain, and replay",
            reopened_counts_preserved
            and reopened_report_preserved
            and reopened_chain_verified
            and reopened_order_verified,
        ),
    )

    print("[1] provenance and policies are content-addressed before outcomes")
    print(
        f"    manifest={report.manifest_id} policies={len(registered_policy_ids)} "
        f"evidence={report.evidence_id} "
        f"report={report.report_id}"
    )
    print("[2] the same paired rows answer different questions")
    for result in report.results:
        print(
            f"    axis={result.axis:<9} tasks={result.task_count:2d} "
            f"clusters={result.cluster_count:2d} "
            f"row_effect={result.row_weighted_effect:+.6f} "
            f"cluster_effect={result.equal_cluster_effect:+.6f} "
            f"p={result.p_value:.6f} pass={result.fixed_alpha_pass}"
        )
    passing = ",".join(
        result.axis for result in report.results if result.fixed_alpha_pass
    )
    print("[3] disagreement is a stop signal, not a policy-selection menu")
    print(f"    status={report.status} passing_axes={passing}")
    print("[4] provenance contract rejects post-hoc relabeling")
    print(
        f"    wrong_assignment={wrong_assignment_rejected} "
        f"duplicate_task={duplicate_task_rejected} "
        f"missing_lineage={missing_lineage_rejected} "
        f"invalid_weights={invalid_weights_rejected} "
        f"incomplete_evidence={incomplete_evidence_rejected}"
    )
    print("[5] weights are part of policy identity")
    print(
        f"    original={original_policy_id} reweighted={reweighted_policy_id} "
        f"changed={original_policy_id != reweighted_policy_id}"
    )
    print("[6] durable preregistration closes before evidence")
    print(
        f"    events={counts_before_retry[-1]} policies={counts_before_retry[1]} "
        f"no_policy_evidence={evidence_without_policy_rejected} "
        f"late_policy={late_policy_rejected} "
        f"policy_retry={policy_retry_after_evidence} "
        f"immutable={immutable_update_rejected and immutable_delete_rejected}"
    )
    print(
        f"    chain={chain_verified} replay={order_verified} "
        f"idempotent={idempotent_retries} "
        f"reopened={reopened_counts_preserved and reopened_report_preserved}"
    )
    print("[7] structural self-check")
    for label, passed in checks:
        print(f"    {'PASS' if passed else 'FAIL'} | {label}")
    passed_count = sum(passed for _, passed in checks)
    print(f"SELF-CHECK: {passed_count}/{len(checks)} PASS")
    print(
        "takeaway: preregistered provenance derives clusters; disagreement "
        "freezes the gate; durable order is auditable after restart."
    )


if __name__ == "__main__":
    demo()
