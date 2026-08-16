#!/usr/bin/env python3
"""L2: cluster-aware trials, evaluator epochs, and alpha spending."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from itertools import product
import json
from pathlib import Path
import sqlite3
import tempfile
from typing import Any, Iterable


def canonical(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def digest(payload: Any) -> str:
    return hashlib.sha256(canonical(payload).encode()).hexdigest()


@dataclass(frozen=True)
class AnchorCase:
    task_id: str
    baseline_score: float
    critical_floor: float


@dataclass(frozen=True)
class DriftPolicy:
    max_mean_abs_drift: float = 0.03
    max_abs_drift: float = 0.08
    max_threshold_flips: int = 0


@dataclass(frozen=True)
class EvaluatorEpoch:
    epoch_id: str
    evaluator_version: str
    suite_id: str
    anchors: tuple[AnchorCase, ...]
    parent_epoch: str | None
    reason: str


@dataclass(frozen=True)
class DriftReport:
    report_id: str
    report_no: int
    epoch_id: str
    status: str
    mean_abs_drift: float | None
    max_abs_drift: float | None
    threshold_flips: int | None
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class TrialEvidence:
    epoch_id: str
    parent_id: str
    candidate_id: str
    paired_deltas: tuple[float, ...]
    cluster_ids: tuple[str, ...]


@dataclass(frozen=True)
class TrialDecision:
    evidence_id: str
    trial_no: int
    epoch_id: str
    task_count: int
    cluster_count: int
    task_mean_delta: float
    cluster_mean_delta: float
    task_level_p_value: float
    cluster_p_value: float
    alpha_t: float
    naive_cluster_alpha_pass: bool
    decision: str


def alpha_at(trial_no: int, alpha_total: float = 0.05) -> float:
    """A summable schedule: sum_t alpha_total/[t(t+1)] = alpha_total."""
    if trial_no < 1 or not 0.0 < alpha_total < 1.0:
        raise ValueError("invalid alpha-spending configuration")
    return alpha_total / (trial_no * (trial_no + 1))


def exact_sign_flip_pvalue(deltas: tuple[float, ...]) -> float:
    """One-sided paired randomization p-value under sign symmetry of the null."""
    if not 1 <= len(deltas) <= 20:
        raise ValueError("exact toy test supports 1..20 paired deltas")
    if any(not -1.0 <= value <= 1.0 for value in deltas):
        raise ValueError("paired deltas must be finite scores in [-1, 1]")
    observed = sum(deltas)
    extreme = 0
    for signs in product((-1.0, 1.0), repeat=len(deltas)):
        statistic = sum(sign * value for sign, value in zip(signs, deltas))
        if statistic >= observed - 1e-12:
            extreme += 1
    return extreme / (2 ** len(deltas))


def cluster_means(
    deltas: tuple[float, ...], cluster_ids: tuple[str, ...]
) -> tuple[float, ...]:
    """Collapse correlated task deltas to equal-weight cluster means."""
    if len(cluster_ids) != len(deltas):
        raise ValueError("every paired delta requires exactly one cluster id")
    normalized = tuple(str(cluster_id).strip() for cluster_id in cluster_ids)
    if any(not cluster_id for cluster_id in normalized):
        raise ValueError("cluster ids must be non-empty")
    grouped: dict[str, list[float]] = {}
    for cluster_id, delta in zip(normalized, deltas):
        grouped.setdefault(cluster_id, []).append(delta)
    return tuple(
        sum(grouped[cluster_id]) / len(grouped[cluster_id])
        for cluster_id in sorted(grouped)
    )


class GovernanceLedger:
    """Append-only evaluator facts plus active epoch and global trial counter."""

    def __init__(self, path: Path, policy: DriftPolicy = DriftPolicy()) -> None:
        self.policy = policy
        self.conn = sqlite3.connect(path)
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.conn.execute("PRAGMA journal_mode = WAL")
        self._create_schema()

    def _create_schema(self) -> None:
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS evaluator_epochs(
                epoch_id TEXT PRIMARY KEY,
                payload_json TEXT NOT NULL,
                payload_sha256 TEXT NOT NULL CHECK(length(payload_sha256)=64)
            );
            CREATE TABLE IF NOT EXISTS drift_reports(
                report_id TEXT PRIMARY KEY,
                report_no INTEGER NOT NULL UNIQUE,
                epoch_id TEXT NOT NULL REFERENCES evaluator_epochs(epoch_id),
                status TEXT NOT NULL CHECK(status IN ('VALID','FREEZE')),
                payload_json TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS candidate_trials(
                evidence_id TEXT PRIMARY KEY,
                trial_no INTEGER NOT NULL UNIQUE,
                epoch_id TEXT NOT NULL REFERENCES evaluator_epochs(epoch_id),
                payload_json TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS rebaselines(
                record_id TEXT PRIMARY KEY,
                old_epoch TEXT NOT NULL REFERENCES evaluator_epochs(epoch_id),
                new_epoch TEXT NOT NULL UNIQUE REFERENCES evaluator_epochs(epoch_id),
                payload_json TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS governance_events(
                seq INTEGER PRIMARY KEY,
                kind TEXT NOT NULL,
                subject_id TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                prev_hash TEXT NOT NULL,
                event_hash TEXT NOT NULL UNIQUE
            );
            CREATE TABLE IF NOT EXISTS control_state(
                singleton INTEGER PRIMARY KEY CHECK(singleton=1),
                active_epoch TEXT NOT NULL REFERENCES evaluator_epochs(epoch_id),
                next_trial INTEGER NOT NULL CHECK(next_trial >= 1)
            );
            """
        )
        immutable_tables = (
            "evaluator_epochs", "drift_reports", "candidate_trials",
            "rebaselines", "governance_events",
        )
        for table in immutable_tables:
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
    def active_epoch_id(self) -> str:
        row = self.conn.execute(
            "SELECT active_epoch FROM control_state WHERE singleton=1"
        ).fetchone()
        if row is None:
            raise RuntimeError("governance state is not initialized")
        return str(row[0])

    @property
    def next_trial(self) -> int:
        row = self.conn.execute(
            "SELECT next_trial FROM control_state WHERE singleton=1"
        ).fetchone()
        if row is None:
            raise RuntimeError("governance state is not initialized")
        return int(row[0])

    @staticmethod
    def _validate_anchors(anchors: Iterable[AnchorCase]) -> tuple[AnchorCase, ...]:
        items = tuple(anchors)
        ids = [item.task_id for item in items]
        if not items or len(ids) != len(set(ids)) or any(not task_id for task_id in ids):
            raise ValueError("anchor suite must contain unique, non-empty task ids")
        for item in items:
            if not 0.0 <= item.baseline_score <= 1.0:
                raise ValueError("anchor baseline score must be in [0, 1]")
            if not 0.0 <= item.critical_floor <= 1.0:
                raise ValueError("anchor critical floor must be in [0, 1]")
        return items

    @staticmethod
    def _epoch_payload(
        evaluator_version: str, suite_id: str, anchors: tuple[AnchorCase, ...],
        parent_epoch: str | None, reason: str,
    ) -> dict[str, Any]:
        return {
            "evaluator_version": evaluator_version,
            "suite_id": suite_id,
            "anchors": [asdict(item) for item in anchors],
            "parent_epoch": parent_epoch,
            "reason": reason,
        }

    @staticmethod
    def _epoch_from_json(epoch_id: str, payload_json: str) -> EvaluatorEpoch:
        payload = json.loads(payload_json)
        return EvaluatorEpoch(
            epoch_id=epoch_id,
            evaluator_version=payload["evaluator_version"],
            suite_id=payload["suite_id"],
            anchors=tuple(AnchorCase(**item) for item in payload["anchors"]),
            parent_epoch=payload["parent_epoch"],
            reason=payload["reason"],
        )

    def _load_epoch(self, epoch_id: str) -> EvaluatorEpoch:
        row = self.conn.execute(
            "SELECT payload_json FROM evaluator_epochs WHERE epoch_id=?", (epoch_id,)
        ).fetchone()
        if row is None:
            raise RuntimeError("unknown evaluator epoch")
        return self._epoch_from_json(epoch_id, str(row[0]))

    def _insert_epoch(
        self, evaluator_version: str, suite_id: str, anchors: tuple[AnchorCase, ...],
        parent_epoch: str | None, reason: str,
    ) -> str:
        if not evaluator_version or not suite_id or not reason.strip():
            raise ValueError("evaluator version, suite id, and reason are required")
        payload = self._epoch_payload(
            evaluator_version, suite_id, anchors, parent_epoch, reason.strip()
        )
        payload_hash = digest(payload)
        epoch_id = f"epoch-{payload_hash[:12]}"
        self.conn.execute(
            "INSERT INTO evaluator_epochs VALUES (?,?,?)",
            (epoch_id, canonical(payload), payload_hash),
        )
        return epoch_id

    def _append_event(self, kind: str, subject_id: str, details: dict[str, Any]) -> str:
        row = self.conn.execute(
            "SELECT seq, event_hash FROM governance_events ORDER BY seq DESC LIMIT 1"
        ).fetchone()
        seq = 1 if row is None else int(row[0]) + 1
        prev_hash = "GENESIS" if row is None else str(row[1])
        payload = {"seq": seq, "kind": kind, "subject_id": subject_id, "details": details}
        event_hash = digest({"prev_hash": prev_hash, "payload": payload})
        self.conn.execute(
            "INSERT INTO governance_events VALUES (?,?,?,?,?,?)",
            (seq, kind, subject_id, canonical(payload), prev_hash, event_hash),
        )
        return event_hash

    def initialize(
        self, evaluator_version: str, suite_id: str, anchors: Iterable[AnchorCase]
    ) -> str:
        if self.conn.execute("SELECT 1 FROM control_state").fetchone() is not None:
            raise RuntimeError("governance state is already initialized")
        checked = self._validate_anchors(anchors)
        with self.conn:
            epoch_id = self._insert_epoch(
                evaluator_version, suite_id, checked, None, "initial audited baseline"
            )
            self.conn.execute("INSERT INTO control_state VALUES (1,?,1)", (epoch_id,))
            self._append_event("EPOCH_START", epoch_id, {"parent_epoch": None})
        return epoch_id

    def record_drift(
        self, evaluator_version: str, suite_id: str,
        observed_scores: Iterable[tuple[str, float]],
    ) -> DriftReport:
        epoch = self._load_epoch(self.active_epoch_id)
        observed = tuple((str(task_id), float(score)) for task_id, score in observed_scores)
        observed_ids = [task_id for task_id, _ in observed]
        reasons: list[str] = []
        if evaluator_version != epoch.evaluator_version:
            reasons.append("evaluator version changed")
        if suite_id != epoch.suite_id:
            reasons.append("anchor suite identity changed")
        if len(observed_ids) != len(set(observed_ids)):
            reasons.append("duplicate anchor ids")
        baseline = {item.task_id: item for item in epoch.anchors}
        if set(observed_ids) != set(baseline):
            reasons.append("anchor coverage mismatch")

        mean_abs: float | None = None
        max_abs: float | None = None
        threshold_flips: int | None = None
        if len(observed_ids) == len(set(observed_ids)) and set(observed_ids) == set(baseline):
            current = dict(observed)
            if any(not 0.0 <= score <= 1.0 for score in current.values()):
                reasons.append("anchor score outside [0, 1]")
            else:
                absolute = [
                    abs(current[task_id] - item.baseline_score)
                    for task_id, item in baseline.items()
                ]
                mean_abs = sum(absolute) / len(absolute)
                max_abs = max(absolute)
                threshold_flips = sum(
                    (item.baseline_score >= item.critical_floor)
                    != (current[task_id] >= item.critical_floor)
                    for task_id, item in baseline.items()
                )
                if mean_abs > self.policy.max_mean_abs_drift:
                    reasons.append("mean absolute anchor drift exceeded")
                if max_abs > self.policy.max_abs_drift:
                    reasons.append("maximum anchor drift exceeded")
                if threshold_flips > self.policy.max_threshold_flips:
                    reasons.append("critical threshold flip detected")

        status = "FREEZE" if reasons else "VALID"
        report_no = int(self.conn.execute(
            "SELECT COALESCE(MAX(report_no),0)+1 FROM drift_reports"
        ).fetchone()[0])
        report_core = {
            "report_no": report_no,
            "epoch_id": epoch.epoch_id,
            "observed_evaluator_version": evaluator_version,
            "observed_suite_id": suite_id,
            "observed_scores": observed,
            "status": status,
            "mean_abs_drift": mean_abs,
            "max_abs_drift": max_abs,
            "threshold_flips": threshold_flips,
            "reasons": reasons,
        }
        report_id = f"drift-{digest(report_core)[:12]}"
        report = DriftReport(
            report_id, report_no, epoch.epoch_id, status,
            mean_abs, max_abs, threshold_flips, tuple(reasons),
        )
        with self.conn:
            self.conn.execute(
                "INSERT INTO drift_reports VALUES (?,?,?,?,?)",
                (report_id, report_no, epoch.epoch_id, status, canonical(report_core)),
            )
            self._append_event(
                "DRIFT_CHECK", report_id, {"epoch_id": epoch.epoch_id, "status": status}
            )
        return report

    def _latest_drift(self, epoch_id: str) -> tuple[str, str] | None:
        row = self.conn.execute(
            "SELECT report_id,status FROM drift_reports WHERE epoch_id=? "
            "ORDER BY report_no DESC LIMIT 1", (epoch_id,)
        ).fetchone()
        return None if row is None else (str(row[0]), str(row[1]))

    def submit_trial(
        self, evidence: TrialEvidence, alpha_total: float = 0.05
    ) -> TrialDecision:
        if evidence.epoch_id != self.active_epoch_id:
            raise RuntimeError("candidate evidence belongs to an inactive evaluator epoch")
        latest = self._latest_drift(evidence.epoch_id)
        if latest is None or latest[1] != "VALID":
            raise RuntimeError("promotion is frozen until the active evaluator passes anchors")
        deltas = tuple(float(value) for value in evidence.paired_deltas)
        task_level_p_value = exact_sign_flip_pvalue(deltas)
        cluster_ids = tuple(str(value).strip() for value in evidence.cluster_ids)
        clustered = cluster_means(deltas, cluster_ids)
        cluster_p_value = exact_sign_flip_pvalue(clustered)
        trial_no = self.next_trial
        alpha_t = alpha_at(trial_no, alpha_total)
        task_mean_delta = sum(deltas) / len(deltas)
        cluster_mean_delta = sum(clustered) / len(clustered)
        decision = (
            "PROMOTE"
            if cluster_mean_delta > 0.0 and cluster_p_value <= alpha_t
            else "REJECT"
        )
        evidence_core = {
            "epoch_id": evidence.epoch_id,
            "parent_id": evidence.parent_id,
            "candidate_id": evidence.candidate_id,
            "paired_deltas": deltas,
            "cluster_ids": cluster_ids,
        }
        evidence_id = f"trial-{digest(evidence_core)[:12]}"
        result = TrialDecision(
            evidence_id=evidence_id,
            trial_no=trial_no,
            epoch_id=evidence.epoch_id,
            task_count=len(deltas),
            cluster_count=len(clustered),
            task_mean_delta=task_mean_delta,
            cluster_mean_delta=cluster_mean_delta,
            task_level_p_value=task_level_p_value,
            cluster_p_value=cluster_p_value,
            alpha_t=alpha_t,
            naive_cluster_alpha_pass=(
                cluster_mean_delta > 0.0 and cluster_p_value <= alpha_total
            ),
            decision=decision,
        )
        payload = {"evidence": evidence_core, "decision": asdict(result)}
        with self.conn:
            self.conn.execute(
                "INSERT INTO candidate_trials VALUES (?,?,?,?)",
                (evidence_id, trial_no, evidence.epoch_id, canonical(payload)),
            )
            self._append_event(
                "CANDIDATE_TRIAL", evidence_id,
                {
                    "trial_no": trial_no,
                    "cluster_count": len(clustered),
                    "cluster_p_value": cluster_p_value,
                    "decision": decision,
                    "alpha_t": alpha_t,
                },
            )
            self.conn.execute(
                "UPDATE control_state SET next_trial=? WHERE singleton=1",
                (trial_no + 1,),
            )
        return result

    def rebaseline(
        self, new_evaluator_version: str, new_suite_id: str,
        new_anchors: Iterable[AnchorCase], reason: str,
        approval_ids: Iterable[str],
    ) -> str:
        old_epoch = self._load_epoch(self.active_epoch_id)
        latest = self._latest_drift(old_epoch.epoch_id)
        if latest is None or latest[1] != "FREEZE":
            raise RuntimeError("re-baseline requires a recorded evaluator freeze")
        approvals = tuple(sorted(set(item.strip() for item in approval_ids if item.strip())))
        if len(approvals) < 2:
            raise ValueError("re-baseline requires two distinct approval ids")
        if len(reason.strip()) < 12:
            raise ValueError("re-baseline requires a substantive reason")
        checked = self._validate_anchors(new_anchors)
        old_by_id = {item.task_id: item for item in old_epoch.anchors}
        new_by_id = {item.task_id: item for item in checked}
        if set(new_by_id) != set(old_by_id):
            raise ValueError("toy re-baseline must preserve anchor task coverage")
        for task_id, old in old_by_id.items():
            new = new_by_id[task_id]
            if new.critical_floor != old.critical_floor:
                raise ValueError("critical floors cannot change during re-baseline")
            if new.baseline_score < new.critical_floor:
                raise ValueError("re-baseline cannot normalize a critical anchor failure")

        with self.conn:
            new_epoch = self._insert_epoch(
                new_evaluator_version, new_suite_id, checked,
                old_epoch.epoch_id, reason.strip(),
            )
            record_core = {
                "old_epoch": old_epoch.epoch_id,
                "new_epoch": new_epoch,
                "source_drift_report": latest[0],
                "reason": reason.strip(),
                "approval_ids": approvals,
                "trial_counter_continues_at": self.next_trial,
            }
            record_id = f"rebaseline-{digest(record_core)[:12]}"
            self.conn.execute(
                "INSERT INTO rebaselines VALUES (?,?,?,?)",
                (record_id, old_epoch.epoch_id, new_epoch, canonical(record_core)),
            )
            self._append_event(
                "REBASELINE", record_id,
                {"old_epoch": old_epoch.epoch_id, "new_epoch": new_epoch},
            )
            self.conn.execute(
                "UPDATE control_state SET active_epoch=? WHERE singleton=1", (new_epoch,)
            )
        return new_epoch

    def verify_chain(self) -> bool:
        expected_seq = 1
        expected_prev = "GENESIS"
        rows = self.conn.execute(
            "SELECT seq,payload_json,prev_hash,event_hash "
            "FROM governance_events ORDER BY seq"
        )
        for seq, payload_json, prev_hash, event_hash in rows:
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
        """Replay state-changing events and compare them with mutable control_state."""
        active_epoch: str | None = None
        expected_trial = 1
        rows = self.conn.execute(
            "SELECT payload_json FROM governance_events ORDER BY seq"
        )
        for (payload_json,) in rows:
            payload = json.loads(payload_json)
            kind = payload["kind"]
            details = payload["details"]
            if kind == "EPOCH_START":
                if active_epoch is not None:
                    return False
                active_epoch = payload["subject_id"]
            elif kind == "REBASELINE":
                if details["old_epoch"] != active_epoch:
                    return False
                active_epoch = details["new_epoch"]
            elif kind == "CANDIDATE_TRIAL":
                if int(details["trial_no"]) != expected_trial:
                    return False
                expected_trial += 1
        row = self.conn.execute(
            "SELECT active_epoch,next_trial FROM control_state WHERE singleton=1"
        ).fetchone()
        return row is not None and (str(row[0]), int(row[1])) == (
            active_epoch, expected_trial,
        )


def demo() -> None:
    anchors_v1 = (
        AnchorCase("safety", 0.95, 0.80),
        AnchorCase("helpfulness", 0.75, 0.70),
        AnchorCase("format", 0.88, 0.80),
    )
    stable_v1 = (("safety", 0.94), ("helpfulness", 0.76), ("format", 0.87))
    recovered_v1 = (("safety", 0.95), ("helpfulness", 0.74), ("format", 0.87))
    anchors_v2 = (
        AnchorCase("safety", 0.90, 0.80),
        AnchorCase("helpfulness", 0.71, 0.70),
        AnchorCase("format", 0.83, 0.80),
    )

    with tempfile.TemporaryDirectory() as tmp:
        store = GovernanceLedger(Path(tmp) / "governance.sqlite")
        epoch_v1 = store.initialize("judge-v1", "anchor-suite-q3-v1", anchors_v1)

        stable = store.record_drift("judge-v1", "anchor-suite-q3-v1", stable_v1)
        missing = store.record_drift(
            "judge-v1", "anchor-suite-q3-v1", stable_v1[:-1]
        )
        recovered = store.record_drift(
            "judge-v1", "anchor-suite-q3-v1", recovered_v1
        )

        neutral = store.submit_trial(TrialEvidence(
            epoch_v1, "parent-v7", "candidate-neutral",
            (0.01,) * 6 + (-0.01,) * 6,
            tuple(f"neutral-{index:02d}" for index in range(12)),
        ))
        pseudo_replicated = store.submit_trial(TrialEvidence(
            epoch_v1, "parent-v7", "candidate-pseudo-replicated",
            (0.01,) * 10 + (-0.01,) * 2,
            ("source-a",) * 6 + ("source-b",) * 6,
        ))
        independent_marginal = store.submit_trial(TrialEvidence(
            epoch_v1, "parent-v7", "candidate-independent-marginal",
            (0.01,) * 10 + (-0.01,) * 2,
            tuple(f"marginal-{index:02d}" for index in range(12)),
        ))

        trial_before_invalid_attempt = store.next_trial
        invalid_clusters_rejected = 0
        invalid_evidence = (
            TrialEvidence(
                epoch_v1, "parent-v7", "candidate-invalid-clusters",
                (0.01, 0.02), ("source-a",),
            ),
            TrialEvidence(
                epoch_v1, "parent-v7", "candidate-empty-cluster",
                (0.01, 0.02), ("source-a", "  "),
            ),
        )
        for invalid in invalid_evidence:
            try:
                store.submit_trial(invalid)
            except ValueError:
                invalid_clusters_rejected += 1
        invalid_attempt_spent_nothing = (
            store.next_trial == trial_before_invalid_attempt
        )

        drifted = store.record_drift(
            "judge-v2", "anchor-suite-q3-v1",
            tuple((item.task_id, item.baseline_score) for item in anchors_v2),
        )
        trial_before_frozen_attempt = store.next_trial
        frozen_attempt_rejected = False
        try:
            store.submit_trial(TrialEvidence(
                epoch_v1, "parent-v7", "candidate-during-drift", (0.02,) * 12,
                tuple(f"frozen-{index:02d}" for index in range(12)),
            ))
        except RuntimeError:
            frozen_attempt_rejected = True
        frozen_attempt_spent_nothing = store.next_trial == trial_before_frozen_attempt

        duplicate_approval_rejected = False
        try:
            store.rebaseline(
                "judge-v2", "anchor-suite-q3-audited-v2", anchors_v2,
                "audited scale shift with preserved critical floors",
                ("review-a", "review-a"),
            )
        except ValueError:
            duplicate_approval_rejected = True

        unsafe_rebaseline_rejected = False
        unsafe_anchors = (
            AnchorCase("safety", 0.79, 0.80),
            AnchorCase("helpfulness", 0.71, 0.70),
            AnchorCase("format", 0.83, 0.80),
        )
        try:
            store.rebaseline(
                "judge-v2", "anchor-suite-q3-audited-v2", unsafe_anchors,
                "attempt to normalize a critical anchor failure",
                ("review-a", "review-b"),
            )
        except ValueError:
            unsafe_rebaseline_rejected = True

        epoch_v2 = store.rebaseline(
            "judge-v2", "anchor-suite-q3-audited-v2", anchors_v2,
            "audited scale shift with preserved task coverage and critical floors",
            ("review-a", "review-b"),
        )
        old_evidence_rejected = False
        try:
            store.submit_trial(TrialEvidence(
                epoch_v1, "parent-v7", "candidate-old-evidence", (0.02,) * 12,
                tuple(f"stale-{index:02d}" for index in range(12)),
            ))
        except RuntimeError:
            old_evidence_rejected = True

        stable_v2 = store.record_drift(
            "judge-v2", "anchor-suite-q3-audited-v2",
            tuple((item.task_id, item.baseline_score) for item in anchors_v2),
        )
        robust = store.submit_trial(TrialEvidence(
            epoch_v2, "parent-v7", "candidate-robust", (0.02,) * 12,
            (
                "robust-0", "robust-0", "robust-1", "robust-1",
                "robust-2", "robust-2", "robust-3", "robust-4",
                "robust-5", "robust-6", "robust-7", "robust-8",
            ),
        ))

        immutable_rejected = False
        try:
            with store.conn:
                store.conn.execute(
                    "UPDATE evaluator_epochs SET payload_json='{}' WHERE epoch_id=?",
                    (epoch_v1,),
                )
        except sqlite3.IntegrityError:
            immutable_rejected = True

        epoch_count = int(store.conn.execute(
            "SELECT COUNT(*) FROM evaluator_epochs"
        ).fetchone()[0])
        trial_count = int(store.conn.execute(
            "SELECT COUNT(*) FROM candidate_trials"
        ).fetchone()[0])
        spent_10000 = sum(alpha_at(index) for index in range(1, 10001))
        checks = (
            ("stable anchors validate the evaluator", stable.status == "VALID"),
            ("missing anchor coverage freezes promotion", missing.status == "FREEZE"),
            ("a complete fresh check can recover", recovered.status == "VALID"),
            ("null-like first candidate is rejected", neutral.decision == "REJECT"),
            (
                "pseudo-replicated tasks look significant if treated as independent",
                pseudo_replicated.task_level_p_value < 0.05,
            ),
            (
                "two source clusters provide only p=0.25 evidence",
                abs(pseudo_replicated.cluster_p_value - 0.25) < 1e-12,
            ),
            (
                "cluster-aware gate rejects pseudo-replication",
                pseudo_replicated.decision == "REJECT",
            ),
            (
                "independent marginal evidence passes naive cluster alpha",
                independent_marginal.naive_cluster_alpha_pass,
            ),
            (
                "trial-3 alpha spending rejects independent marginal evidence",
                independent_marginal.decision == "REJECT",
            ),
            (
                "invalid cluster metadata is fail-closed before spending",
                invalid_clusters_rejected == 2 and invalid_attempt_spent_nothing,
            ),
            ("the infinite-style alpha schedule stays within budget", spent_10000 < 0.05),
            ("evaluator version drift freezes promotion", drifted.status == "FREEZE"),
            (
                "a frozen attempt neither admits nor spends a trial",
                frozen_attempt_rejected and frozen_attempt_spent_nothing,
            ),
            ("duplicate approval ids cannot re-baseline", duplicate_approval_rejected),
            ("re-baseline cannot normalize a critical failure", unsafe_rebaseline_rejected),
            ("old and new evaluator epochs both remain", epoch_count == 2),
            ("old-epoch candidate evidence is stale", old_evidence_rejected),
            ("new epoch must pass its own anchors", stable_v2.status == "VALID"),
            ("fresh strong evidence contains nine independent clusters", robust.cluster_count == 9),
            (
                "strong fresh evidence passes the trial-4 threshold",
                robust.decision == "PROMOTE"
                and robust.cluster_p_value < robust.alpha_t,
            ),
            ("only four admitted trials consumed alpha", trial_count == 4),
            ("immutable evaluator facts reject UPDATE", immutable_rejected),
            ("governance event hash chain verifies", store.verify_chain()),
            ("mutable control state matches event replay", store.verify_projection()),
        )

        print("[1] anchor coverage is a fail-closed precondition")
        print(
            f"    stable={stable.status} missing={missing.status} "
            f"recovered={recovered.status}"
        )
        print("[2] task replication is not independent evidence")
        print(
            f"    trial=1 clusters={neutral.cluster_count} "
            f"p_cluster={neutral.cluster_p_value:.6f} alpha={neutral.alpha_t:.6f} "
            f"decision={neutral.decision}"
        )
        print(
            f"    trial=2 tasks={pseudo_replicated.task_count} "
            f"clusters={pseudo_replicated.cluster_count} "
            f"p_task={pseudo_replicated.task_level_p_value:.6f} "
            f"p_cluster={pseudo_replicated.cluster_p_value:.6f} "
            f"decision={pseudo_replicated.decision}"
        )
        print("[3] cluster-valid trials still share one alpha budget")
        print(
            f"    trial=3 p_cluster={independent_marginal.cluster_p_value:.6f} "
            f"naive_0.05={independent_marginal.naive_cluster_alpha_pass} "
            f"alpha={independent_marginal.alpha_t:.6f} "
            f"decision={independent_marginal.decision}"
        )
        print("[4] evaluator drift freezes admission")
        print(
            f"    status={drifted.status} mean_abs={drifted.mean_abs_drift:.3f} "
            f"frozen_attempt_rejected={frozen_attempt_rejected}"
        )
        print("[5] explicit re-baseline creates a new evidence epoch")
        print(
            f"    epochs={epoch_count} duplicate_approval_rejected="
            f"{duplicate_approval_rejected} old_evidence_rejected={old_evidence_rejected}"
        )
        print("[6] fresh evidence resumes under the global trial counter")
        print(
            f"    trial={robust.trial_no} clusters={robust.cluster_count} "
            f"p_cluster={robust.cluster_p_value:.6f} "
            f"alpha={robust.alpha_t:.6f} decision={robust.decision}"
        )
        print("[7] structural self-check")
        for label, passed in checks:
            print(f"    {'PASS' if passed else 'FAIL'} | {label}")
        passed_count = sum(passed for _, passed in checks)
        print(f"SELF-CHECK: {passed_count}/{len(checks)} PASS")
        print(
            "takeaway: clusters define independent evidence; evaluator change freezes "
            "promotion; neither replication nor re-baselining resets the error budget."
        )
        store.close()


if __name__ == "__main__":
    demo()
