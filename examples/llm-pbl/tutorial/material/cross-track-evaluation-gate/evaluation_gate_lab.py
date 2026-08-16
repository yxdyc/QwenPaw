#!/usr/bin/env python3
"""L0: a paired candidate-parent promotion gate with hidden sentinels."""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import random
from statistics import mean


@dataclass(frozen=True)
class EvalRow:
    task_id: str
    split: str
    critical: bool
    parent_score: float
    candidate_score: float
    parent_seed: int
    candidate_seed: int

    @property
    def delta(self) -> float:
        return self.candidate_score - self.parent_score


@dataclass(frozen=True)
class EvalBundle:
    parent_id: str
    candidate_id: str
    dataset_snapshot: str
    evaluator_version: str
    environment_version: str
    parent_cost: float
    candidate_cost: float
    rows: tuple[EvalRow, ...]


@dataclass(frozen=True)
class GateConfig:
    min_total: int = 20
    min_hidden: int = 4
    min_effect: float = 0.01
    hidden_mean_floor: float = -0.005
    critical_delta_floor: float = -0.02
    max_cost_ratio: float = 1.15
    bootstrap_draws: int = 8_000


@dataclass(frozen=True)
class PromotionRecord:
    record_id: str
    parent_id: str
    candidate_id: str
    evidence_id: str
    decision: str
    reasons: tuple[str, ...]
    metrics: tuple[tuple[str, float], ...]
    rollback_target: str


def evidence_errors(bundle: EvalBundle, cfg: GateConfig) -> list[str]:
    errors: list[str] = []
    required_ids = (bundle.parent_id, bundle.candidate_id, bundle.dataset_snapshot,
                    bundle.evaluator_version, bundle.environment_version)
    if any(not item.strip() for item in required_ids):
        errors.append("missing version or lineage identity")
    keys = [(row.task_id, row.parent_seed) for row in bundle.rows]
    if len(keys) != len(set(keys)):
        errors.append("duplicate task/seed pair")
    if len(bundle.rows) < cfg.min_total:
        errors.append(f"coverage {len(bundle.rows)} < {cfg.min_total}")
    hidden = [row for row in bundle.rows if row.split == "hidden"]
    if len(hidden) < cfg.min_hidden:
        errors.append(f"hidden coverage {len(hidden)} < {cfg.min_hidden}")
    for row in bundle.rows:
        if row.split not in {"public", "hidden"}:
            errors.append(f"unknown split: {row.task_id}")
        if row.parent_seed != row.candidate_seed:
            errors.append(f"unpaired seed: {row.task_id}")
        if not (0.0 <= row.parent_score <= 1.0 and 0.0 <= row.candidate_score <= 1.0):
            errors.append(f"score out of range: {row.task_id}")
        if row.critical and row.split != "hidden":
            errors.append(f"critical task is not hidden: {row.task_id}")
    if bundle.parent_cost <= 0 or bundle.candidate_cost <= 0:
        errors.append("cost must be positive")
    return errors


def paired_bootstrap_ci(deltas: list[float], draws: int, seed: int = 20260814) -> tuple[float, float]:
    """Percentile CI over paired task deltas; task pairs are resampled together."""
    rng = random.Random(seed)
    n = len(deltas)
    samples = sorted(mean(rng.choice(deltas) for _ in range(n)) for _ in range(draws))
    return samples[int(0.025 * draws)], samples[min(draws - 1, int(0.975 * draws))]


def make_record(bundle: EvalBundle, decision: str, reasons: list[str], metrics: dict[str, float]) -> PromotionRecord:
    evidence_id = ":".join((bundle.dataset_snapshot, bundle.evaluator_version, bundle.environment_version))
    payload = {
        "parent": bundle.parent_id,
        "candidate": bundle.candidate_id,
        "evidence": evidence_id,
        "decision": decision,
        "reasons": reasons,
        "metrics": sorted(metrics.items()),
        "rollback_target": bundle.parent_id,
    }
    record_id = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]
    return PromotionRecord(record_id, bundle.parent_id, bundle.candidate_id, evidence_id, decision,
                           tuple(reasons), tuple(sorted(metrics.items())), bundle.parent_id)


def run_gate(bundle: EvalBundle, cfg: GateConfig) -> PromotionRecord:
    errors = evidence_errors(bundle, cfg)
    if errors:
        return make_record(bundle, "REJECT", [f"invalid evidence: {error}" for error in errors],
                           {"n": float(len(bundle.rows))})

    deltas = [row.delta for row in bundle.rows]
    hidden = [row for row in bundle.rows if row.split == "hidden"]
    lower, upper = paired_bootstrap_ci(deltas, cfg.bootstrap_draws)
    hidden_mean = mean(row.delta for row in hidden)
    cost_ratio = bundle.candidate_cost / bundle.parent_cost
    reasons: list[str] = []
    if lower < cfg.min_effect:
        reasons.append(f"gain lower bound {lower:.3f} < {cfg.min_effect:.3f}")
    if hidden_mean < cfg.hidden_mean_floor:
        reasons.append(f"hidden mean {hidden_mean:.3f} < {cfg.hidden_mean_floor:.3f}")
    for row in hidden:
        if row.critical and row.delta < cfg.critical_delta_floor:
            reasons.append(f"critical regression {row.task_id}: {row.delta:.3f}")
    if cost_ratio > cfg.max_cost_ratio:
        reasons.append(f"cost ratio {cost_ratio:.3f} > {cfg.max_cost_ratio:.3f}")
    metrics = {
        "cost_ratio": cost_ratio,
        "gain_ci_high": upper,
        "gain_ci_low": lower,
        "hidden_mean": hidden_mean,
        "mean_gain": mean(deltas),
        "n": float(len(deltas)),
    }
    return make_record(bundle, "REJECT" if reasons else "PROMOTE", reasons or ["all gates passed"], metrics)


def synthetic_bundle(candidate_id: str, public: list[float], hidden: list[float], cost: float) -> EvalBundle:
    deltas = [("public", delta) for delta in public] + [("hidden", delta) for delta in hidden]
    rows = []
    for i, (split, delta) in enumerate(deltas):
        base = 0.56 + 0.01 * ((i * 7) % 9)
        rows.append(EvalRow(f"{split}-{i:02d}", split, split == "hidden" and i == 22,
                            base, base + delta, 1000 + i, 1000 + i))
    return EvalBundle("model-parent-v7", candidate_id, "evalset-2026-08-14", "evaluator-v3",
                      "sandbox-v5", 1.0, cost, tuple(rows))


def show(label: str, record: PromotionRecord) -> None:
    metrics = dict(record.metrics)
    print(f"[{label}] {record.decision} | record={record.record_id}")
    if "mean_gain" in metrics:
        print(f"  mean={metrics['mean_gain']:+.3f}  95% CI=[{metrics['gain_ci_low']:+.3f}, "
              f"{metrics['gain_ci_high']:+.3f}]  hidden={metrics['hidden_mean']:+.3f}  "
              f"cost={metrics['cost_ratio']:.2f}x")
    for reason in record.reasons:
        print(f"  - {reason}")
    print(f"  rollback_target={record.rollback_target}")


def main() -> None:
    cfg = GateConfig()
    headline = synthetic_bundle("candidate-headline", [0.08] * 20, [-0.03, -0.02, -0.18, -0.01], 1.05)
    robust = synthetic_bundle("candidate-robust", [0.03, 0.04, 0.05, 0.04, 0.03] * 4,
                              [0.02, 0.03, 0.04, 0.025], 1.10)
    malformed = replace(robust, candidate_id="candidate-malformed", rows=robust.rows + (robust.rows[0],))

    records = [run_gate(headline, cfg), run_gate(robust, cfg), run_gate(malformed, cfg)]
    show("public winner, hidden regression", records[0])
    show("robust paired gain", records[1])
    show("duplicate evidence", records[2])

    checks = {
        "headline mean is positive": dict(records[0].metrics)["mean_gain"] > 0,
        "headline statistical gate alone passes": dict(records[0].metrics)["gain_ci_low"] >= cfg.min_effect,
        "hidden regression blocks promotion": records[0].decision == "REJECT" and len(records[0].reasons) == 2,
        "robust candidate promotes": records[1].decision == "PROMOTE",
        "promotion keeps parent rollback target": records[1].rollback_target == robust.parent_id,
        "duplicate pair fails closed": records[2].decision == "REJECT",
        "invalid evidence is explicit": records[2].reasons[0].startswith("invalid evidence:"),
        "records bind evidence versions": all(record.evidence_id.count(":") == 2 for record in records),
    }
    for name, passed in checks.items():
        print(f"{'PASS' if passed else 'FAIL'} | {name}")
    assert all(checks.values())
    print(f"self-check: {sum(checks.values())}/{len(checks)} PASS")


if __name__ == "__main__":
    main()
