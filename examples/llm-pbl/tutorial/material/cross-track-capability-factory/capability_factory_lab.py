#!/usr/bin/env python3
"""CPU-only toy for multi-teacher OPD estimation and governed promotion."""

from __future__ import annotations

import math
import random
import statistics
from dataclasses import dataclass


DOMAINS = ("math", "code", "style")
TEACHERS = {
    "math": (0.92, 0.06, 0.02),
    "code": (0.05, 0.90, 0.05),
    "style": (0.15, 0.15, 0.70),
}
STUDENTS = {
    "math": (0.46, 0.34, 0.20),
    "code": (0.35, 0.45, 0.20),
    "style": (0.34, 0.31, 0.35),
}


def categorical_sample(probs: tuple[float, ...], rng: random.Random) -> int:
    r = rng.random()
    cumulative = 0.0
    for i, prob in enumerate(probs):
        cumulative += prob
        if r < cumulative:
            return i
    return len(probs) - 1


def reverse_kl(q: tuple[float, ...], p: tuple[float, ...]) -> float:
    return sum(qi * (math.log(qi) - math.log(pi)) for qi, pi in zip(q, p))


def exact_logit_gradient(
    q: tuple[float, ...], p: tuple[float, ...]
) -> tuple[float, ...]:
    """Exact gradient of KL(q || p) with respect to q's softmax logits."""
    kl = reverse_kl(q, p)
    return tuple(qi * ((math.log(qi) - math.log(pi)) - kl) for qi, pi in zip(q, p))


def sampled_token_gradient(
    q: tuple[float, ...],
    p: tuple[float, ...],
    samples: int,
    rng: random.Random,
) -> tuple[float, ...]:
    """Unbiased score-function estimate using only p(y) for y sampled from q."""
    baseline = reverse_kl(q, p)
    estimate = [0.0] * len(q)
    for _ in range(samples):
        y = categorical_sample(q, rng)
        centered_cost = math.log(q[y]) - math.log(p[y]) - baseline
        for k, qk in enumerate(q):
            estimate[k] += ((1.0 if y == k else 0.0) - qk) * centered_cost
    return tuple(value / samples for value in estimate)


def estimator_rmse(samples: int, replications: int = 96) -> float:
    squared_errors: list[float] = []
    for domain_i, domain in enumerate(DOMAINS):
        q, p = STUDENTS[domain], TEACHERS[domain]
        exact = exact_logit_gradient(q, p)
        for replication in range(replications):
            seed = 20260813 + 100_003 * samples + 997 * domain_i + replication
            estimate = sampled_token_gradient(q, p, samples, random.Random(seed))
            squared_errors.extend((got - want) ** 2 for got, want in zip(estimate, exact))
    return math.sqrt(sum(squared_errors) / len(squared_errors))


def routed_objective(route: dict[str, str]) -> float:
    return sum(
        reverse_kl(STUDENTS[domain], TEACHERS[route[domain]]) for domain in DOMAINS
    ) / len(DOMAINS)


def routed_gradient(route: dict[str, str]) -> tuple[float, ...]:
    return tuple(
        value
        for domain in DOMAINS
        for value in exact_logit_gradient(STUDENTS[domain], TEACHERS[route[domain]])
    )


def cosine(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    return dot / (norm_a * norm_b)


@dataclass(frozen=True)
class Snapshot:
    snapshot_id: str
    parent_id: str | None
    teacher_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class Candidate:
    snapshot: Snapshot
    public_paired_deltas: tuple[float, ...]
    hidden_paired_deltas: tuple[float, ...]
    domain_deltas: tuple[tuple[str, float], ...]
    cost_ratio: float


def lower_confidence_bound(values: tuple[float, ...], z: float = 1.645) -> float:
    """Toy one-sided normal LCB; a real gate should choose a justified estimator."""
    mean = statistics.fmean(values)
    if len(values) < 2:
        return float("-inf")
    return mean - z * statistics.stdev(values) / math.sqrt(len(values))


def traceable(snapshot: Snapshot, registry: dict[str, Snapshot]) -> bool:
    seen: set[str] = set()
    current: Snapshot | None = snapshot
    while current is not None:
        if current.snapshot_id in seen:
            return False
        seen.add(current.snapshot_id)
        if any(teacher_id not in registry for teacher_id in current.teacher_ids):
            return False
        if current.parent_id is None:
            return True
        current = registry.get(current.parent_id)
        if current is None:
            return False
    return False


def promotion_decision(
    candidate: Candidate, registry: dict[str, Snapshot]
) -> tuple[str, tuple[str, ...]]:
    reasons: list[str] = []
    public_lcb = lower_confidence_bound(candidate.public_paired_deltas)
    hidden_lcb = lower_confidence_bound(candidate.hidden_paired_deltas)
    worst_domain = min(delta for _, delta in candidate.domain_deltas)

    if public_lcb <= 0.0:
        reasons.append(f"public LCB={public_lcb:+.3f} <= 0")
    if hidden_lcb < 0.0:
        reasons.append(f"hidden LCB={hidden_lcb:+.3f} < 0")
    if worst_domain < -0.02:
        reasons.append(f"worst domain regression={worst_domain:+.3f} < -0.020")
    if candidate.cost_ratio > 1.15:
        reasons.append(f"cost ratio={candidate.cost_ratio:.2f} > 1.15")
    if not traceable(candidate.snapshot, registry):
        reasons.append("lineage is incomplete or cyclic")
    return ("PROMOTE" if not reasons else "REJECT", tuple(reasons))


def build_registry_and_candidates() -> tuple[dict[str, Snapshot], tuple[Candidate, ...]]:
    snapshots = (
        Snapshot("base-v0", None),
        Snapshot("expert-math-v1", "base-v0"),
        Snapshot("expert-code-v1", "base-v0"),
        Snapshot("expert-style-v1", "base-v0"),
    )
    registry = {snapshot.snapshot_id: snapshot for snapshot in snapshots}
    teachers = ("expert-math-v1", "expert-code-v1", "expert-style-v1")
    candidates = (
        Candidate(
            Snapshot("mixed-rl-public-peak", "base-v0", teachers),
            (0.052, 0.048, 0.050, 0.047, 0.055, 0.049, 0.051, 0.050),
            (-0.031, -0.027, -0.035, -0.030, -0.028, -0.033, -0.029, -0.032),
            (("math", 0.061), ("code", -0.041), ("style", 0.032)),
            1.04,
        ),
        Candidate(
            Snapshot("opd-integrated-v1", "base-v0", teachers),
            (0.031, 0.029, 0.034, 0.032, 0.027, 0.030, 0.033, 0.031),
            (0.012, 0.009, 0.013, 0.011, 0.008, 0.010, 0.012, 0.009),
            (("math", 0.042), ("code", 0.031), ("style", 0.015)),
            1.08,
        ),
        Candidate(
            Snapshot("opd-router-bug", "base-v0", teachers),
            (0.028, 0.031, 0.027, 0.029, 0.030, 0.028, 0.032, 0.029),
            (0.006, 0.004, 0.007, 0.005, 0.003, 0.006, 0.004, 0.005),
            (("math", 0.039), ("code", -0.071), ("style", 0.026)),
            1.06,
        ),
    )
    registry.update({candidate.snapshot.snapshot_id: candidate.snapshot for candidate in candidates})
    return registry, candidates


def main() -> None:
    print("=" * 76)
    print("Capability Factory L0 — multi-teacher signal + governed promotion")
    print("=" * 76)
    print("口径: 纯算术/治理 toy；不训练 LLM，不把 toy 分数冒充模型 benchmark。")

    sample_sizes = (16, 256, 4096)
    rmses = tuple(estimator_rmse(samples) for samples in sample_sizes)
    print("\n[1] 同一个 reverse-KL：full-vocabulary 精确梯度 vs sampled-token 估计")
    for samples, rmse in zip(sample_sizes, rmses):
        print(f"    sampled tokens={samples:4d} | gradient RMSE={rmse:.6f}")
    print("    => sampled-token 省教师全词表传输，但用采样方差换带宽/显存。")

    correct_route = {domain: domain for domain in DOMAINS}
    wrong_route = {"math": "code", "code": "style", "style": "math"}
    correct_objective = routed_objective(correct_route)
    wrong_objective = routed_objective(wrong_route)
    gradient_cosine = cosine(routed_gradient(correct_route), routed_gradient(wrong_route))
    print("\n[2] 多教师不是把 checkpoint 列出来：routing 本身定义了优化目标")
    print(f"    correct-route KL={correct_objective:.4f}")
    print(f"    wrong-route   KL={wrong_objective:.4f}")
    print(f"    gradient cosine(correct, wrong)={gradient_cosine:+.4f}")
    print("    => 教师选错时，训练可以很稳定地优化错误目标。")

    registry, candidates = build_registry_and_candidates()
    decisions: dict[str, str] = {}
    print("\n[3] candidate-parent gate：总分上涨不是 promotion 的充分条件")
    for candidate in candidates:
        decision, reasons = promotion_decision(candidate, registry)
        decisions[candidate.snapshot.snapshot_id] = decision
        public_lcb = lower_confidence_bound(candidate.public_paired_deltas)
        hidden_lcb = lower_confidence_bound(candidate.hidden_paired_deltas)
        print(
            f"    {candidate.snapshot.snapshot_id:22s} {decision:7s} | "
            f"public_LCB={public_lcb:+.3f} hidden_LCB={hidden_lcb:+.3f}"
        )
        for reason in reasons:
            print(f"      - {reason}")

    promoted = [model_id for model_id, decision in decisions.items() if decision == "PROMOTE"]
    rollback_target = registry[promoted[0]].parent_id
    print("\n[4] append-only lineage / rollback")
    print(f"    promoted={promoted}")
    print(f"    rollback_target={rollback_target} (parent snapshot remains immutable)")

    checks = (
        (rmses[0] > rmses[1] > rmses[2], "sampled-token RMSE decreases with samples"),
        (
            all(abs(sum(exact_logit_gradient(STUDENTS[d], TEACHERS[d]))) < 1e-12 for d in DOMAINS),
            "exact softmax-logit gradients sum to zero",
        ),
        (wrong_objective > correct_objective, "wrong routing increases this toy objective"),
        (gradient_cosine < 0.5, "wrong routing materially changes update direction"),
        (decisions["mixed-rl-public-peak"] == "REJECT", "hidden regression is rejected"),
        (decisions["opd-integrated-v1"] == "PROMOTE", "bounded integrated gain is promoted"),
        (decisions["opd-router-bug"] == "REJECT", "single-domain regression is rejected"),
        (all(traceable(candidate.snapshot, registry) for candidate in candidates), "lineage is traceable"),
        (rollback_target == "base-v0", "promoted candidate has an immutable rollback target"),
    )
    failed = [name for ok, name in checks if not ok]
    print("\n[5] self-check")
    for ok, name in checks:
        print(f"    {'PASS' if ok else 'FAIL'} | {name}")
    if failed:
        raise AssertionError(f"self-check failed: {failed}")
    print(f"\nSELF-CHECK: {len(checks)}/{len(checks)} PASS")
    print("takeaway: 专家可以并行生产；集成仍需估计器选择、正确 routing 与独立 promotion gate。")


if __name__ == "__main__":
    main()
