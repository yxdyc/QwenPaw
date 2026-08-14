#!/usr/bin/env python3
"""Dependency-free EpisodeRecord contract and train-admission lab."""
from __future__ import annotations
from dataclasses import dataclass, replace
import math

@dataclass(frozen=True)
class EpisodeRecord:
    episode_id: str
    prompt_source: str
    token_ids: tuple[int, ...]
    loss_mask: tuple[int, ...]
    old_logprobs: tuple[float, ...]
    reference_logprobs: tuple[float, ...]
    teacher_logprobs: tuple[float, ...]
    rewards: tuple[float, ...]
    values: tuple[float, ...]
    done: bool
    truncated: bool
    bootstrap_value: float | None
    group_id: str | None
    actions: tuple[str, ...]
    observations: tuple[str, ...]
    policy_version: str
    reward_version: str
    evaluator_version: str
    environment_version: str
    teacher_id: str | None
    router_version: str | None

def base_record() -> EpisodeRecord:
    return EpisodeRecord(
        "ep-0007", "dataset://math/v3#item-42", (101, 17, 18, 19), (0, 1, 1, 1),
        (-0.10, -0.42, -0.31, -0.28), (-0.11, -0.45, -0.33, -0.30),
        (-0.08, -0.24, -0.20, -0.18), (0.0, 0.0, 0.0, 1.0),
        (0.40, 0.45, 0.52, 0.58), False, True, 0.70, "prompt-42",
        ("calculator:6*7",), ("42",), "policy-sha256:aaa", "reward-v4",
        "eval-v9", "sandbox-v5", "teacher-math-v2", "router-v3",
    )

def expected(algorithm: str) -> dict[str, str]:
    return {
        "algorithm": algorithm, "policy_version": "policy-sha256:aaa",
        "reward_version": "reward-v4", "evaluator_version": "eval-v9",
        "environment_version": "sandbox-v5", "teacher_id": "teacher-math-v2",
        "router_version": "router-v3",
    }

def common_errors(record: EpisodeRecord, ctx: dict[str, str]) -> list[str]:
    errors: list[str] = []
    n = len(record.token_ids)
    aligned = (
        "loss_mask", "old_logprobs", "reference_logprobs", "teacher_logprobs",
        "rewards", "values",
    )
    for name in aligned:
        values = getattr(record, name)
        if values and len(values) != n:
            errors.append(f"{name} length={len(values)} != token_ids length={n}")
    if any(mask not in (0, 1) for mask in record.loss_mask):
        errors.append("loss_mask must contain only 0/1")
    if not record.prompt_source:
        errors.append("prompt_source is required for lineage")
    if record.done and record.truncated:
        errors.append("done and truncated are mutually exclusive")
    if record.truncated and record.bootstrap_value is None:
        errors.append("truncated episode requires bootstrap_value")
    if record.done and record.bootstrap_value not in (None, 0.0):
        errors.append("terminal episode must not bootstrap a non-zero value")
    if len(record.actions) != len(record.observations):
        errors.append("actions and observations must form complete tool transitions")
    for name in ("policy_version", "reward_version", "evaluator_version", "environment_version"):
        if getattr(record, name) != ctx[name]:
            errors.append(f"stale {name}: got {getattr(record, name)}, expected {ctx[name]}")
    return errors

def algorithm_errors(record: EpisodeRecord, ctx: dict[str, str]) -> list[str]:
    algorithm, n = ctx["algorithm"], len(record.token_ids)
    errors = [] if sum(record.loss_mask) else ["no trainable assistant tokens"]
    if algorithm == "ppo":
        if len(record.old_logprobs) != n:
            errors.append("PPO requires token-aligned old_logprobs")
        if len(record.values) != n:
            errors.append("PPO requires token-aligned values")
    elif algorithm == "grpo":
        if record.group_id is None:
            errors.append("GRPO requires group_id")
    elif algorithm == "opd":
        if len(record.teacher_logprobs) != n:
            errors.append("sampled-token OPD requires token-aligned teacher_logprobs")
        for name in ("teacher_id", "router_version"):
            if getattr(record, name) != ctx[name]:
                errors.append(f"{name} mismatch: got {getattr(record, name)}, expected {ctx[name]}")
    else:
        errors.append(f"unknown algorithm: {algorithm}")
    return errors

def admit(record: EpisodeRecord, algorithm: str) -> tuple[bool, tuple[str, ...]]:
    ctx = expected(algorithm)
    errors = common_errors(record, ctx) + algorithm_errors(record, ctx)
    return not errors, tuple(errors)

def td_target(reward: float, gamma: float, record: EpisodeRecord) -> float:
    if record.done:
        return reward
    if record.truncated:
        assert record.bootstrap_value is not None
        return reward + gamma * record.bootstrap_value
    raise ValueError("lab records only terminal or truncated boundaries")

def grpo_advantages(records: tuple[EpisodeRecord, ...]) -> tuple[float, ...]:
    groups = {record.group_id for record in records}
    if None in groups or len(groups) != 1:
        raise ValueError("records must share one non-null group_id")
    returns = tuple(sum(record.rewards) for record in records)
    mean = sum(returns) / len(returns)
    variance = sum((value - mean) ** 2 for value in returns) / len(returns)
    if variance < 1e-12:
        raise ValueError("dead group: all returns are identical")
    std = math.sqrt(variance)
    return tuple((value - mean) / std for value in returns)

def main() -> None:
    print("=" * 78)
    print("EpisodeRecord L0 — one trajectory contract, three learning algorithms")
    print("=" * 78)
    record = base_record()
    print(f"episode={record.episode_id} tokens={len(record.token_ids)} trainable={sum(record.loss_mask)}")
    print("\n[1] Algorithm admission on the same record")
    admitted = {}
    for algorithm in ("ppo", "grpo", "opd"):
        ok, errors = admit(record, algorithm)
        admitted[algorithm] = ok
        print(f"    {algorithm:4s} -> {'ADMIT' if ok else 'REJECT'}" + (f" | {errors}" if errors else ""))

    print("\n[2] done != truncated: bootstrap changes the target")
    truncated_target = td_target(1.0, 0.99, record)
    terminal_target = td_target(1.0, 0.99, replace(record, done=True, truncated=False, bootstrap_value=0.0))
    print(f"    truncated target = 1.0 + 0.99*0.70 = {truncated_target:.3f}")
    print(f"    terminal  target = 1.0             = {terminal_target:.3f}")
    print(f"    silent bias if truncation is treated as terminal = {terminal_target-truncated_target:+.3f}")

    print("\n[3] Fail closed before train admission")
    failures = {
        "missing bootstrap": replace(record, bootstrap_value=None),
        "stale policy": replace(record, policy_version="policy-sha256:old"),
        "teacher/router mismatch": replace(record, router_version="router-v2"),
        "broken tool trace": replace(record, observations=()),
        "missing provenance": replace(record, prompt_source=""),
    }
    rejected = {}
    for name, bad_record in failures.items():
        errors = admit(bad_record, "opd")[1]
        rejected[name] = errors
        print(f"    {name:24s} -> REJECT | {errors[0]}")

    print("\n[4] GRPO group semantics")
    group = (
        record, replace(record, episode_id="ep-0008", rewards=(0.0, 0.0, 0.0, 0.0)),
        replace(record, episode_id="ep-0009", rewards=(0.0, 0.0, 0.0, 0.5)),
    )
    advantages = grpo_advantages(group)
    print("    diverse rewards=(1.0, 0.0, 0.5) -> advantages=" + str(tuple(round(x, 3) for x in advantages)))
    dead_group_rejected = False
    try:
        grpo_advantages((record, replace(record, episode_id="ep-0010")))
    except ValueError as exc:
        dead_group_rejected = "dead group" in str(exc)
        print(f"    identical rewards=(1.0, 1.0) -> REJECT | {exc}")

    checks = (
        (all(admitted.values()), "valid record is admitted for PPO/GRPO/sampled-token OPD"),
        (abs(truncated_target - 1.693) < 1e-12, "truncation bootstraps"),
        (abs(terminal_target - 1.0) < 1e-12, "terminal does not bootstrap"),
        (all(rejected.values()), "all injected contract/version failures are rejected"),
        (dead_group_rejected, "GRPO dead group is rejected"),
        (abs(sum(advantages)) < 1e-12, "GRPO standardized advantages are centered"),
        (record.policy_version.startswith("policy-sha256:"), "policy version binds a weight identity"),
        (bool(record.prompt_source), "prompt provenance is present"),
        (len(record.actions) == len(record.observations), "tool transitions are complete"),
    )
    print("\n[5] self-check")
    for ok, name in checks:
        print(f"    {'PASS' if ok else 'FAIL'} | {name}")
    failed = [name for ok, name in checks if not ok]
    if failed:
        raise AssertionError(f"self-check failed: {failed}")
    print(f"\nSELF-CHECK: {len(checks)}/{len(checks)} PASS")
    print("takeaway: algorithms change field consumption; provenance, termination and versions remain invariants.")

if __name__ == "__main__":
    main()
