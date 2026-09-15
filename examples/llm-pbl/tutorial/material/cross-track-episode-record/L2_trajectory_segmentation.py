#!/usr/bin/env python3
"""EpisodeRecord L2: real multi-turn provenance -> flat/split training samples.

Pure-standard-library teaching lab.  It demonstrates four contracts:
1. environment observations are context, never model targets;
2. an episode can become one flat sample or several physical segments;
3. flat and split SFT agree only with token-count weighting and exact prefixes;
4. terminal reward is reduced once per rollout, not once per segment.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
from typing import Optional


@dataclass(frozen=True)
class Span:
    kind: str
    turn_id: int
    token_ids: tuple[int, ...]
    token_nll: tuple[float, ...]
    source: str
    action_id: Optional[str] = None
    caused_by_action: Optional[str] = None
    sft_train: bool = False
    ppo_train: bool = False
    behavior_logprobs: tuple[Optional[float], ...] = ()


@dataclass(frozen=True)
class Trajectory:
    episode_id: str
    rollout_id: str
    policy_version: str
    spans: tuple[Span, ...]
    terminal_reward: float


@dataclass(frozen=True)
class TrainingSegment:
    rollout_id: str
    segment_id: str
    token_ids: tuple[int, ...]
    sft_mask: tuple[int, ...]
    token_nll: tuple[float, ...]


def demo_trajectory() -> Trajectory:
    """One real environment trajectory with an old-policy error and repair."""
    return Trajectory(
        episode_id="episode-medical-agent-007",
        rollout_id="rollout-policy-v12-007",
        policy_version="policy-v12",
        terminal_reward=1.0,
        spans=(
            Span("system", 0, (1, 2), (0.0, 0.0), "system"),
            Span("user", 0, (3, 4, 5), (0.0, 0.0, 0.0), "user"),
            Span(
                "thinking+tool_call", 1, (10, 11, 12), (0.30, 0.40, 0.50),
                "current_policy", action_id="call-1", sft_train=True,
                ppo_train=True, behavior_logprobs=(-0.10, -0.20, -0.30),
            ),
            Span(
                "observation", 1, (20, 21), (0.0, 0.0), "environment",
                caused_by_action="call-1",
            ),
            Span(
                "bad_old_action", 2, (30, 31), (0.0, 0.0), "old_policy",
                action_id="call-2", behavior_logprobs=(-0.60, -0.70),
            ),
            Span(
                "observation", 2, (40,), (0.0,), "environment",
                caused_by_action="call-2",
            ),
            Span(
                "teacher_repair+final", 3, (50, 51, 52, 53),
                (0.90, 1.00, 1.10, 1.20), "teacher", sft_train=True,
            ),
        ),
    )


def validate_trajectory(trajectory: Trajectory) -> None:
    previous_span_action: Optional[str] = None
    for span in trajectory.spans:
        width = len(span.token_ids)
        if len(span.token_nll) != width:
            raise ValueError(f"{span.kind}: token_nll length mismatch")
        if span.behavior_logprobs and len(span.behavior_logprobs) != width:
            raise ValueError(f"{span.kind}: behavior_logprobs length mismatch")
        if span.kind == "observation":
            if span.source != "environment":
                raise ValueError("observation must come from the environment, not the model")
            if span.caused_by_action != previous_span_action:
                raise ValueError("observation must reference the immediately preceding action")
            if span.sft_train or span.ppo_train:
                raise ValueError("environment observation must be context-only")
        if span.ppo_train:
            if span.source != "current_policy":
                raise ValueError("PPO target must come from the admitted behavior policy")
            if not span.behavior_logprobs or any(value is None for value in span.behavior_logprobs):
                raise ValueError("PPO target requires exact behavior logprobs")
        previous_span_action = span.action_id


def flatten(trajectory: Trajectory) -> TrainingSegment:
    return TrainingSegment(
        rollout_id=trajectory.rollout_id,
        segment_id="flat",
        token_ids=tuple(token for span in trajectory.spans for token in span.token_ids),
        sft_mask=tuple(bit for span in trajectory.spans for bit in (int(span.sft_train),) * len(span.token_ids)),
        token_nll=tuple(value for span in trajectory.spans for value in span.token_nll),
    )


def split_by_supervised_turn(trajectory: Trajectory) -> tuple[TrainingSegment, ...]:
    """Turn-expanded SFT: each target keeps the exact accumulated prefix."""
    history_ids: tuple[int, ...] = ()
    history_nll: tuple[float, ...] = ()
    segments: list[TrainingSegment] = []
    for span in trajectory.spans:
        if span.sft_train:
            segments.append(
                TrainingSegment(
                    rollout_id=trajectory.rollout_id,
                    segment_id=f"turn-{span.turn_id}",
                    token_ids=history_ids + span.token_ids,
                    sft_mask=(0,) * len(history_ids) + (1,) * len(span.token_ids),
                    token_nll=history_nll + span.token_nll,
                )
            )
        history_ids += span.token_ids
        history_nll += span.token_nll
    return tuple(segments)


def masked_loss(segment: TrainingSegment) -> tuple[float, int]:
    selected = [value for value, bit in zip(segment.token_nll, segment.sft_mask) if bit]
    if not selected:
        raise ValueError(f"{segment.segment_id}: no trainable tokens")
    return sum(selected) / len(selected), len(selected)


def validate_append_only_prefix(trajectory: Trajectory, segments: tuple[TrainingSegment, ...]) -> None:
    targets = [span for span in trajectory.spans if span.sft_train]
    for target, segment in zip(targets, segments):
        expected_prefix: tuple[int, ...] = ()
        for span in trajectory.spans:
            if span is target:
                break
            expected_prefix += span.token_ids
        if segment.token_ids[: len(expected_prefix)] != expected_prefix:
            raise ValueError(f"{segment.segment_id}: prefix drift; decoded text was likely re-tokenized")


def main() -> None:
    trajectory = demo_trajectory()
    validate_trajectory(trajectory)
    flat = flatten(trajectory)
    segments = split_by_supervised_turn(trajectory)
    validate_append_only_prefix(trajectory, segments)

    flat_loss, flat_tokens = masked_loss(flat)
    per_segment = [masked_loss(segment) for segment in segments]
    token_weighted = sum(loss * count for loss, count in per_segment) / sum(count for _, count in per_segment)
    simple_mean = sum(loss for loss, _ in per_segment) / len(per_segment)
    expanded_tokens = sum(len(segment.token_ids) for segment in segments)
    ppo_tokens = sum(len(span.token_ids) for span in trajectory.spans if span.ppo_train)

    fake_observation = replace(
        trajectory,
        spans=tuple(
            replace(span, source="model") if span.kind == "observation" and span.turn_id == 1 else span
            for span in trajectory.spans
        ),
    )
    fake_observation_rejected = False
    try:
        validate_trajectory(fake_observation)
    except ValueError:
        fake_observation_rejected = True

    corrupted = list(segments)
    drifted_ids = list(corrupted[-1].token_ids)
    drifted_ids[1] = 999
    corrupted[-1] = replace(corrupted[-1], token_ids=tuple(drifted_ids))
    prefix_drift_rejected = False
    try:
        validate_append_only_prefix(trajectory, tuple(corrupted))
    except ValueError:
        prefix_drift_rejected = True

    naive_reward_sum = sum(trajectory.terminal_reward for _ in segments)
    rollout_reward = {
        segment.rollout_id: trajectory.terminal_reward for segment in segments
    }
    correct_reward_sum = sum(rollout_reward.values())

    print("=" * 78)
    print("EpisodeRecord L2 — real multi-turn -> flat/split training segments")
    print("=" * 78)
    print("\n[1] One semantic trajectory, two training layouts")
    print(f"    episode={trajectory.episode_id} rollout={trajectory.rollout_id}")
    print(f"    flat: tokens={len(flat.token_ids)} sft_targets={flat_tokens} ppo_targets={ppo_tokens}")
    print(f"    split: segments={len(segments)} physical_tokens={expanded_tokens} (prefixes are repeated)")
    print("    observation source=environment and mask=0; old bad action stays in context and mask=0")

    print("\n[2] Loss equivalence requires token-count weighting")
    print(f"    flat token mean       = {flat_loss:.6f}")
    print(f"    split token-weighted  = {token_weighted:.6f}")
    print(f"    split simple turn mean= {simple_mean:.6f} (different objective)")

    print("\n[3] Fail-closed provenance and token continuity")
    print(f"    model-fabricated observation -> {'REJECT' if fake_observation_rejected else 'ADMIT'}")
    print(f"    re-tokenized prefix drift     -> {'REJECT' if prefix_drift_rejected else 'ADMIT'}")

    print("\n[4] Terminal reward belongs to the rollout")
    print(f"    naive per-segment copy={naive_reward_sum:.1f} | rollout reducer={correct_reward_sum:.1f}")

    checks = (
        (len(flat.token_ids) == 17, "flat sample contains the complete trajectory"),
        (flat_tokens == 7, "SFT trains current action plus teacher repair/final"),
        (ppo_tokens == 3, "PPO trains only exact current-policy sampled tokens"),
        (sum(1 for span in trajectory.spans if span.kind == "observation" and span.sft_train) == 0,
         "environment observations are context-only"),
        (all(segment.rollout_id == trajectory.rollout_id for segment in segments),
         "all physical segments retain one rollout identity"),
        (abs(flat_loss - token_weighted) < 1e-12, "flat equals token-weighted split"),
        (abs(flat_loss - simple_mean) > 1e-3, "simple turn mean changes the objective"),
        (expanded_tokens > len(flat.token_ids), "turn expansion recomputes accumulated prefixes"),
        (fake_observation_rejected, "model-fabricated observation is rejected"),
        (prefix_drift_rejected, "re-tokenized prefix drift is rejected"),
        (naive_reward_sum == 2.0 and correct_reward_sum == 1.0,
         "terminal reward is counted once per rollout"),
        (all(
            span.behavior_logprobs and all(value is not None for value in span.behavior_logprobs)
            for span in trajectory.spans if span.ppo_train
        ), "PPO targets retain exact behavior logprobs"),
    )
    print("\n[5] self-check")
    for ok, name in checks:
        print(f"    {'PASS' if ok else 'FAIL'} | {name}")
    failed = [name for ok, name in checks if not ok]
    if failed:
        raise AssertionError(f"self-check failed: {failed}")

    metrics = {
        "checks": len(checks),
        "flat_tokens": len(flat.token_ids),
        "sft_targets": flat_tokens,
        "ppo_targets": ppo_tokens,
        "segments": len(segments),
    }
    digest = hashlib.sha256(json.dumps(metrics, sort_keys=True).encode()).hexdigest()[:16]
    print(f"\nSELF-CHECK: {len(checks)}/{len(checks)} PASS")
    print(f"digest={digest}")
    print("takeaway: multi-turn truth lives in environment provenance; sample count is a trainer layout choice.")


if __name__ == "__main__":
    main()
