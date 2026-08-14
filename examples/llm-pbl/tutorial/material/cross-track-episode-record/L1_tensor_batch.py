#!/usr/bin/env python3
"""PyTorch EpisodeRecord collation and PPO/GRPO/OPD adapter lab."""
from __future__ import annotations

from dataclasses import dataclass, replace
import io
from typing import Any

import torch

from episode_record_lab import EpisodeRecord, admit, base_record


@dataclass(frozen=True)
class TensorBatch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    loss_mask: torch.Tensor
    old_logprobs: torch.Tensor
    reference_logprobs: torch.Tensor
    teacher_logprobs: torch.Tensor
    rewards: torch.Tensor
    values: torch.Tensor
    done: torch.Tensor
    truncated: torch.Tensor
    bootstrap_values: torch.Tensor
    metadata: list[dict[str, Any]]


FLOAT_FIELDS = (
    "old_logprobs", "reference_logprobs", "teacher_logprobs", "rewards", "values"
)


def variant(
    episode_id: str,
    group_id: str,
    length: int,
    final_reward: float,
    *,
    truncated: bool = False,
    bootstrap: float = 0.0,
) -> EpisodeRecord:
    base = base_record()
    return replace(
        base,
        episode_id=episode_id,
        token_ids=base.token_ids[:length],
        loss_mask=(0,) + (1,) * (length - 1),
        old_logprobs=base.old_logprobs[:length],
        reference_logprobs=base.reference_logprobs[:length],
        teacher_logprobs=base.teacher_logprobs[:length],
        rewards=(0.0,) * (length - 1) + (final_reward,),
        values=base.values[:length],
        done=not truncated,
        truncated=truncated,
        bootstrap_value=bootstrap,
        group_id=group_id,
        actions=(),
        observations=(),
    )


def records() -> tuple[EpisodeRecord, ...]:
    return (
        variant("ep-A1", "prompt-A", 4, 1.0, truncated=True, bootstrap=0.70),
        variant("ep-A2", "prompt-A", 3, 0.0),
        variant("ep-B1", "prompt-B", 3, 0.5),
        variant("ep-B2", "prompt-B", 2, 0.5),
    )


def collate(items: tuple[EpisodeRecord, ...], pad_id: int = 0) -> TensorBatch:
    for record in items:
        errors = []
        for algorithm in ("ppo", "grpo", "opd"):
            errors.extend(admit(record, algorithm)[1])
        if errors:
            raise ValueError(f"{record.episode_id} failed L0 admission: {sorted(set(errors))}")

    batch_size, width = len(items), max(len(record.token_ids) for record in items)
    input_ids = torch.full((batch_size, width), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((batch_size, width), dtype=torch.bool)
    loss_mask = torch.zeros((batch_size, width), dtype=torch.bool)
    floats = {name: torch.zeros((batch_size, width), dtype=torch.float32) for name in FLOAT_FIELDS}
    done = torch.tensor([record.done for record in items], dtype=torch.bool)
    truncated = torch.tensor([record.truncated for record in items], dtype=torch.bool)
    bootstrap = torch.tensor(
        [record.bootstrap_value or 0.0 for record in items], dtype=torch.float32
    )
    metadata: list[dict[str, Any]] = []

    for row, record in enumerate(items):
        length = len(record.token_ids)
        input_ids[row, :length] = torch.tensor(record.token_ids)
        attention_mask[row, :length] = True
        loss_mask[row, :length] = torch.tensor(record.loss_mask, dtype=torch.bool)
        for name in FLOAT_FIELDS:
            floats[name][row, :length] = torch.tensor(getattr(record, name), dtype=torch.float32)
        metadata.append(
            {
                "episode_id": record.episode_id,
                "group_id": record.group_id,
                "length": length,
                "prompt_source": record.prompt_source,
                "policy_version": record.policy_version,
                "reward_version": record.reward_version,
                "evaluator_version": record.evaluator_version,
                "environment_version": record.environment_version,
                "teacher_id": record.teacher_id,
                "router_version": record.router_version,
            }
        )
    return TensorBatch(
        input_ids, attention_mask, loss_mask, *(floats[name] for name in FLOAT_FIELDS),
        done, truncated, bootstrap, metadata
    )


def validate_batch(batch: TensorBatch) -> None:
    shape = batch.input_ids.shape
    tensors = (
        batch.attention_mask, batch.loss_mask, batch.old_logprobs,
        batch.reference_logprobs, batch.teacher_logprobs, batch.rewards, batch.values,
    )
    if any(tensor.shape != shape for tensor in tensors):
        raise ValueError("all token fields must share [batch, time] shape")
    if torch.any(batch.loss_mask & ~batch.attention_mask):
        raise ValueError("loss_mask cannot select padding")
    if torch.any(batch.done & batch.truncated) or torch.any(~(batch.done | batch.truncated)):
        raise ValueError("this boundary batch requires exactly one of done/truncated")
    for row, meta in enumerate(batch.metadata):
        if int(batch.attention_mask[row].sum()) != meta["length"]:
            raise ValueError(f"{meta['episode_id']} length metadata disagrees with attention_mask")
        if batch.truncated[row] and not torch.isfinite(batch.bootstrap_values[row]):
            raise ValueError(f"{meta['episode_id']} truncation requires finite bootstrap")


def ppo_gae(batch: TensorBatch, gamma: float = 0.99, lam: float = 0.95) -> dict[str, torch.Tensor]:
    advantages = torch.zeros_like(batch.rewards)
    value_targets = torch.zeros_like(batch.rewards)
    for row in range(batch.input_ids.shape[0]):
        positions = torch.where(batch.loss_mask[row])[0].tolist()
        gae = torch.tensor(0.0)
        for offset in range(len(positions) - 1, -1, -1):
            pos = positions[offset]
            if offset == len(positions) - 1:
                next_value = batch.bootstrap_values[row] if batch.truncated[row] else torch.tensor(0.0)
            else:
                next_value = batch.values[row, positions[offset + 1]]
            delta = batch.rewards[row, pos] + gamma * next_value - batch.values[row, pos]
            gae = delta + gamma * lam * gae
            advantages[row, pos] = gae
            value_targets[row, pos] = gae + batch.values[row, pos]
    return {
        "old_logprobs": batch.old_logprobs,
        "advantages": advantages,
        "value_targets": value_targets,
        "mask": batch.loss_mask,
    }


def grpo_view(batch: TensorBatch) -> dict[str, Any]:
    episode_returns = (batch.rewards * batch.loss_mask).sum(dim=1)
    episode_advantages = torch.zeros_like(episode_returns)
    token_advantages = torch.zeros_like(batch.rewards)
    admitted = torch.zeros(batch.input_ids.shape[0], dtype=torch.bool)
    quarantined: list[str] = []
    groups: dict[str, list[int]] = {}
    for row, meta in enumerate(batch.metadata):
        groups.setdefault(meta["group_id"], []).append(row)
    for group_id, rows in groups.items():
        values = episode_returns[rows]
        std = values.std(unbiased=False)
        if std < 1e-12:
            quarantined.append(group_id)
            continue
        normalized = (values - values.mean()) / std
        for row, advantage in zip(rows, normalized):
            admitted[row] = True
            episode_advantages[row] = advantage
            token_advantages[row] = advantage * batch.loss_mask[row]
    return {
        "episode_returns": episode_returns,
        "episode_advantages": episode_advantages,
        "token_advantages": token_advantages,
        "admitted": admitted,
        "quarantined_groups": tuple(quarantined),
    }


def opd_view(batch: TensorBatch) -> dict[str, Any]:
    return {
        "student_sample_logprobs": batch.old_logprobs,
        "teacher_sample_logprobs": batch.teacher_logprobs,
        "mask": batch.loss_mask,
        "teacher_versions": tuple(
            (meta["teacher_id"], meta["router_version"]) for meta in batch.metadata
        ),
    }


def round_trip(batch: TensorBatch) -> TensorBatch:
    payload = {"schema_version": 1, "metadata": batch.metadata}
    payload.update({name: getattr(batch, name) for name in batch.__dataclass_fields__ if name != "metadata"})
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    loaded = torch.load(io.BytesIO(buffer.getvalue()), weights_only=True)
    if loaded.pop("schema_version") != 1:
        raise ValueError("unsupported tensor-batch schema")
    return TensorBatch(**loaded)


def main() -> None:
    torch.set_num_threads(1)
    batch = collate(records())
    validate_batch(batch)
    ppo, grpo, opd = ppo_gae(batch), grpo_view(batch), opd_view(batch)
    restored = round_trip(batch)
    restored_ppo, restored_grpo, restored_opd = (
        ppo_gae(restored), grpo_view(restored), opd_view(restored)
    )
    tensor_fields = [name for name in batch.__dataclass_fields__ if name != "metadata"]
    tensors_equal = all(torch.equal(getattr(batch, name), getattr(restored, name)) for name in tensor_fields)
    views_equal = (
        all(torch.equal(ppo[key], restored_ppo[key]) for key in ppo)
        and all(
            torch.equal(grpo[key], restored_grpo[key])
            for key in ("episode_returns", "episode_advantages", "token_advantages", "admitted")
        )
        and torch.equal(opd["mask"], restored_opd["mask"])
    )

    print("=" * 78)
    print("EpisodeRecord L1 — immutable records -> tensor batch -> algorithm views")
    print("=" * 78)
    print("\n[1] Right-padded tensor batch")
    print(f"    shape={tuple(batch.input_ids.shape)} lengths={[m['length'] for m in batch.metadata]}")
    print(f"    attention tokens={int(batch.attention_mask.sum())} trainable tokens={int(batch.loss_mask.sum())}")
    print(f"    padding selected by loss={int((batch.loss_mask & ~batch.attention_mask).sum())}")

    last_a1, last_a2 = 3, 2
    print("\n[2] PPO/GAE boundary semantics")
    print(f"    truncated ep-A1 last target={ppo['value_targets'][0, last_a1]:.3f} (1 + .99*.70)")
    print(f"    terminal  ep-A2 last target={ppo['value_targets'][1, last_a2]:.3f} (0, no bootstrap)")

    print("\n[3] GRPO group gate")
    print(f"    returns={grpo['episode_returns'].tolist()}")
    print(f"    admitted rows={grpo['admitted'].tolist()} quarantined={grpo['quarantined_groups']}")
    print(f"    prompt-A episode advantages={grpo['episode_advantages'][:2].tolist()} (centered)")
    print(f"    after token broadcast, row sums={grpo['token_advantages'][:2].sum(dim=1).tolist()} (length weighting)")

    valid_gap = (opd["student_sample_logprobs"] - opd["teacher_sample_logprobs"])[opd["mask"]]
    print("\n[4] Sampled-token OPD view")
    print(f"    aligned teacher signals={valid_gap.numel()} padding signals=0")
    print(f"    teacher/router identities={sorted(set(opd['teacher_versions']))}")

    corrupted = replace(batch, loss_mask=batch.loss_mask.clone())
    corrupted.loss_mask[3, 3] = True
    rejected_padding = False
    try:
        validate_batch(corrupted)
    except ValueError as exc:
        rejected_padding = "padding" in str(exc)
        print(f"\n[5] Fail closed: mask-on-padding -> REJECT | {exc}")

    checks = (
        (batch.input_ids.shape == (4, 4), "variable-length records collate to [4,4]"),
        (int(batch.attention_mask.sum()) == 12, "attention mask counts real tokens"),
        (int((batch.loss_mask & ~batch.attention_mask).sum()) == 0, "padding never enters loss"),
        (abs(float(ppo["value_targets"][0, 3]) - 1.693) < 1e-6, "truncation bootstraps"),
        (abs(float(ppo["value_targets"][1, 2])) < 1e-7, "terminal boundary does not bootstrap"),
        (grpo["admitted"].tolist() == [True, True, False, False], "dead GRPO group is quarantined"),
        (abs(float(grpo["episode_advantages"][:2].sum())) < 1e-7, "valid GRPO episode advantages are centered"),
        (grpo["token_advantages"][:2].sum(dim=1).tolist() == [3.0, -2.0], "token broadcast exposes length weighting"),
        (valid_gap.numel() == int(batch.loss_mask.sum()), "OPD signals align only to trainable tokens"),
        (rejected_padding, "batch gate rejects mask-on-padding"),
        (tensors_equal, "round-trip preserves every tensor bitwise"),
        (batch.metadata == restored.metadata, "round-trip preserves metadata and versions"),
        (views_equal and opd["teacher_versions"] == restored_opd["teacher_versions"], "views are resume-stable"),
    )
    print("\n[6] round-trip + self-check")
    print(f"    tensors bitwise equal={tensors_equal} metadata equal={batch.metadata == restored.metadata}")
    for ok, name in checks:
        print(f"    {'PASS' if ok else 'FAIL'} | {name}")
    failed = [name for ok, name in checks if not ok]
    if failed:
        raise AssertionError(f"self-check failed: {failed}")
    print(f"\nSELF-CHECK: {len(checks)}/{len(checks)} PASS")
    print("takeaway: records are facts; padded batches and algorithm views are reproducible derivatives.")


if __name__ == "__main__":
    main()
