#!/usr/bin/env python3
"""Pure-Python pretraining lifecycle: data order -> CE -> AdamW -> exact resume."""
from __future__ import annotations
import json
import math
import random
TOKENS = ("<bos>", "a", "b", "c", "<eos>")
V = len(TOKENS)
TRAIN_DOCS = (
    ("general", (0, 1, 2, 4)),
    ("general", (0, 2, 1, 4)),
    ("domain", (0, 3, 3, 4)),
)
VALID_DOCS = ((0, 1, 2, 4), (0, 3, 3, 4))
def zeros() -> list[list[float]]:
    return [[0.0 for _ in range(V)] for _ in range(V)]
def init_state(seed: int = 17) -> dict:
    rng = random.Random(seed)
    return {
        "model": [[rng.uniform(-0.04, 0.04) for _ in range(V)] for _ in range(V)],
        "adam_m": zeros(),
        "adam_v": zeros(),
        "optimizer_step": 0,
        "sampler_epoch": 0,
        "sampler_cursor": 0,
        "sampler_seed": 20260813,
        "mixture": {"general": 1, "domain": 2},
    }
def epoch_examples(state: dict) -> list[tuple[int, int, str]]:
    weighted_docs = [
        (domain, tokens)
        for domain, tokens in TRAIN_DOCS
        for _ in range(state["mixture"][domain])
    ]
    random.Random(state["sampler_seed"] + state["sampler_epoch"]).shuffle(weighted_docs)
    return [
        (tokens[i], tokens[i + 1], domain)
        for domain, tokens in weighted_docs
        for i in range(len(tokens) - 1)
    ]
def next_examples(state: dict, count: int) -> list[tuple[int, int, str]]:
    result: list[tuple[int, int, str]] = []
    while len(result) < count:
        examples = epoch_examples(state)
        remaining = len(examples) - state["sampler_cursor"]
        take = min(count - len(result), remaining)
        start = state["sampler_cursor"]
        result.extend(examples[start : start + take])
        state["sampler_cursor"] += take
        if state["sampler_cursor"] == len(examples):
            state["sampler_epoch"] += 1
            state["sampler_cursor"] = 0
    return result
def softmax(row: list[float]) -> list[float]:
    peak = max(row)
    exp = [math.exp(value - peak) for value in row]
    total = sum(exp)
    return [value / total for value in exp]
def loss_and_grad(model: list[list[float]], batch: list[tuple[int, int, str]]) -> tuple[float, list[list[float]]]:
    grad = zeros()
    loss = 0.0
    for source, target, _ in batch:
        probs = softmax(model[source])
        loss -= math.log(probs[target])
        for token in range(V):
            grad[source][token] += probs[token] - (1.0 if token == target else 0.0)
    scale = 1.0 / len(batch)
    return loss * scale, [[value * scale for value in row] for row in grad]
def add_grad(total: list[list[float]], grad: list[list[float]], scale: float) -> None:
    for i in range(V):
        for j in range(V):
            total[i][j] += scale * grad[i][j]
def learning_rate(step: int, total_steps: int = 20, warmup: int = 3, peak: float = 0.12) -> float:
    if step < warmup:
        return peak * (step + 1) / warmup
    return peak * max(0.0, (total_steps - step) / (total_steps - warmup))
def adamw_step(state: dict, grad: list[list[float]], total_steps: int) -> float:
    step = state["optimizer_step"]
    lr = learning_rate(step, total_steps)
    beta1, beta2, eps, weight_decay = 0.9, 0.99, 1e-8, 0.01
    for i in range(V):
        for j in range(V):
            g = grad[i][j]
            state["adam_m"][i][j] = beta1 * state["adam_m"][i][j] + (1 - beta1) * g
            state["adam_v"][i][j] = beta2 * state["adam_v"][i][j] + (1 - beta2) * g * g
            m_hat = state["adam_m"][i][j] / (1 - beta1 ** (step + 1))
            v_hat = state["adam_v"][i][j] / (1 - beta2 ** (step + 1))
            state["model"][i][j] *= 1 - lr * weight_decay
            state["model"][i][j] -= lr * m_hat / (math.sqrt(v_hat) + eps)
    state["optimizer_step"] += 1
    return lr
def validation_loss(model: list[list[float]]) -> float:
    batch = [(doc[i], doc[i + 1], "valid") for doc in VALID_DOCS for i in range(len(doc) - 1)]
    return loss_and_grad(model, batch)[0]
def train(state: dict, until_step: int, total_steps: int = 20) -> list[tuple[int, float, float, float]]:
    history: list[tuple[int, float, float, float]] = []
    while state["optimizer_step"] < until_step:
        accumulated = zeros()
        losses = []
        for _ in range(2):  # two micro-batches -> one global optimizer step
            batch = next_examples(state, 4)
            loss, grad = loss_and_grad(state["model"], batch)
            losses.append(loss)
            add_grad(accumulated, grad, 0.5)
        lr = adamw_step(state, accumulated, total_steps)
        history.append((state["optimizer_step"], sum(losses) / 2, lr, validation_loss(state["model"])))
    return history
def checkpoint_roundtrip(state: dict) -> dict:
    return json.loads(json.dumps(state, sort_keys=True, separators=(",", ":")))
def max_model_diff(a: dict, b: dict) -> float:
    return max(
        abs(a["model"][i][j] - b["model"][i][j])
        for i in range(V)
        for j in range(V)
    )
def cross_document_pairs() -> tuple[int, int]:
    docs = [tokens for _, tokens in TRAIN_DOCS]
    safe = sum(len(doc) - 1 for doc in docs)
    naive = len([token for doc in docs for token in doc]) - 1
    return safe, naive - safe
def main() -> None:
    print("=" * 78)
    print("Pretraining lifecycle L0 — data cursor, causal loss, AdamW and exact resume")
    print("=" * 78)
    safe_pairs, leaked_pairs = cross_document_pairs()
    print("\n[1] Document boundary + causal shift")
    print(f"    within-document (x_t -> x_t+1) pairs={safe_pairs}")
    print(f"    naive concatenation adds cross-document pairs={leaked_pairs}")

    full = init_state()
    initial_val = validation_loss(full["model"])
    history = train(full, 20)
    final_val = validation_loss(full["model"])
    selected_checkpoint = min(history, key=lambda item: item[3])
    print("\n[2] One complete training run")
    print(f"    validation loss: {initial_val:.4f} -> {final_val:.4f}")
    print(f"    lr: step1={history[0][2]:.4f} peak={max(item[2] for item in history):.4f} step20={history[-1][2]:.4f}")
    print(f"    selected checkpoint: step={selected_checkpoint[0]} validation={selected_checkpoint[3]:.4f}")
    print(f"    final sampler=(epoch={full['sampler_epoch']}, cursor={full['sampler_cursor']})")

    split = init_state()
    train(split, 8)
    serialized = checkpoint_roundtrip(split)
    checkpoint_fields = tuple(sorted(serialized))
    train(serialized, 20)
    exact_diff = max_model_diff(full, serialized)
    print("\n[3] Full-state checkpoint: uninterrupted == resume")
    print("    fields=" + ",".join(checkpoint_fields))
    print(f"    max parameter diff={exact_diff:.3e}")

    no_optimizer = checkpoint_roundtrip(split)
    no_optimizer["adam_m"], no_optimizer["adam_v"] = zeros(), zeros()
    train(no_optimizer, 20)
    optimizer_diff = max_model_diff(full, no_optimizer)
    wrong_cursor = checkpoint_roundtrip(split)
    wrong_cursor["sampler_epoch"], wrong_cursor["sampler_cursor"] = 0, 0
    train(wrong_cursor, 20)
    cursor_diff = max_model_diff(full, wrong_cursor)
    print("\n[4] Failure injection: weights-only is not exact resume")
    print(f"    reset Adam moments -> max parameter diff={optimizer_diff:.3e}")
    print(f"    reset data cursor  -> max parameter diff={cursor_diff:.3e}")

    checks = (
        (safe_pairs == 9 and leaked_pairs == 2, "document boundaries remove two cross-document targets"),
        (final_val < initial_val, "validation loss improves on this constructed task"),
        (history[0][2] < max(item[2] for item in history), "warmup raises learning rate"),
        (history[-1][2] < max(item[2] for item in history), "decay lowers learning rate"),
        (exact_diff == 0.0, "serialized full-state resume is exactly equal"),
        (full["sampler_epoch"] == serialized["sampler_epoch"], "sampler epoch resumes exactly"),
        (full["sampler_cursor"] == serialized["sampler_cursor"], "sampler cursor resumes exactly"),
        (optimizer_diff > 1e-4, "dropping Adam moments changes the result"),
        (cursor_diff > 1e-4, "dropping data cursor changes the result"),
        (selected_checkpoint[0] == 20, "best checkpoint is selected by its versioned validation artifact"),
    )
    print("\n[5] self-check")
    for ok, name in checks:
        print(f"    {'PASS' if ok else 'FAIL'} | {name}")
    failed = [name for ok, name in checks if not ok]
    if failed:
        raise AssertionError(f"self-check failed: {failed}")
    print(f"\nSELF-CHECK: {len(checks)}/{len(checks)} PASS")
    print("takeaway: sharding changes placement; a pretraining lifecycle must also preserve data order and optimizer state.")
if __name__ == "__main__":
    main()
