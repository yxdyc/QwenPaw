"""L0: deterministic mechanism model for visual tokens -> language readout."""

from __future__ import annotations

import hashlib
import json
import math
import random


def patchify(image: list[list[int]]) -> list[dict]:
    """Turn a 3x3 pixel grid into 1x1 patches with explicit 2D coordinates."""
    return [
        {"value": float(value), "row": row, "col": col}
        for row, line in enumerate(image)
        for col, value in enumerate(line)
    ]


def project(patch: dict, use_2d_position: bool = True) -> dict:
    """A fixed visual projector: content scalar + row/column one-hot keys."""
    row_key = [0.0] * 3
    col_key = [0.0] * 3
    if use_2d_position:
        row_key[patch["row"]] = 1.0
        col_key[patch["col"]] = 1.0
    return {"content": patch["value"], "key": row_key + col_key}


def pack(image_tokens: list[dict], question: dict) -> list[dict]:
    """Place all image tokens before one question token, as in a causal stream."""
    query = [0.0] * 6
    query[question["row"]] = 1.0
    query[3 + question["col"]] = 1.0
    return image_tokens + [{"role": "question", "query": query}]


def causal_attention_readout(sequence: list[dict]) -> int:
    """The final token may attend only to preceding visual tokens."""
    query = sequence[-1]["query"]
    visual = sequence[:-1]
    scores = [12.0 * sum(a * b for a, b in zip(query, token["key"])) for token in visual]
    peak = max(scores)
    weights = [math.exp(score - peak) for score in scores]
    value = sum(w * token["content"] for w, token in zip(weights, visual)) / sum(weights)
    return int(round(value))


def predict(image: list[list[int]], question: dict, mode: str) -> int:
    patches = patchify(image)
    if mode == "patch_shuffle":
        values = [patch["value"] for patch in patches]
        random.Random(7).shuffle(values)
        for patch, value in zip(patches, values):
            patch["value"] = value
    if mode == "image_drop":
        for patch in patches:
            patch["value"] = 0.0
    tokens = [project(patch, mode != "no_2d_position") for patch in patches]
    return causal_attention_readout(pack(tokens, question))


IMAGES = [
    [[0, 1, 2], [2, 0, 1], [1, 2, 0]],
    [[2, 0, 1], [1, 2, 0], [0, 1, 2]],
    [[1, 2, 0], [0, 1, 2], [2, 0, 1]],
]
QUESTIONS = [
    {"skill": "top_left", "row": 0, "col": 0},
    {"skill": "center", "row": 1, "col": 1},
    {"skill": "bottom_right", "row": 2, "col": 2},
]


def evaluate(mode: str) -> tuple[dict[str, float], list[int]]:
    hits = {question["skill"]: [] for question in QUESTIONS}
    predictions = []
    for image in IMAGES:
        for question in QUESTIONS:
            prediction = predict(image, question, mode)
            target = image[question["row"]][question["col"]]
            predictions.append(prediction)
            hits[question["skill"]].append(prediction == target)
    return ({skill: sum(values) / len(values) for skill, values in hits.items()}, predictions)


def mean(values: dict[str, float]) -> float:
    return sum(values.values()) / len(values)


def main() -> None:
    baseline, predictions = evaluate("baseline")
    drop, _ = evaluate("image_drop")
    shuffle, _ = evaluate("patch_shuffle")
    no_position, _ = evaluate("no_2d_position")
    swapped_predictions = []
    for index, _image in enumerate(IMAGES):
        swapped = IMAGES[(index + 1) % len(IMAGES)]
        for question in QUESTIONS:
            swapped_predictions.append(predict(swapped, question, "baseline"))
    sensitivity = sum(a != b for a, b in zip(predictions, swapped_predictions)) / len(predictions)
    metrics = {
        "counterfactual_sensitivity": round(sensitivity, 3),
        "image_dependence_gain": round(mean(baseline) - mean(drop), 3),
        "skill_exact_match": baseline,
    }
    checks = {
        "baseline_all_correct": mean(baseline) == 1.0,
        "drop_hurts": mean(drop) < mean(baseline),
        "no_2d_position_hurts": mean(no_position) < mean(baseline),
        "patch_shuffle_hurts": mean(shuffle) < mean(baseline),
        "swap_changes_answer": sensitivity > 0.0,
    }
    print("L0 visual tokens -> language")
    print(f"patches/image=9 packed_tokens/example=10")
    print(f"baseline skill EM={baseline}")
    print(f"image-drop mean EM={mean(drop):.3f}")
    print(f"image-swap counterfactual sensitivity={sensitivity:.3f}")
    print(f"patch-shuffle mean EM={mean(shuffle):.3f}")
    print(f"remove-2D-position mean EM={mean(no_position):.3f}")
    print(f"checks={sum(checks.values())}/{len(checks)}")
    boundary = "fixed projector/readout mechanism simulation; not a trained VLM"
    payload = {"metrics": metrics, "checks": checks, "evidence_boundary": boundary}
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]
    result = {
        "schema_version": "1.0",
        "module": "nano-vlm-understanding/L0",
        **payload,
        "digest": digest,
    }
    print("RESULT_JSON=" + json.dumps(result, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
