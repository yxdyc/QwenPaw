"""L0: extend 2D latent tokens to joint (time, height, width) video tokens."""

from __future__ import annotations

import hashlib
import json


def patchify_3d(video: list[list[list[float]]]) -> list[dict]:
    """Use 1x1x1 patches so every latent token carries an explicit (t,h,w) position."""
    return [
        {"value": value, "position": (time, row, col)}
        for time, frame in enumerate(video)
        for row, line in enumerate(frame)
        for col, value in enumerate(line)
    ]


def frame(value: float) -> list[list[float]]:
    return [[value, value + 0.05], [value + 0.10, value + 0.15]]


def joint_prediction(first: float, last: float, frames: int) -> list[float]:
    """One temporal predictor couples all frames through both endpoint conditions."""
    return [first + (last - first) * time / (frames - 1) for time in range(frames)]


def independent_prediction(first: float, last: float, frames: int) -> list[float]:
    """Per-frame estimates satisfy endpoints but have no cross-frame consistency term."""
    base = joint_prediction(first, last, frames)
    jitter = [0.0] + [0.24 if time % 2 else -0.24 for time in range(1, frames - 1)] + [0.0]
    return [value + delta for value, delta in zip(base, jitter)]


def endpoint_mae(values: list[float], first: float, last: float) -> float:
    return (abs(values[0] - first) + abs(values[-1] - last)) / 2


def trajectory_roughness(values: list[float]) -> float:
    second = [abs(values[i + 1] - 2 * values[i] + values[i - 1]) for i in range(1, len(values) - 1)]
    return sum(second) / len(second)


def temporal_flicker(values: list[float]) -> float:
    changes = [values[i + 1] - values[i] for i in range(len(values) - 1)]
    center = sum(changes) / len(changes)
    return sum(abs(change - center) for change in changes) / len(changes)


def attention_ledger(frames: int, height: int = 2, width: int = 2) -> dict:
    tokens = frames * height * width
    return {"frames": frames, "tokens": tokens, "full_attention_pairs": tokens * tokens}


def main() -> None:
    first, last, frames = 0.0, 1.0, 6
    joint = joint_prediction(first, last, frames)
    independent = independent_prediction(first, last, frames)
    latent_video = [frame(value) for value in joint]
    tokens = patchify_3d(latent_video)
    scaling = [attention_ledger(count) for count in (2, 4, 8)]
    metrics = {
        "full_attention_pairs": len(tokens) ** 2,
        "independent_flicker": round(temporal_flicker(independent), 6),
        "independent_roughness": round(trajectory_roughness(independent), 6),
        "joint_endpoint_mae": round(endpoint_mae(joint, first, last), 12),
        "joint_flicker": round(temporal_flicker(joint), 12),
        "joint_roughness": round(trajectory_roughness(joint), 12),
        "video_tokens": len(tokens),
    }
    checks = {
        "3d_positions_present": tokens[-1]["position"] == (5, 1, 1),
        "attention_is_quadratic": scaling[-1]["full_attention_pairs"] == 16 * scaling[0]["full_attention_pairs"],
        "endpoints_respected": metrics["joint_endpoint_mae"] == 0.0,
        "joint_is_smoother": metrics["joint_roughness"] < metrics["independent_roughness"],
        "joint_reduces_flicker": metrics["joint_flicker"] < metrics["independent_flicker"],
    }
    print("L0 spatiotemporal latent DiT")
    print(f"shape=(t=6,h=2,w=2) tokens={len(tokens)} full_attention_pairs={len(tokens) ** 2}")
    print(f"first_position={tokens[0]['position']} last_position={tokens[-1]['position']}")
    print(f"joint_trajectory={[round(value, 2) for value in joint]}")
    print(f"independent_trajectory={[round(value, 2) for value in independent]}")
    print(f"joint endpoint_mae={metrics['joint_endpoint_mae']:.3f} roughness={metrics['joint_roughness']:.3f} flicker={metrics['joint_flicker']:.3f}")
    print(f"independent roughness={metrics['independent_roughness']:.3f} flicker={metrics['independent_flicker']:.3f}")
    print(f"attention_scaling={scaling}")
    print(f"checks={sum(checks.values())}/{len(checks)}")
    boundary = "deterministic latent trajectory simulation; not a trained video generator"
    payload = {"metrics": metrics, "checks": checks, "evidence_boundary": boundary}
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]
    result = {
        "schema_version": "1.0",
        "module": "nano-video-dit/L0",
        **payload,
        "digest": digest,
    }
    print("RESULT_JSON=" + json.dumps(result, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
