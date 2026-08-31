"""L0: latent DiT contracts and rectified-flow direction with an oracle velocity."""

from __future__ import annotations

import hashlib
import json
import math


def patchify_latent(latent: list[list[float]]) -> list[float]:
    return [value for row in latent for value in row]


def adaln(tokens: list[float], time: float, text_condition: float) -> list[float]:
    """Tiny AdaLN: normalize latent tokens, then apply time/text scale and shift."""
    center = sum(tokens) / len(tokens)
    variance = sum((value - center) ** 2 for value in tokens) / len(tokens)
    normalized = [(value - center) / math.sqrt(variance + 1e-6) for value in tokens]
    scale = 1.0 + 0.1 * time
    shift = 0.05 * text_condition
    return [scale * value + shift for value in normalized]


def flow_target(noise: list[float], data: list[float]) -> list[float]:
    """For z_t=(1-t)z_0+t z_1, the oracle target is dz_t/dt=z_1-z_0."""
    return [target - source for source, target in zip(noise, data)]


def cfg_velocity(noise: list[float], data: list[float], scale: float) -> list[float]:
    unconditioned = [0.5] * len(data)
    velocity_u = flow_target(noise, unconditioned)
    velocity_c = flow_target(noise, data)
    return [u + scale * (c - u) for u, c in zip(velocity_u, velocity_c)]


def euler(noise: list[float], velocity: list[float], steps: int, sign: float = 1.0) -> list[float]:
    state = list(noise)
    step_size = 1.0 / steps
    for _ in range(steps):
        state = [value + sign * step_size * speed for value, speed in zip(state, velocity)]
    return state


def mae(left: list[float], right: list[float]) -> float:
    return sum(abs(a - b) for a, b in zip(left, right)) / len(left)


def attribute_hit(latent: list[float]) -> float:
    expected = [False, True, False, True]
    observed = [value >= 0.5 for value in latent]
    return sum(a == b for a, b in zip(expected, observed)) / len(expected)


def main() -> None:
    pixels = [[0.2 if col < 4 else 0.8 for col in range(8)] for _row in range(8)]
    latent = [[pixels[row * 4][col * 4] for col in range(2)] for row in range(2)]
    data = patchify_latent(latent)
    noise = [0.9, 0.1, 0.7, 0.3]
    conditioned = adaln(noise, time=0.25, text_condition=1.0)
    velocity = cfg_velocity(noise, data, scale=1.0)
    reconstructed = euler(noise, velocity, steps=8)
    wrong_sign = euler(noise, velocity, steps=8, sign=-1.0)
    strong_cfg = euler(noise, cfg_velocity(noise, data, scale=3.0), steps=8)
    metrics = {
        "condition_attribute_hit_rate": attribute_hit(reconstructed),
        "latent_reconstruction_mae": round(mae(reconstructed, data), 12),
        "pixel_to_latent_token_ratio": len(pixels) * len(pixels[0]) / len(data),
        "strong_cfg_mae": round(mae(strong_cfg, data), 6),
        "wrong_sign_mae": round(mae(wrong_sign, data), 6),
    }
    checks = {
        "adaln_injects_condition": conditioned != noise,
        "cfg_one_reconstructs": metrics["latent_reconstruction_mae"] == 0.0,
        "condition_attribute_matches": metrics["condition_attribute_hit_rate"] == 1.0,
        "latent_is_cheaper": metrics["pixel_to_latent_token_ratio"] > 1.0,
        "strong_cfg_overshoots": metrics["strong_cfg_mae"] > 0.0,
        "wrong_direction_fails": metrics["wrong_sign_mae"] > 0.5,
    }
    print("L0 rectified-flow DiT oracle")
    print(f"pixel_tokens=64 latent_tokens={len(data)} ratio={metrics['pixel_to_latent_token_ratio']:.1f}x")
    print(f"latent_patches={data}")
    print(f"AdaLN_probe={[round(value, 3) for value in conditioned]}")
    print(f"Euler_CFG1_reconstruction_mae={metrics['latent_reconstruction_mae']:.12f}")
    print(f"condition_attribute_hit_rate={metrics['condition_attribute_hit_rate']:.3f}")
    print(f"wrong_sign_mae={metrics['wrong_sign_mae']:.3f}")
    print(f"CFG3_overshoot_mae={metrics['strong_cfg_mae']:.3f}")
    print(f"checks={sum(checks.values())}/{len(checks)}")
    boundary = "explicit oracle velocity; validates contracts and direction, not learned generation"
    payload = {"metrics": metrics, "checks": checks, "evidence_boundary": boundary}
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]
    result = {
        "schema_version": "1.0",
        "module": "nano-image-dit/L0",
        **payload,
        "digest": digest,
    }
    print("RESULT_JSON=" + json.dumps(result, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
