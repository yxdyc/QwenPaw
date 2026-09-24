"""Ledger compressed bytes, decoded signals, and model positions.

This deterministic surrogate also tests adaptive sampling and split-safe dedup.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, replace


MIB = 1024**2


@dataclass(frozen=True)
class Video:
    duration_s: int = 60
    width: int = 1920
    height: int = 1080
    fps: int = 30
    compressed_bitrate_bps: int = 10_000_000


@dataclass(frozen=True)
class Event:
    name: str
    start_s: float
    duration_s: float


@dataclass(frozen=True)
class Asset:
    asset_id: str
    split: str
    byte_hash: str
    content_group: str


VIDEO = Video()
EVENTS = (
    Event("door_opens", 12.00, 1.20),
    Event("brief_flash", 31.21, 0.18),
    Event("small_gesture", 48.57, 0.22),
)
ASSETS = (
    Asset("original.mp4", "train", "sha256-a", "scene-7"),
    Asset("reencoded.webm", "eval", "sha256-b", "scene-7"),
    Asset("other.mp4", "eval", "sha256-c", "scene-9"),
)


def mib(value: int) -> float:
    return round(value / MIB, 3)


def visual_tokens(width: int, height: int, cell: int = 32) -> int:
    return math.ceil(width / cell) * math.ceil(height / cell)


def regular_times(duration_s: int, fps: int) -> tuple[float, ...]:
    return tuple(round(index / fps, 6) for index in range(duration_s * fps))


def window_times(start_s: float, end_s: float, fps: int) -> tuple[float, ...]:
    count = math.ceil((end_s - start_s) * fps)
    return tuple(
        round(start_s + index / fps, 6)
        for index in range(count + 1)
        if start_s + index / fps < end_s
    )


def event_recall(times: tuple[float, ...]) -> float:
    hits = sum(
        any(event.start_s <= time < event.start_s + event.duration_s for time in times)
        for event in EVENTS
    )
    return round(hits / len(EVENTS), 3)


def policy(name: str, times: tuple[float, ...], width: int, height: int) -> dict:
    tokens = len(times) * visual_tokens(width, height)
    return {
        "name": name,
        "frames": len(times),
        "frame_shape": [height, width],
        "selected_rgb_mib": mib(len(times) * width * height * 3),
        "visual_tokens": tokens,
        "event_recall": event_recall(times),
    }


def has_cross_split_group(assets: tuple[Asset, ...], field: str) -> bool:
    splits_by_group: dict[str, set[str]] = {}
    for asset in assets:
        key = getattr(asset, field)
        splits_by_group.setdefault(key, set()).add(asset.split)
    return any(len(splits) > 1 for splits in splits_by_group.values())


def main() -> None:
    compressed_video = VIDEO.compressed_bitrate_bps * VIDEO.duration_s // 8
    decoded_video = VIDEO.width * VIDEO.height * 3 * VIDEO.fps * VIDEO.duration_s
    compressed_image, decoded_image = 3 * MIB, 4096 * 3072 * 3

    dense_times = regular_times(VIDEO.duration_s, VIDEO.fps)
    uniform_times = regular_times(VIDEO.duration_s, 1)
    adaptive_set = set(uniform_times)
    for event in EVENTS:
        adaptive_set.update(
            window_times(
                max(0.0, event.start_s - 0.25),
                min(VIDEO.duration_s, event.start_s + event.duration_s + 0.25),
                8,
            )
        )
    adaptive_times = tuple(sorted(adaptive_set))

    dense = policy("dense_30fps_1080p", dense_times, 1920, 1080)
    uniform = policy("uniform_1fps_640x352", uniform_times, 640, 352)
    adaptive = policy("adaptive_1_to_8fps_640x352", adaptive_times, 640, 352)

    audio_pcm_bytes = 48_000 * 2 * 2 * VIDEO.duration_s
    model_pcm_samples = 16_000 * VIDEO.duration_s
    audio_codec_steps = 25 * VIDEO.duration_s

    exact_hash_finds_leak = has_cross_split_group(ASSETS, "byte_hash")
    content_group_finds_leak = has_cross_split_group(ASSETS, "content_group")
    split_by_group = {"scene-7": "train", "scene-9": "eval"}
    grouped_assets = tuple(
        replace(asset, split=split_by_group[asset.content_group]) for asset in ASSETS
    )

    checks = {
        "decode_amplification_gt_100x": decoded_video / compressed_video > 100,
        "uniform_sampling_misses_short_events": uniform["event_recall"] < 1.0,
        "adaptive_sampling_recovers_events": adaptive["event_recall"] == 1.0,
        "adaptive_tokens_lt_1pct_dense": adaptive["visual_tokens"] < dense["visual_tokens"] / 100,
        "global_plus_roi_lt_full_image_tokens": visual_tokens(1024, 768)
        + visual_tokens(512, 512) < visual_tokens(4096, 3072),
        "codec_steps_are_not_waveform_samples": audio_codec_steps < model_pcm_samples / 100,
        "byte_hash_misses_reencode": not exact_hash_finds_leak
        and content_group_finds_leak,
        "group_before_split_blocks_leak": not has_cross_split_group(grouped_assets, "content_group"),
    }

    payload = {
        "three_ledgers": {
            "video_file_mib": mib(compressed_video),
            "video_decoded_rgb_mib": mib(decoded_video),
            "video_decode_amplification": round(decoded_video / compressed_video, 2),
            "image_file_mib": mib(compressed_image),
            "image_decoded_rgb_mib": mib(decoded_image),
        },
        "video_policies": [dense, uniform, adaptive],
        "image_model_view": {
            "full_tokens": visual_tokens(4096, 3072),
            "global_plus_roi_tokens": visual_tokens(1024, 768) + visual_tokens(512, 512),
        },
        "audio_model_view": {
            "source_pcm_mib": mib(audio_pcm_bytes),
            "resampled_waveform_samples": model_pcm_samples,
            "codec_temporal_steps": audio_codec_steps,
        },
        "dedup": {
            "exact_hash_detects_cross_split_copy": exact_hash_finds_leak,
            "content_group_detects_cross_split_copy": content_group_finds_leak,
            "grouped_split_has_leak": has_cross_split_group(grouped_assets, "content_group"),
        },
        "checks": checks,
        "evidence_boundary": "Deterministic arithmetic surrogate: event windows and "
        "content groups are oracle labels, not learned production detectors.",
    }
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    payload["digest"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]

    print("THREE LEDGERS")
    print(f"video: file={mib(compressed_video):.3f} MiB -> "
          f"decoded={mib(decoded_video):.3f} MiB ({decoded_video / compressed_video:.2f}x)")
    for item in (dense, uniform, adaptive):
        print(f"{item['name']}: frames={item['frames']}, tokens={item['visual_tokens']}, "
              f"event_recall={item['event_recall']:.3f}")
    print(f"dedup: exact_hash_cross_split={exact_hash_finds_leak}, "
          f"semantic_group_cross_split={content_group_finds_leak}")
    print("RESULT_JSON=" + json.dumps(payload, ensure_ascii=False, sort_keys=True))
    if not all(checks.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
