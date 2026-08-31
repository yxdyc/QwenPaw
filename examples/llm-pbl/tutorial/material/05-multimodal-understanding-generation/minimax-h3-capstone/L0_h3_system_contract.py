"""L0: executable teaching contract for MiniMax H3's packed omni-modal system."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Context:
    task: str
    prompt: str
    duration_s: int = 4
    fps: int = 24
    height: int = 768
    width: int = 1344
    audio_hz: int = 32000
    audio_channels: int = 2
    first_frame: bool = True
    last_frame: bool = True
    reference_media: bool = False


@dataclass(frozen=True)
class TeachingContextIR:
    surrogate: bool
    task: str
    shot: str
    motion: str
    sound_event: str


def build_ir(context: Context) -> TeachingContextIR:
    """Course-owned surrogate: deliberately not MiniMax's private Context-IR schema."""
    return TeachingContextIR(True, context.task, "wide-to-close", "left-to-right", "impact-at-2s")


def validate_input(context: Context) -> None:
    contracts = {
        "FL2VA": not context.reference_media,
        "Ref2VA": context.reference_media,
    }
    if context.task not in contracts or not contracts[context.task]:
        raise ValueError(f"invalid {context.task} input contract")


def video_tokens(context: Context) -> int:
    """Teaching ledger: spatial 16x, temporal 4x, d24 channels, then patch 1x2x2."""
    latent_t = math.ceil(context.duration_s * context.fps / 4)
    latent_h = math.ceil(context.height / 16)
    latent_w = math.ceil(context.width / 16)
    return latent_t * math.ceil(latent_h / 2) * math.ceil(latent_w / 2)


def audio_tokens(context: Context) -> int:
    """Teaching ledger for 40 Hz latent tokens per audio channel."""
    return context.duration_s * 40 * context.audio_channels


def pack_rows(context: Context, ir: TeachingContextIR) -> list[dict]:
    rows = [
        {"row_index": 0, "tag": "text", "tokens": len(context.prompt.split()), "rope": (0, 0, 0)},
        {"row_index": 1, "tag": "video", "tokens": video_tokens(context), "rope": (24, 24, 42)},
        {"row_index": 2, "tag": "audio", "tokens": audio_tokens(context), "rope": (160, 0, 1)},
    ]
    if context.reference_media:
        rows.insert(1, {"row_index": 1, "tag": "reference", "tokens": 64, "rope": (0, 8, 8)})
        for index, row in enumerate(rows):
            row["row_index"] = index
    assert ir.surrogate
    return rows


def validate_rows(rows: list[dict]) -> None:
    required = {"text", "video", "audio"}
    if [row.get("row_index") for row in rows] != list(range(len(rows))):
        raise ValueError("row indices must be contiguous")
    if any(not row.get("tag") for row in rows) or not required.issubset({row["tag"] for row in rows}):
        raise ValueError("missing modality tag")


def joint_flow(rows: list[dict], video_shift: int = 12, audio_shift: int = 3) -> dict:
    """One packed full-self-attention call, then modality-specific flow heads."""
    validate_rows(rows)
    if (video_shift, audio_shift) != (12, 3):
        raise ValueError("scheduler shifts were mixed")
    return {
        "attention": "full_self_attention",
        "cfg_distilled_forward_calls": 1,
        "video_scheduler": {"kind": "rectified_flow", "shift": video_shift},
        "audio_scheduler": {"kind": "rectified_flow", "shift": audio_shift},
    }


def decode_boundary(requested: str, execution: str) -> str:
    allowed = {("768p", "local"): "local_decode", ("2K", "hosted"): "hosted_regenerate"}
    if (requested, execution) not in allowed:
        raise ValueError("hosted module was misreported as local")
    return allowed[(requested, execution)]


def catches(callback) -> bool:
    try:
        callback()
    except ValueError:
        return True
    return False


def main() -> None:
    context = Context("FL2VA", "a robot crosses the frame as a bell strikes")
    validate_input(context)
    ir = build_ir(context)
    rows = pack_rows(context, ir)
    flow = joint_flow(rows)
    local = decode_boundary("768p", "local")
    hosted = decode_boundary("2K", "hosted")
    broken_index = [dict(row) for row in rows]
    broken_index[1]["row_index"] = 9
    missing_tag = [dict(row) for row in rows]
    missing_tag[2].pop("tag")
    checks = {
        "FL2VA_contract": True,
        "Ref2VA_contract": not catches(lambda: validate_input(Context("Ref2VA", "use reference", reference_media=True))),
        "T2VA_via_FL2VA_contract": not catches(
            lambda: validate_input(Context("FL2VA", "text only", first_frame=False, last_frame=False))
        ),
        "hosted_as_local_rejected": catches(lambda: decode_boundary("2K", "local")),
        "missing_tag_rejected": catches(lambda: validate_rows(missing_tag)),
        "row_index_error_rejected": catches(lambda: validate_rows(broken_index)),
        "scheduler_mix_rejected": catches(lambda: joint_flow(rows, 3, 12)),
        "surrogate_marked": ir.surrogate is True,
    }
    metrics = {
        "audio_latent_tokens": audio_tokens(context),
        "cfg_distilled_forward_calls": flow["cfg_distilled_forward_calls"],
        "packed_tokens": sum(row["tokens"] for row in rows),
        "video_latent_tokens": video_tokens(context),
    }
    print("L0 MiniMax H3 system contract")
    print("pipeline=Context -> TeachingContextIR -> packed rows -> joint flow -> decode/regenerate")
    print("TeachingContextIR=" + json.dumps(asdict(ir), sort_keys=True, separators=(",", ":")))
    print(f"contract={context.task} rows={[(row['row_index'], row['tag'], row['tokens']) for row in rows]}")
    print(f"3D_MM_RoPE={[row['rope'] for row in rows]}")
    print("attention=full_self_attention cross_attention=false")
    print("schedulers=video:rectified_flow/shift12 audio:rectified_flow/shift3")
    print(f"CFG_distilled_forward_calls={flow['cfg_distilled_forward_calls']}")
    print(f"boundaries=768p:{local} 2K:{hosted}")
    print(f"contract_and_failure_checks={sum(checks.values())}/{len(checks)}")
    boundary = "surrogate contract only; not official Context-IR, weights, hosted 2K, or production proof"
    payload = {"metrics": metrics, "checks": checks, "evidence_boundary": boundary}
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]
    result = {
        "schema_version": "1.0",
        "module": "minimax-h3-capstone/L0",
        **payload,
        "digest": digest,
    }
    print("RESULT_JSON=" + json.dumps(result, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
