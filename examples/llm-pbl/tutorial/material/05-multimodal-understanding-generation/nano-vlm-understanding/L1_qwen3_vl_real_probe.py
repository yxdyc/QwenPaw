"""L1: run a real Qwen3-VL-2B checkpoint on six synthetic visual diagnostics."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import statistics
import sys
import tempfile
import time
from pathlib import Path


DEFAULT_MODEL = "Qwen/Qwen3-VL-2B-Instruct"
DEFAULT_REVISION = "89644892e4d85e24eaac8bacfd4f463576704203"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--model-path", help="optional local snapshot; the path is never emitted")
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument(
        "--artifact-source",
        choices=("huggingface_hub", "qwen_modelscope"),
        default="huggingface_hub",
    )
    parser.add_argument("--artifact-source-revision")
    parser.add_argument("--expected-manifest-sha256")
    parser.add_argument("--expected-weight-sha256")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--repeat", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument(
        "--self-test-metrics",
        action="store_true",
        help="validate evaluator semantics without torch, model weights, or a GPU",
    )
    return parser.parse_args()


def canonical(text: str) -> str:
    return re.sub(r"[^A-Z0-9]+", " ", text.upper()).strip()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_local_snapshot(
    directory: Path,
    expected_weight_sha256: str | None,
    expected_manifest_sha256: str | None,
) -> dict:
    if not directory.is_dir():
        raise SystemExit("--model-path must point to an existing local directory")
    if not expected_weight_sha256:
        raise SystemExit("--model-path requires --expected-weight-sha256")
    if not expected_manifest_sha256:
        raise SystemExit("--model-path requires --expected-manifest-sha256")
    files = sorted(path for path in directory.iterdir() if path.is_file())
    weight = directory / "model.safetensors"
    if weight not in files:
        raise SystemExit("local snapshot must contain one model.safetensors file")
    file_hashes = {path.name: sha256_file(path) for path in files}
    manifest_sha256 = hashlib.sha256(
        json.dumps(file_hashes, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    expected = expected_weight_sha256.lower()
    expected_manifest = expected_manifest_sha256.lower()
    actual = file_hashes[weight.name]
    if actual != expected:
        raise SystemExit(f"model.safetensors SHA256 mismatch: expected {expected}, got {actual}")
    if manifest_sha256 != expected_manifest:
        raise SystemExit(
            f"snapshot manifest SHA256 mismatch: expected {expected_manifest}, got {manifest_sha256}"
        )
    return {
        "file_count": len(files),
        "manifest_sha256": manifest_sha256,
        "manifest_sha256_expected": expected_manifest,
        "manifest_sha256_verified": True,
        "weight_sha256": actual,
        "weight_sha256_expected": expected,
        "weight_sha256_verified": True,
    }


def score_swap(records: list[dict], repeat: int) -> tuple[bool, bool]:
    """Return (answer_changed, answer_changed_and_both_sides_correct)."""
    triangle = [
        record for record in records if record["id"] == "swap_triangle" and "error" not in record
    ]
    circle = [record for record in records if record["id"] == "swap_circle" and "error" not in record]
    sensitive = (
        len(triangle) == repeat
        and len(circle) == repeat
        and all(left["normalized"] != right["normalized"] for left, right in zip(triangle, circle))
    )
    correct = sensitive and all(record["normalized_match"] for record in triangle + circle)
    return sensitive, correct


def token_ledger_complete(record: dict) -> bool:
    return all(
        record.get(key) is not None
        for key in ("raw_visual_patches", "visual_tokens_after_merge", "input_tokens")
    )


def token_merge_exact(record: dict) -> bool:
    merge_size = record.get("spatial_merge_size")
    return (
        token_ledger_complete(record)
        and merge_size is not None
        and merge_size > 0
        and record["raw_visual_patches"]
        == record["visual_tokens_after_merge"] * merge_size**2
    )


def self_test_metrics() -> None:
    def pair(triangle: str, circle: str, triangle_ok: bool, circle_ok: bool) -> list[dict]:
        return [
            {"id": "swap_triangle", "normalized": triangle, "normalized_match": triangle_ok},
            {"id": "swap_circle", "normalized": circle, "normalized_match": circle_ok},
        ]

    exact_ledger = {
        "input_tokens": 12,
        "raw_visual_patches": 16,
        "spatial_merge_size": 2,
        "visual_tokens_after_merge": 4,
    }
    wrong_ledger = {**exact_ledger, "visual_tokens_after_merge": 3}
    checks = {
        "normalized_not_strict": canonical("code-7319") == canonical("CODE 7319")
        and "code-7319" != "CODE 7319",
        "sensitive_and_correct": score_swap(pair("TRIANGLE", "CIRCLE", True, True), 1)
        == (True, True),
        "sensitive_but_wrong": score_swap(pair("CIRCLE", "TRIANGLE", False, False), 1)
        == (True, False),
        "same_answer_not_sensitive": score_swap(pair("TRIANGLE", "TRIANGLE", True, False), 1)
        == (False, False),
        "token_merge_exact": token_merge_exact(exact_ledger),
        "token_merge_mismatch_detected": not token_merge_exact(wrong_ledger),
    }
    payload = {
        "checks": checks,
        "evidence_boundary": "evaluator semantics only; no image processing or model inference",
        "module": "nano-vlm-understanding/L1-metric-self-test",
        "schema_version": "1.0",
    }
    print("METRIC_SELF_TEST_JSON=" + json.dumps(payload, sort_keys=True, separators=(",", ":")))
    if not all(checks.values()):
        raise SystemExit(1)


GLYPHS = {
    " ": ("00000",) * 7,
    "1": ("00100", "01100", "00100", "00100", "00100", "00100", "01110"),
    "3": ("11110", "00001", "00001", "01110", "00001", "00001", "11110"),
    "7": ("11111", "00001", "00010", "00100", "01000", "01000", "01000"),
    "9": ("01110", "10001", "10001", "01111", "00001", "00010", "11100"),
    "C": ("01111", "10000", "10000", "10000", "10000", "10000", "01111"),
    "D": ("11110", "10001", "10001", "10001", "10001", "10001", "11110"),
    "E": ("11111", "10000", "10000", "11110", "10000", "10000", "11111"),
    "O": ("01110", "10001", "10001", "10001", "10001", "10001", "01110"),
}


def draw_bitmap_text(draw, origin: tuple[int, int], text: str, scale: int = 10) -> None:
    x0, y0 = origin
    for char_index, char in enumerate(text):
        glyph = GLYPHS[char]
        for row, bits in enumerate(glyph):
            for col, bit in enumerate(bits):
                if bit == "1":
                    x = x0 + char_index * 6 * scale + col * scale
                    y = y0 + row * scale
                    draw.rectangle((x, y, x + scale - 1, y + scale - 1), fill="black")


def make_images(directory: Path, image_module, image_draw) -> dict[str, Path]:
    paths = {}

    image = image_module.new("RGB", (768, 448), "white")
    draw = image_draw.Draw(image)
    draw.rectangle((30, 30, 738, 418), outline="black", width=5)
    draw_bitmap_text(draw, (85, 180), "CODE 7319", scale=12)
    paths["ocr"] = directory / "ocr.png"
    image.save(paths["ocr"])

    image = image_module.new("RGB", (768, 448), "white")
    draw = image_draw.Draw(image)
    draw.rectangle((75, 130, 255, 310), fill=(220, 30, 30))
    draw.ellipse((510, 130, 690, 310), fill=(30, 80, 220))
    paths["spatial"] = directory / "spatial.png"
    image.save(paths["spatial"])

    image = image_module.new("RGB", (768, 448), "white")
    draw = image_draw.Draw(image)
    for center in (170, 384, 598):
        draw.polygon(((center, 90), (center - 85, 330), (center + 85, 330)), fill=(30, 170, 70))
    paths["count"] = directory / "count.png"
    image.save(paths["count"])

    image = image_module.new("RGB", (768, 448), "white")
    draw = image_draw.Draw(image)
    draw.polygon(((384, 70), (180, 360), (588, 360)), fill=(220, 30, 30))
    paths["swap_triangle"] = directory / "swap_triangle.png"
    image.save(paths["swap_triangle"])

    image = image_module.new("RGB", (768, 448), "white")
    draw = image_draw.Draw(image)
    draw.ellipse((180, 35, 588, 413), fill=(30, 80, 220))
    paths["swap_circle"] = directory / "swap_circle.png"
    image.save(paths["swap_circle"])

    image = image_module.new("RGB", (768, 448), (210, 210, 210))
    paths["refusal"] = directory / "refusal.png"
    image.save(paths["refusal"])
    return paths


def cases(paths: dict[str, Path]) -> list[dict]:
    return [
        {
            "id": "ocr_exact",
            "skill": "ocr",
            "image": paths["ocr"],
            "prompt": "Read the printed code. Answer only the code, including the word CODE.",
            "expected": "CODE 7319",
        },
        {
            "id": "spatial_left",
            "skill": "spatial",
            "image": paths["spatial"],
            "prompt": "Which colored shape is on the left? Answer only RED SQUARE or BLUE CIRCLE.",
            "expected": "RED SQUARE",
        },
        {
            "id": "count_triangles",
            "skill": "count",
            "image": paths["count"],
            "prompt": "How many green triangles are visible? Answer only one integer.",
            "expected": "3",
        },
        {
            "id": "swap_triangle",
            "skill": "image_swap",
            "image": paths["swap_triangle"],
            "prompt": "What single shape is visible? Answer only TRIANGLE or CIRCLE.",
            "expected": "TRIANGLE",
        },
        {
            "id": "swap_circle",
            "skill": "image_swap",
            "image": paths["swap_circle"],
            "prompt": "What single shape is visible? Answer only TRIANGLE or CIRCLE.",
            "expected": "CIRCLE",
        },
        {
            "id": "no_evidence_refusal",
            "skill": "refusal",
            "image": paths["refusal"],
            "prompt": "What serial number is printed? If none is visible, answer only NOT VISIBLE.",
            "expected": "NOT VISIBLE",
        },
    ]


def infer(model, processor, torch, image_module, case: dict, device: str, max_new_tokens: int) -> dict:
    end_to_end_started = time.perf_counter()
    image = image_module.open(case["image"]).convert("RGB")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": case["prompt"]},
            ],
        }
    ]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to(device)
    preprocessing_s = time.perf_counter() - end_to_end_started
    generation_started = time.perf_counter()
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            use_cache=True,
        )
    generation_s = time.perf_counter() - generation_started
    end_to_end_s = time.perf_counter() - end_to_end_started
    generated = output[:, inputs["input_ids"].shape[1] :]
    text = processor.batch_decode(
        generated,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()
    normalized = canonical(text)
    grid = inputs.get("image_grid_thw")
    raw_patches = int(grid.prod(dim=1).sum().item()) if grid is not None else None
    merge_size = getattr(getattr(processor, "image_processor", None), "merge_size", None)
    try:
        merge_size = int(merge_size)
    except (TypeError, ValueError):
        merge_size = None
    visual_tokens = (
        raw_patches // merge_size**2
        if raw_patches is not None and merge_size is not None and merge_size > 0
        else None
    )
    generated_tokens = int(generated.shape[1])
    return {
        "answer": text,
        "end_to_end_s": round(end_to_end_s, 3),
        "expected": case["expected"],
        "generated_tokens": generated_tokens,
        "generation_s": round(generation_s, 3),
        "input_tokens": int(inputs["input_ids"].shape[1]),
        "normalized": normalized,
        "normalized_match": normalized == canonical(case["expected"]),
        "preprocessing_s": round(preprocessing_s, 3),
        "raw_visual_patches": raw_patches,
        "spatial_merge_size": merge_size,
        "strict_format_match": text == case["expected"],
        "tokens_per_s": round(generated_tokens / generation_s, 3),
        "visual_tokens_after_merge": visual_tokens,
    }


def main() -> None:
    args = parse_args()
    if args.repeat < 1:
        raise SystemExit("--repeat must be >= 1")
    if args.self_test_metrics:
        self_test_metrics()
        return
    if args.model_path and not args.artifact_source_revision:
        raise SystemExit("--model-path requires --artifact-source-revision")
    artifact_verification_started = time.perf_counter()
    local_snapshot = (
        verify_local_snapshot(
            Path(args.model_path),
            args.expected_weight_sha256,
            args.expected_manifest_sha256,
        )
        if args.model_path
        else None
    )
    artifact_verification_s = time.perf_counter() - artifact_verification_started
    try:
        import torch
        import torchvision
        import transformers
        from PIL import Image, ImageDraw
        from transformers import AutoModelForImageTextToText, AutoProcessor
    except ImportError as error:
        raise SystemExit(
            "missing L1 dependencies; install torch, torchvision, transformers>=4.57, "
            "accelerate, and pillow"
        ) from error
    if not torch.cuda.is_available() or not args.device.startswith("cuda"):
        raise SystemExit("this real-model L1 requires an explicitly selected CUDA device")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.set_device(args.device)
    torch.cuda.reset_peak_memory_stats(args.device)
    load_started = time.perf_counter()
    load_source = args.model_path or args.model
    load_kwargs = {"local_files_only": True} if args.model_path else {
        "local_files_only": args.local_files_only,
        "revision": args.revision,
    }
    processor = AutoProcessor.from_pretrained(load_source, **load_kwargs)
    model = AutoModelForImageTextToText.from_pretrained(
        load_source,
        **load_kwargs,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    ).to(args.device)
    model.eval()
    load_s = time.perf_counter() - load_started

    records = []
    with tempfile.TemporaryDirectory(prefix="qwen3_vl_l1_") as temp:
        probe_cases = cases(make_images(Path(temp), Image, ImageDraw))
        for repeat in range(args.repeat):
            for case in probe_cases:
                try:
                    record = infer(model, processor, torch, Image, case, args.device, args.max_new_tokens)
                    record.update({"id": case["id"], "repeat": repeat + 1, "skill": case["skill"]})
                except Exception as error:  # Preserve failures in the denominator and report.
                    record = {
                        "answer": "",
                        "end_to_end_s": None,
                        "error": f"{type(error).__name__}: {error}",
                        "expected": case["expected"],
                        "generated_tokens": 0,
                        "generation_s": None,
                        "id": case["id"],
                        "input_tokens": None,
                        "normalized": "",
                        "normalized_match": False,
                        "preprocessing_s": None,
                        "raw_visual_patches": None,
                        "repeat": repeat + 1,
                        "skill": case["skill"],
                        "spatial_merge_size": None,
                        "strict_format_match": False,
                        "tokens_per_s": None,
                        "visual_tokens_after_merge": None,
                    }
                records.append(record)

    completed = [record for record in records if "error" not in record]
    skills = sorted({record["skill"] for record in records})
    skill_em = {
        skill: round(
            sum(record["normalized_match"] for record in records if record["skill"] == skill)
            / sum(record["skill"] == skill for record in records),
            3,
        )
        for skill in skills
    }
    by_id = {
        case_id: [record["normalized"] for record in completed if record["id"] == case_id]
        for case_id in {record["id"] for record in records}
    }
    stable = all(len(values) == args.repeat and len(set(values)) == 1 for values in by_id.values())
    swap_sensitive, swap_correct = score_swap(records, args.repeat)
    metrics = {
        "completion_rate": round(len(completed) / len(records), 3),
        "normalized_semantic_accuracy": round(
            sum(record["normalized_match"] for record in records) / len(records), 3
        ),
        "prediction_stability": stable,
        "skill_normalized_accuracy": skill_em,
        "strict_format_accuracy": round(
            sum(record["strict_format_match"] for record in records) / len(records), 3
        ),
        "swap_counterfactual_correct": swap_correct,
        "swap_counterfactual_sensitivity": swap_sensitive,
    }
    performance = {
        "median_end_to_end_s": round(
            statistics.median(record["end_to_end_s"] for record in completed), 3
        )
        if completed
        else None,
        "median_generation_s": round(
            statistics.median(record["generation_s"] for record in completed), 3
        )
        if completed
        else None,
        "median_tokens_per_s": round(
            statistics.median(record["tokens_per_s"] for record in completed), 3
        )
        if completed
        else None,
    }
    checks = {
        "all_attempts_counted": len(records) == 6 * args.repeat,
        "all_completed": metrics["completion_rate"] == 1.0,
        "all_answers_nonempty": all(record["normalized"] for record in completed),
        "repeat_coverage_complete": all(len(values) == args.repeat for values in by_id.values()),
        "six_case_ids_present": len(by_id) == 6,
        "model_artifact_verified": (
            local_snapshot["weight_sha256_verified"]
            and local_snapshot["manifest_sha256_verified"]
        )
        if local_snapshot
        else getattr(model.config, "_commit_hash", None) == args.revision,
        "visual_token_ledger_complete": len(completed) == len(records)
        and all(token_ledger_complete(record) for record in completed),
        "visual_token_merge_exact": len(completed) == len(records)
        and all(token_merge_exact(record) for record in completed),
    }
    evidence = {
        "artifact_source": args.artifact_source,
        "artifact_source_revision": args.artifact_source_revision,
        "artifact_verification_s": round(artifact_verification_s, 3),
        "bf16_supported": torch.cuda.is_bf16_supported(),
        "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(args.device),
        "device_total_memory_gib": round(
            torch.cuda.get_device_properties(args.device).total_memory / 2**30, 3
        ),
        "gpu_count_visible": torch.cuda.device_count(),
        "load_s": round(load_s, 3),
        "model": args.model,
        "model_load_mode": "verified_local_snapshot" if local_snapshot else "huggingface_hub",
        "peak_vram_allocated_gib": round(torch.cuda.max_memory_allocated(args.device) / 2**30, 3),
        "peak_vram_reserved_gib": round(torch.cuda.max_memory_reserved(args.device) / 2**30, 3),
        "revision_requested": args.revision,
        "revision_resolved": getattr(model.config, "_commit_hash", None),
        "snapshot": local_snapshot,
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "transformers": transformers.__version__,
    }
    stable_payload = {
        "artifact_identity": local_snapshot
        or {"revision_resolved": evidence["revision_resolved"]},
        "checks": checks,
        "metrics": metrics,
        "normalized_answers": by_id,
        "token_ledger_first_pass": [
            {
                "id": record["id"],
                "input_tokens": record["input_tokens"],
                "raw_visual_patches": record["raw_visual_patches"],
                "visual_tokens_after_merge": record["visual_tokens_after_merge"],
            }
            for record in records
            if record["repeat"] == 1
        ],
    }
    digest = hashlib.sha256(json.dumps(stable_payload, sort_keys=True).encode()).hexdigest()[:16]
    first_pass = [record for record in records if record["repeat"] == 1]
    print("L1 Qwen3-VL real visual diagnostics")
    print(
        f"model={args.model} revision_requested={args.revision} "
        f"revision_resolved={evidence['revision_resolved']} load_mode={evidence['model_load_mode']}"
    )
    if local_snapshot:
        print(
            f"artifact_source={args.artifact_source} files={local_snapshot['file_count']} "
            f"weight_sha256={local_snapshot['weight_sha256']} "
            f"manifest_sha256={local_snapshot['manifest_sha256']}"
        )
    print(f"device={evidence['device']} torch={evidence['torch']} transformers={evidence['transformers']}")
    for record in first_pass:
        print(
            f"{record['id']}: answer={record['answer']!r} "
            f"normalized_match={record['normalized_match']} strict_format={record['strict_format_match']} "
            f"raw_patches={record['raw_visual_patches']} "
            f"visual_tokens={record['visual_tokens_after_merge']} input_tokens={record['input_tokens']} "
            f"e2e_s={record['end_to_end_s']}"
        )
    print(f"metrics={json.dumps(metrics, sort_keys=True)}")
    print(f"performance={json.dumps(performance, sort_keys=True)}")
    print(
        f"checks={sum(checks.values())}/{len(checks)} "
        f"peak_vram_allocated_gib={evidence['peak_vram_allocated_gib']} load_s={evidence['load_s']}"
    )
    result = {
        "schema_version": "1.2",
        "module": "nano-vlm-understanding/L1",
        "metrics": metrics,
        "checks": checks,
        "digest": digest,
        "evidence": evidence,
        "performance": performance,
        "evidence_boundary": "real 2B checkpoint on six generated diagnostics; a local ModelScope transfer is accepted only when its weight SHA matches the official pinned-HF artifact and its full local manifest is recorded; normalized accuracy is not strict-format accuracy, swap sensitivity is not correctness, and this is not a natural-image benchmark",
        "records": records,
    }
    print("RESULT_JSON=" + json.dumps(result, sort_keys=True, separators=(",", ":")))
    if not all(checks.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
