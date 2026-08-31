"""L1: run a real Qwen3-VL-2B checkpoint on six synthetic visual diagnostics."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
import tempfile
import time
from pathlib import Path


DEFAULT_MODEL = "Qwen/Qwen3-VL-2B-Instruct"
DEFAULT_REVISION = "89644892e4d85e24eaac8bacfd4f463576704203"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--repeat", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def canonical(text: str) -> str:
    return re.sub(r"[^A-Z0-9]+", " ", text.upper()).strip()


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
    started = time.perf_counter()
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            use_cache=True,
        )
    elapsed = time.perf_counter() - started
    generated = output[:, inputs["input_ids"].shape[1] :]
    text = processor.batch_decode(
        generated,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()
    normalized = canonical(text)
    grid = inputs.get("image_grid_thw")
    raw_patches = int(grid.prod(dim=1).sum().item()) if grid is not None else None
    return {
        "answer": text,
        "correct": normalized == canonical(case["expected"]),
        "elapsed_s": round(elapsed, 3),
        "normalized": normalized,
        "raw_visual_patches": raw_patches,
    }


def main() -> None:
    args = parse_args()
    if args.repeat < 1:
        raise SystemExit("--repeat must be >= 1")
    try:
        import torch
        import transformers
        from PIL import Image, ImageDraw
        from transformers import AutoModelForImageTextToText, AutoProcessor
    except ImportError as error:
        raise SystemExit(
            "missing L1 dependencies; install torch, transformers>=4.57, accelerate, and pillow"
        ) from error
    if not torch.cuda.is_available() or not args.device.startswith("cuda"):
        raise SystemExit("this real-model L1 requires an explicitly selected CUDA device")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.set_device(args.device)
    torch.cuda.reset_peak_memory_stats(args.device)
    load_started = time.perf_counter()
    processor = AutoProcessor.from_pretrained(
        args.model,
        revision=args.revision,
        local_files_only=args.local_files_only,
    )
    model = AutoModelForImageTextToText.from_pretrained(
        args.model,
        revision=args.revision,
        local_files_only=args.local_files_only,
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
                        "correct": False,
                        "elapsed_s": None,
                        "error": f"{type(error).__name__}: {error}",
                        "id": case["id"],
                        "normalized": "",
                        "raw_visual_patches": None,
                        "repeat": repeat + 1,
                        "skill": case["skill"],
                    }
                records.append(record)

    completed = [record for record in records if "error" not in record]
    skills = sorted({record["skill"] for record in records})
    skill_em = {
        skill: round(
            sum(record["correct"] for record in records if record["skill"] == skill)
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
    triangle_answers = by_id.get("swap_triangle", [])
    circle_answers = by_id.get("swap_circle", [])
    swap_sensitive = (
        len(triangle_answers) == args.repeat
        and len(circle_answers) == args.repeat
        and all(left != right for left, right in zip(triangle_answers, circle_answers))
    )
    metrics = {
        "completion_rate": round(len(completed) / len(records), 3),
        "exact_match": round(sum(record["correct"] for record in records) / len(records), 3),
        "prediction_stability": stable,
        "skill_exact_match": skill_em,
        "swap_counterfactual_sensitivity": swap_sensitive,
    }
    checks = {
        "all_attempts_counted": len(records) == 6 * args.repeat,
        "all_completed": metrics["completion_rate"] == 1.0,
        "all_answers_nonempty": all(record["normalized"] for record in completed),
        "repeat_coverage_complete": all(len(values) == args.repeat for values in by_id.values()),
        "six_case_ids_present": len(by_id) == 6,
    }
    evidence = {
        "device": torch.cuda.get_device_name(args.device),
        "load_s": round(load_s, 3),
        "model": args.model,
        "peak_vram_gib": round(torch.cuda.max_memory_allocated(args.device) / 2**30, 3),
        "revision_requested": args.revision,
        "revision_resolved": getattr(model.config, "_commit_hash", None),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
    }
    stable_payload = {
        "checks": checks,
        "metrics": metrics,
        "normalized_answers": by_id,
        "revision": evidence["revision_resolved"] or args.revision,
    }
    digest = hashlib.sha256(json.dumps(stable_payload, sort_keys=True).encode()).hexdigest()[:16]
    first_pass = [record for record in records if record["repeat"] == 1]
    print("L1 Qwen3-VL real visual diagnostics")
    print(f"model={args.model} revision={evidence['revision_resolved'] or args.revision}")
    print(f"device={evidence['device']} torch={evidence['torch']} transformers={evidence['transformers']}")
    for record in first_pass:
        print(
            f"{record['id']}: answer={record['answer']!r} correct={record['correct']} "
            f"raw_visual_patches={record['raw_visual_patches']} elapsed_s={record['elapsed_s']}"
        )
    print(f"metrics={json.dumps(metrics, sort_keys=True)}")
    print(f"checks={sum(checks.values())}/{len(checks)} peak_vram_gib={evidence['peak_vram_gib']} load_s={evidence['load_s']}")
    result = {
        "schema_version": "1.0",
        "module": "nano-vlm-understanding/L1",
        "metrics": metrics,
        "checks": checks,
        "digest": digest,
        "evidence": evidence,
        "evidence_boundary": "real 2B checkpoint on six generated diagnostics; not a benchmark or natural-image quality claim",
        "records": records,
    }
    print("RESULT_JSON=" + json.dumps(result, sort_keys=True, separators=(",", ":")))
    if not all(checks.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
