#!/usr/bin/env python3
"""Non-blocking style diagnostics for tutorial Markdown first screens and prose."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


CONTRAST_RE = re.compile(r"(?:不是.{0,80}而是|并非.{0,80}而是|不在.{0,80}而在)")
FIELDS = {
    "question": re.compile(r"核心问题|本节目标|本节抓|>\s*目标"),
    "prerequisite": re.compile(r"先修|前置"),
    "run": re.compile(r"运行|可跑文件"),
    "acceptance": re.compile(r"验收|学完.{0,20}能|跑完.{0,20}能|本节 K\+1"),
    "boundary": re.compile(r"边界|L0 不|本模块只|本节只|不覆盖|不证明"),
}


def arguments() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1] / "tutorial" / "material"
    parser = argparse.ArgumentParser(description="Audit tutorial prose style without grading it.")
    parser.add_argument("--root", type=Path, default=root)
    parser.add_argument("--top", type=int, default=5)
    return parser.parse_args()


def visible_prose(text: str) -> list[str]:
    lines: list[str] = []
    fence = ""
    fence_len = 0
    for line in text.splitlines():
        marker = re.match(r"\s*(`{3,}|~{3,})", line)
        if marker:
            token = marker.group(1)
            if not fence:
                fence, fence_len = token[0], len(token)
            elif token[0] == fence and len(token) >= fence_len:
                fence, fence_len = "", 0
            continue
        if not fence:
            lines.append(re.sub(r"`[^`]*`", "", line))
    return lines


def main() -> int:
    args = arguments()
    root = args.root.resolve()
    files = sorted(root.rglob("tutorial_L*.md"))
    field_counts = {field: 0 for field in FIELDS}
    contrast_total = 0
    dash_total = 0
    rows = []

    for path in files:
        text = path.read_text(encoding="utf-8")
        first_screen = "\n".join(text.splitlines()[:40])
        present = {field: bool(pattern.search(first_screen)) for field, pattern in FIELDS.items()}
        for field, found in present.items():
            field_counts[field] += int(found)
        prose = "\n".join(visible_prose(text))
        contrasts = len(CONTRAST_RE.findall(prose))
        dashes = prose.count("——")
        contrast_total += contrasts
        dash_total += dashes
        rows.append(
            {
                "path": path.relative_to(root).as_posix(),
                "missing": sum(not value for value in present.values()),
                "contrasts": contrasts,
                "dashes": dashes,
            }
        )

    top = sorted(rows, key=lambda row: (-row["missing"], -row["contrasts"], -row["dashes"], row["path"]))[: args.top]
    print(f"tutorials={len(files)}")
    print("first_screen=" + " ".join(f"{key}:{value}" for key, value in field_counts.items()))
    print(f"stock_contrasts={contrast_total} double_dashes_in_prose={dash_total}")
    for row in top:
        print(
            f"REVIEW missing={row['missing']} contrasts={row['contrasts']} "
            f"dashes={row['dashes']} {row['path']}"
        )

    payload = {
        "schema_version": 1,
        "module": "tutorial_style_audit",
        "metrics": {
            "tutorials": len(files),
            "first_screen_fields": field_counts,
            "stock_contrasts": contrast_total,
            "double_dashes_in_prose": dash_total,
        },
        "review_queue": top,
        "evidence_boundary": (
            "Counts are editing signals, not a writing-quality score. Examples, logical "
            "contrasts, tables, and long explanations still require human judgment."
        ),
    }
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    payload["digest"] = hashlib.sha256(canonical.encode()).hexdigest()[:16]
    print("RESULT_JSON=" + json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
