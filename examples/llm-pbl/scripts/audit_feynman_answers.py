#!/usr/bin/env python3
"""Audit tutorial-level Feynman checks and their explicit reference answers."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
INLINE_RE = re.compile(r"^\s*(?:\*\*)?费曼自检[:：]")
ANSWER_RE = re.compile(r"参考答案|参考讲法|答案要点")


def arguments() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1] / "tutorial" / "material"
    parser = argparse.ArgumentParser(description="Audit explicit answers in Feynman sections.")
    parser.add_argument("--root", type=Path, default=root)
    return parser.parse_args()


def feynman_sections(text: str) -> list[tuple[str, str]]:
    lines = text.splitlines()
    found: list[tuple[str, str]] = []
    for index, line in enumerate(lines):
        match = HEADING_RE.match(line)
        if not match or "费曼" not in match.group(2) or "费曼审查" in match.group(2):
            continue
        level = len(match.group(1))
        end = len(lines)
        for cursor in range(index + 1, len(lines)):
            next_heading = HEADING_RE.match(lines[cursor])
            if next_heading and len(next_heading.group(1)) <= level:
                end = cursor
                break
        found.append((match.group(2), "\n".join(lines[index + 1 : end])))
    for index, line in enumerate(lines):
        if not INLINE_RE.match(line):
            continue
        end = len(lines)
        for cursor in range(index + 1, len(lines)):
            if HEADING_RE.match(lines[cursor]):
                end = cursor
                break
        found.append(("inline 费曼自检", "\n".join(lines[index:end])))
    return found


def main() -> int:
    args = arguments()
    root = args.root.resolve()
    files = sorted(root.rglob("tutorial_L*.md"))
    sections = 0
    answered = 0
    missing_answers: list[dict[str, str]] = []
    tutorials_missing_feynman: list[str] = []

    for path in files:
        found = feynman_sections(path.read_text(encoding="utf-8"))
        if not found:
            tutorials_missing_feynman.append(path.relative_to(root).as_posix())
        for title, body in found:
            sections += 1
            if ANSWER_RE.search(body):
                answered += 1
            else:
                missing_answers.append(
                    {"path": path.relative_to(root).as_posix(), "section": title}
                )

    tutorials_with_feynman = len(files) - len(tutorials_missing_feynman)
    print(
        f"tutorials={len(files)} tutorials_with_feynman={tutorials_with_feynman} "
        f"tutorials_missing_feynman={len(tutorials_missing_feynman)}"
    )
    print(
        f"feynman_sections={sections} explicit_answers={answered} "
        f"missing_answers={len(missing_answers)}"
    )
    for path in tutorials_missing_feynman:
        print(f"MISSING_SECTION {path}")
    for item in missing_answers:
        print(f"MISSING_ANSWER {item['path']} :: {item['section']}")

    payload = {
        "schema_version": 1,
        "module": "feynman_answer_audit",
        "metrics": {
            "tutorials": len(files),
            "tutorials_with_feynman": tutorials_with_feynman,
            "tutorials_missing_feynman": len(tutorials_missing_feynman),
            "sections": sections,
            "explicit_answers": answered,
            "missing_answers": len(missing_answers),
        },
        "tutorials_missing_feynman": tutorials_missing_feynman,
        "missing_answers": missing_answers,
        "evidence_boundary": (
            "This checks tutorial-level Feynman-section and explicit-answer coverage, "
            "not correctness, explanatory depth, or whether a learner attempted the "
            "questions before reading the answers."
        ),
    }
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    payload["digest"] = hashlib.sha256(canonical.encode()).hexdigest()[:16]
    print("RESULT_JSON=" + json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    return 0 if not tutorials_missing_feynman and not missing_answers else 1


if __name__ == "__main__":
    raise SystemExit(main())
