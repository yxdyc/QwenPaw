#!/usr/bin/env python3
"""Deterministic, standard-library release checks for tutorial/material."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from urllib.parse import unquote


LINK_RE = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")
TODO_RE = re.compile(r"\b(?:TODO|FIXME|TBD)\b", re.IGNORECASE)
SENSITIVE_PATTERNS = {
    "absolute_local_path": re.compile(r"/(?:Users|nas|root)/"),
    "internal_role": re.compile(r"\b(?:Writer|Reviewer)(?:\s+[A-C])?\b"),
    "internal_round": re.compile(r"\bRound\s+\d+\b", re.IGNORECASE),
    "internal_lane": re.compile(r"(?<![A-Za-z])[A-C]\s*线"),
    "internal_report": re.compile(r"REVIEW-REPORT(?:-[A-C])?\.md"),
    "workspace_metadata": re.compile(r"(?:\.qoderwork|workspace/[0-9a-f-]{8,})"),
}
KNOWN_LINK_SUFFIXES = {
    ".csv", ".gif", ".html", ".ipynb", ".jpeg", ".jpg", ".json",
    ".md", ".mp4", ".pdf", ".png", ".py", ".sh", ".svg", ".toml",
    ".tsv", ".webp", ".yaml", ".yml",
}


def arguments() -> argparse.Namespace:
    default_root = Path(__file__).resolve().parents[1] / "tutorial" / "material"
    parser = argparse.ArgumentParser(
        description="Validate the public tutorial/material tree without running lessons."
    )
    parser.add_argument("--root", type=Path, default=default_root)
    return parser.parse_args()


def relative(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.name


def markdown_visible_lines(path: Path, text: str, issues: list[str]) -> list[tuple[int, str]]:
    visible: list[tuple[int, str]] = []
    fence_char = ""
    fence_len = 0
    fence_line = 0
    for line_no, line in enumerate(text.splitlines(), 1):
        stripped = line.lstrip()
        marker = re.match(r"(`{3,}|~{3,})", stripped)
        if marker:
            token = marker.group(1)
            if not fence_char:
                fence_char, fence_len, fence_line = token[0], len(token), line_no
            elif token[0] == fence_char and len(token) >= fence_len:
                fence_char, fence_len, fence_line = "", 0, 0
            continue
        if not fence_char:
            visible.append((line_no, re.sub(r"`[^`]*`", "", line)))
    if fence_char:
        issues.append(f"fence:{path.as_posix()}:{fence_line}:unclosed")
    return visible


def link_target(raw: str) -> str:
    raw = raw.strip()
    if raw.startswith("<") and ">" in raw:
        raw = raw[1 : raw.index(">")]
    else:
        raw = raw.split(maxsplit=1)[0]
    return unquote(raw.replace("\\ ", " "))


def is_local_file_link(target: str) -> bool:
    if not target or target.startswith(("#", "/", "//")):
        return False
    if re.match(r"^[A-Za-z][A-Za-z0-9+.-]*:", target):
        return False
    path_part = target.split("#", 1)[0].split("?", 1)[0]
    return (
        path_part.startswith(("./", "../"))
        or "/" in path_part
        or Path(path_part).suffix.lower() in KNOWN_LINK_SUFFIXES
    )


def git_ignored(paths: set[Path], git_root: Path | None) -> set[Path]:
    if git_root is None or not paths:
        return set()
    payload = "\0".join(str(path.resolve()) for path in sorted(paths)) + "\0"
    try:
        process = subprocess.run(
            ["git", "-C", str(git_root), "check-ignore", "-z", "--stdin"],
            input=payload,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return set()
    return {
        Path(item).resolve() for item in process.stdout.split("\0") if item
    }


def validate_markdown(
    path: Path,
    root: Path,
    text: str,
    issues: list[str],
    link_records: list[tuple[Path, Path, str, int, str]],
) -> None:
    rel = relative(path, root)
    local_issues: list[str] = []
    for line_no, visible in markdown_visible_lines(Path(rel), text, local_issues):
        for match in LINK_RE.finditer(visible):
            target = link_target(match.group(1))
            if not is_local_file_link(target):
                continue
            path_part = target.split("#", 1)[0].split("?", 1)[0]
            destination = (path.parent / path_part).resolve()
            if not destination.exists():
                issues.append(f"link:{rel}:{line_no}:{path_part}")
            else:
                link_records.append((path.resolve(), destination, rel, line_no, path_part))
    issues.extend(local_issues)


def validate_python(path: Path, root: Path, text: str, issues: list[str]) -> None:
    try:
        ast.parse(text, filename=relative(path, root))
    except SyntaxError as exc:
        issues.append(
            f"ast:{relative(path, root)}:{exc.lineno or 0}:{exc.msg}"
        )


def main() -> int:
    root = arguments().root.resolve()
    if not root.is_dir():
        print("FAIL root: tutorial/material directory not found")
        return 2

    git_root = next(
        (candidate for candidate in (root, *root.parents) if (candidate / ".git").exists()),
        None,
    )
    link_records: list[tuple[Path, Path, str, int, str]] = []
    files = sorted(path for path in root.rglob("*") if path.is_file())
    markdown = [path for path in files if path.suffix.lower() == ".md"]
    python = [path for path in files if path.suffix.lower() == ".py"]
    issues: list[str] = []
    todo_count = 0

    for path in files:
        rel = relative(path, root)
        if path.name in {".DS_Store"} or path.suffix == ".pyc":
            issues.append(f"artifact:{rel}")
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        todo_count += len(TODO_RE.findall(text))
        for name, pattern in SENSITIVE_PATTERNS.items():
            for match in pattern.finditer(text):
                line_no = text.count("\n", 0, match.start()) + 1
                issues.append(f"sensitive:{rel}:{line_no}:{name}")
        if path.suffix.lower() == ".md":
            validate_markdown(path, root, text, issues, link_records)
        elif path.suffix.lower() == ".py":
            validate_python(path, root, text, issues)

    ignored = git_ignored(
        {path for record in link_records for path in record[:2]}, git_root
    )
    for source, destination, rel, line_no, path_part in link_records:
        if source not in ignored and destination in ignored:
            issues.append(f"git_visibility:{rel}:{line_no}:{path_part}")

    for directory in sorted(path for path in root.rglob("*") if path.is_dir()):
        if directory.name == "__pycache__":
            issues.append(f"artifact:{relative(directory, root)}/")

    issues = sorted(set(issues))
    for issue in issues:
        print(f"FAIL {issue}")
    if not issues:
        print(
            f"PASS material release checks: {len(markdown)} Markdown, "
            f"{len(python)} Python, {len(files)} total files"
        )
    print(f"INFO planning markers (non-blocking): {todo_count}")

    evidence = {
        "checks": {
            "artifacts": not any(item.startswith("artifact:") for item in issues),
            "python_ast": not any(item.startswith("ast:") for item in issues),
            "markdown_fences": not any(item.startswith("fence:") for item in issues),
            "git_visibility": not any(item.startswith("git_visibility:") for item in issues),
            "relative_links": not any(item.startswith("link:") for item in issues),
            "sensitive_content": not any(item.startswith("sensitive:") for item in issues),
        },
        "counts": {
            "issues": len(issues),
            "markdown_files": len(markdown),
            "planning_markers": todo_count,
            "python_files": len(python),
            "total_files": len(files),
        },
        "evidence_boundary": (
            "Static structure and release hygiene only; lesson execution and external "
            "claim verification are out of scope."
        ),
        "module": "material_release_validator",
        "schema_version": 1,
    }
    canonical = json.dumps(evidence, ensure_ascii=False, sort_keys=True)
    evidence["digest"] = hashlib.sha256(canonical.encode()).hexdigest()[:16]
    print("RESULT_JSON=" + json.dumps(evidence, ensure_ascii=False, sort_keys=True))
    return 1 if issues else 0


if __name__ == "__main__":
    sys.exit(main())
