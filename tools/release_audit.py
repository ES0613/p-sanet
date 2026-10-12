#!/usr/bin/env python3
"""Check that the standalone inference tree contains only public assets."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

TEXT_SUFFIXES = {"", ".cff", ".csv", ".in", ".ipynb", ".json", ".md", ".py", ".toml", ".txt", ".yml", ".yaml"}
SKIP_PARTS = {".git", ".pytest_cache", "__pycache__", "build", "dist"}
FORBIDDEN_SUFFIXES = {".joblib", ".pkl", ".pickle", ".pt", ".xlsx", ".xls"}
HIDDEN_LABELS = ("rota" + "tion", "f" + "old")
INTERNAL_PATTERN = re.compile(r"(?<![A-Za-z])(?:" + "|".join(HIDDEN_LABELS) + r")(?![A-Za-z])", re.I)
CJK_PATTERN = re.compile(r"[\u3000-\u303f\u3400-\u9fff\uf900-\ufaff\uff00-\uffef]")


def audit(root: Path) -> list[str]:
    problems: list[str] = []
    for path in root.rglob("*"):
        relative = path.relative_to(root)
        if any(part in SKIP_PARTS or part.endswith(".egg-info") for part in relative.parts):
            continue
        label = relative.as_posix().lower()
        if path.name == ".DS_Store" or (path.is_file() and path.suffix.lower() in FORBIDDEN_SUFFIXES):
            problems.append(f"forbidden file: {relative}")
            continue
        if INTERNAL_PATTERN.search(label):
            problems.append(f"internal identifier in path: {relative}")
        if CJK_PATTERN.search(relative.as_posix()):
            problems.append(f"non-English characters in path: {relative}")
        if not path.is_file():
            continue
        if relative.as_posix() == "tools/release_audit.py":
            continue
        raw = path.read_bytes()
        if b"/Users/" in raw or b"\\\\Users\\\\" in raw:
            problems.append(f"local absolute path: {relative}")
        if path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        try:
            content = raw.decode("utf-8")
        except UnicodeDecodeError:
            continue
        if INTERNAL_PATTERN.search(content):
            problems.append(f"internal identifier in content: {relative}")
        if CJK_PATTERN.search(content):
            problems.append(f"non-English characters in content: {relative}")
        if re.search(r"source_checkpoint_sha256|best_validation_metrics|private_reference", content, re.I):
            problems.append(f"internal model metadata: {relative}")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", nargs="?", type=Path, default=Path.cwd())
    args = parser.parse_args()
    problems = audit(args.root.resolve())
    if problems:
        print("Release audit failed:", file=sys.stderr)
        for problem in problems:
            print(f"- {problem}", file=sys.stderr)
        return 1
    print("Release audit passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
