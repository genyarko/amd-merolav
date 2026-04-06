"""Unified diff generation between original and migrated code."""

from __future__ import annotations

import difflib


def generate_diff(original: str, migrated: str, filename: str = "input.py") -> str:
    """Generate a unified diff between original and migrated code."""
    original_lines = original.splitlines(keepends=True)
    migrated_lines = migrated.splitlines(keepends=True)

    diff = difflib.unified_diff(
        original_lines,
        migrated_lines,
        fromfile=f"a/{filename}",
        tofile=f"b/{filename} (ROCm)",
        lineterm="",
    )
    return "\n".join(diff)


def generate_side_by_side(original: str, migrated: str, width: int = 80) -> str:
    """Generate a side-by-side comparison of changes only."""
    orig_lines = original.splitlines()
    migr_lines = migrated.splitlines()

    matcher = difflib.SequenceMatcher(None, orig_lines, migr_lines)
    output_lines: list[str] = []
    half = width // 2 - 2

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        if tag in ("replace", "delete"):
            for line in orig_lines[i1:i2]:
                output_lines.append(f"- {line}")
        if tag in ("replace", "insert"):
            for line in migr_lines[j1:j2]:
                output_lines.append(f"+ {line}")
        output_lines.append("")

    return "\n".join(output_lines)
