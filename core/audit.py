"""Migration audit log — append-only structured record of every migration run.

Log file: ``rocm_output/migration_log.json`` (one JSON object per line, JSONL format).
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from core.logging import get_logger

logger = get_logger(__name__)

_DEFAULT_LOG_PATH = "rocm_output/migration_log.jsonl"


def append_audit_entry(
    file_path: str,
    backend: str,
    applied: list,       # list[AppliedChange]
    remaining: list,     # list[RemainingIssue]
    optimizations: list, # list[OptimizationRule]
    overall_confidence: float,
    validation_passed: bool | None = None,
    agent_used: bool = False,
    warnings: list[str] | None = None,
    log_path: str | Path = _DEFAULT_LOG_PATH,
) -> dict[str, Any]:
    """Append a structured audit entry to the migration log.

    Returns the entry dict that was written.
    """
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "epoch": time.time(),
        "file": file_path,
        "backend": backend,
        "agent_used": agent_used,
        "validation_passed": validation_passed,
        "overall_confidence": round(overall_confidence, 3),
        "applied_count": len(applied),
        "remaining_count": len(remaining),
        "optimization_count": len(optimizations),
        "warnings": warnings or [],
        "applied": [
            {
                "line": c.line,
                "rule": c.rule,
                "confidence": getattr(c, "confidence", 1.0),
            }
            for c in applied
        ],
        "remaining": [
            {
                "line": r.line,
                "symbol": r.symbol,
                "reason": r.reason,
                "confidence": getattr(r, "confidence", 0.5),
            }
            for r in remaining
        ],
    }

    p = Path(log_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")

    logger.info("Audit entry appended to %s for %s", log_path, file_path)
    return entry


def read_audit_log(
    log_path: str | Path = _DEFAULT_LOG_PATH,
    file_filter: str | None = None,
) -> list[dict[str, Any]]:
    """Read the audit log, optionally filtering by file path.

    Args:
        log_path: Path to the JSONL log file.
        file_filter: If set, only return entries where 'file' contains this string.

    Returns:
        List of audit entry dicts, oldest first.
    """
    p = Path(log_path)
    if not p.exists():
        return []

    entries: list[dict[str, Any]] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
            if file_filter and file_filter not in entry.get("file", ""):
                continue
            entries.append(entry)
        except json.JSONDecodeError:
            logger.warning("Skipping malformed audit log line")
            continue

    return entries


def format_history(entries: list[dict[str, Any]]) -> str:
    """Format audit entries as a human-readable history table."""
    if not entries:
        return "No migration history found."

    lines: list[str] = []
    lines.append(f"{'Timestamp':<22} {'File':<30} {'Backend':<10} {'Applied':>7} {'Remain':>7} {'Conf':>6} {'Valid':>6}")
    lines.append("-" * 90)

    for e in entries:
        ts = e.get("timestamp", "?")[:19]  # trim microseconds
        f = e.get("file", "?")
        if len(f) > 28:
            f = "..." + f[-25:]
        b = e.get("backend", "?")
        a = e.get("applied_count", 0)
        r = e.get("remaining_count", 0)
        c = e.get("overall_confidence", 0)
        v = e.get("validation_passed")
        v_str = "pass" if v is True else ("fail" if v is False else "-")

        lines.append(f"{ts:<22} {f:<30} {b:<10} {a:>7} {r:>7} {c:>5.0%} {v_str:>6}")

    return "\n".join(lines)
