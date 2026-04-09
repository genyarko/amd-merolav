"""Rule-based CUDA→ROCm migration pre-pass.

Applies deterministic, high-confidence replacements from the knowledge base
before handing remaining issues to the LLM agents.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from core.analyzer import AnalysisReport, CudaUsage
from core.logging import get_logger
from knowledge.cuda_rocm_map import ENV_VAR_MAP, get_all_mappings
from knowledge.optimizations import OptimizationRule, find_matching_optimizations
from knowledge.torch_cuda_map import BACKENDS_REPLACE, IMPORT_MAP

logger = get_logger(__name__)


@dataclass
class AppliedChange:
    line: int
    original: str
    replacement: str
    rule: str  # description of what mapping was applied
    confidence: float = 1.0  # 0.0–1.0, how certain this change is correct


@dataclass
class RemainingIssue:
    line: int
    symbol: str
    reason: str  # why it wasn't auto-fixed
    confidence: float = 0.5  # 0.0–1.0, estimated difficulty / review priority


@dataclass
class MigrationResult:
    code: str
    applied: list[AppliedChange] = field(default_factory=list)
    remaining: list[RemainingIssue] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    optimizations: list[OptimizationRule] = field(default_factory=list)


def migrate(source: str, report: AnalysisReport, rocm_version: str = "") -> MigrationResult:
    """Apply rule-based migrations to source code.

    Only applies changes with confidence == 1.0.
    Flags everything else as remaining issues for LLM agents.

    If *rocm_version* is set (e.g. ``"6.0"``), mappings requiring a newer
    ROCm version are excluded.
    """
    result = MigrationResult(code=source)
    lines = source.splitlines(keepends=True)
    all_maps = get_all_mappings(rocm_version=rocm_version or None)

    # Filter out false positives before processing
    fp_count = sum(1 for u in report.usages if u.is_false_positive)
    if fp_count:
        logger.info("Skipping %d false-positive CUDA usage(s)", fp_count)

    logger.info(
        "Starting rule-based migration for %s (%d usages to process, %d false positives skipped)",
        report.file_path,
        report.total - fp_count,
        fp_count,
    )

    # Track which lines were modified to avoid double-edits
    modified_lines: set[int] = set()

    # 1. Apply torch.backends.cudnn replacements (exact line match)
    for usage in report.usages:
        if usage.is_false_positive:
            continue
        if usage.category == "backend" and usage.line not in modified_lines:
            line_idx = usage.line - 1
            if 0 <= line_idx < len(lines):
                original_line = lines[line_idx]
                replaced = _apply_backends_replace(original_line)
                if replaced != original_line:
                    lines[line_idx] = replaced
                    modified_lines.add(usage.line)
                    result.applied.append(AppliedChange(
                        line=usage.line,
                        original=original_line.rstrip(),
                        replacement=replaced.rstrip(),
                        rule="torch.backends.cudnn → miopen",
                        confidence=1.0,
                    ))
                else:
                    result.remaining.append(RemainingIssue(
                        line=usage.line, symbol=usage.symbol,
                        reason="cudnn backend reference needs manual review",
                        confidence=0.5,
                    ))

    # 2. Apply import replacements
    for usage in report.usages:
        if usage.is_false_positive:
            continue
        if usage.category == "import" and usage.line not in modified_lines:
            line_idx = usage.line - 1
            if 0 <= line_idx < len(lines):
                original_line = lines[line_idx]
                replaced = _apply_import_replace(original_line)
                if replaced != original_line:
                    lines[line_idx] = replaced
                    modified_lines.add(usage.line)
                    result.applied.append(AppliedChange(
                        line=usage.line,
                        original=original_line.rstrip(),
                        replacement=replaced.rstrip(),
                        rule="Import replacement",
                        confidence=0.95,
                    ))
                else:
                    result.remaining.append(RemainingIssue(
                        line=usage.line, symbol=usage.symbol,
                        reason="Import may need manual migration",
                        confidence=0.5,
                    ))

    # 3. Apply environment variable replacements
    for usage in report.usages:
        if usage.is_false_positive:
            continue
        if usage.category == "env_var" and usage.line not in modified_lines:
            line_idx = usage.line - 1
            if 0 <= line_idx < len(lines):
                original_line = lines[line_idx]
                replaced = _apply_env_var_replace(original_line)
                if replaced != original_line:
                    lines[line_idx] = replaced
                    modified_lines.add(usage.line)
                    result.applied.append(AppliedChange(
                        line=usage.line,
                        original=original_line.rstrip(),
                        replacement=replaced.rstrip(),
                        rule="Environment variable CUDA → HIP",
                        confidence=1.0,
                    ))

    # 4. Apply CUDA C API replacements (in strings/comments or direct calls)
    for usage in report.usages:
        if usage.is_false_positive:
            continue
        if usage.category == "api_call" and usage.line not in modified_lines:
            if usage.symbol in all_maps:
                hip_name, _notes, confidence = all_maps[usage.symbol]
                if confidence == 1.0:
                    line_idx = usage.line - 1
                    if 0 <= line_idx < len(lines):
                        original_line = lines[line_idx]
                        replaced = original_line.replace(usage.symbol, hip_name)
                        if replaced != original_line:
                            lines[line_idx] = replaced
                            modified_lines.add(usage.line)
                            result.applied.append(AppliedChange(
                                line=usage.line,
                                original=original_line.rstrip(),
                                replacement=replaced.rstrip(),
                                rule=f"{usage.symbol} → {hip_name}",
                                confidence=confidence,
                            ))
                else:
                    result.remaining.append(RemainingIssue(
                        line=usage.line, symbol=usage.symbol,
                        reason=f"Low confidence ({confidence}) — needs LLM review",
                        confidence=confidence,
                    ))
            elif usage.symbol == "<<<...>>>":
                result.remaining.append(RemainingIssue(
                    line=usage.line, symbol=usage.symbol,
                    reason="CUDA kernel launch syntax — needs LLM migration",
                    confidence=0.3,
                ))
            else:
                result.remaining.append(RemainingIssue(
                    line=usage.line, symbol=usage.symbol,
                    reason="Unknown CUDA API — needs LLM review",
                    confidence=0.4,
                ))

    # 5. Flag attribute usages that weren't handled
    for usage in report.usages:
        if usage.is_false_positive:
            continue
        if usage.category == "attribute" and usage.line not in modified_lines:
            result.remaining.append(RemainingIssue(
                line=usage.line, symbol=usage.symbol,
                reason="CUDA attribute reference — needs LLM review",
                confidence=0.6,
            ))

    result.code = "".join(lines)

    # 6. Collect optimization suggestions
    result.optimizations = find_matching_optimizations(source)

    logger.info(
        "Rule-based migration complete: %d applied, %d remaining, %d warnings, %d optimizations",
        len(result.applied),
        len(result.remaining),
        len(result.warnings),
        len(result.optimizations),
    )
    for change in result.applied:
        logger.debug("  Applied line %d: %s", change.line, change.rule)
    for issue in result.remaining:
        logger.debug("  Remaining line %d: %s — %s", issue.line, issue.symbol, issue.reason)

    return result


def _apply_backends_replace(line: str) -> str:
    """Replace torch.backends.cudnn patterns in a line."""
    for pattern, replacement in BACKENDS_REPLACE.items():
        if pattern in line:
            # Preserve leading whitespace
            indent = line[: len(line) - len(line.lstrip())]
            return indent + replacement + "\n"
    return line


def _apply_import_replace(line: str) -> str:
    """Replace CUDA-specific imports, preserving the rest of the import path."""
    stripped = line.strip()
    indent = line[: len(line) - len(line.lstrip())]
    for pattern, (replacement, note) in IMPORT_MAP.items():
        # Skip if this line was already migrated (note comment already present)
        if note in stripped:
            continue
        if stripped == pattern:
            # Exact match
            return indent + replacement + f"  # {note}\n"
        elif stripped.startswith(pattern + ".") or stripped.startswith(pattern + " "):
            # Prefix match — substitute only the module prefix, keep the rest
            suffix = stripped[len(pattern):]
            # Strip inline comment from replacement before appending suffix
            base_replacement = replacement.split("  #")[0].rstrip()
            return indent + base_replacement + suffix + f"  # {note}\n"
    return line


def _apply_env_var_replace(line: str) -> str:
    """Replace CUDA environment variable names."""
    result = line
    for cuda_var, hip_var in ENV_VAR_MAP.items():
        result = result.replace(cuda_var, hip_var)
    return result
