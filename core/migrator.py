"""Rule-based CUDA→ROCm migration pre-pass.

Applies deterministic, high-confidence replacements from the knowledge base
before handing remaining issues to the LLM agents.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from core.analyzer import AnalysisReport, CudaUsage
from knowledge.cuda_rocm_map import ENV_VAR_MAP, get_all_mappings
from knowledge.optimizations import OptimizationRule, find_matching_optimizations
from knowledge.torch_cuda_map import BACKENDS_REPLACE, IMPORT_MAP


@dataclass
class AppliedChange:
    line: int
    original: str
    replacement: str
    rule: str  # description of what mapping was applied


@dataclass
class RemainingIssue:
    line: int
    symbol: str
    reason: str  # why it wasn't auto-fixed


@dataclass
class MigrationResult:
    code: str
    applied: list[AppliedChange] = field(default_factory=list)
    remaining: list[RemainingIssue] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    optimizations: list[OptimizationRule] = field(default_factory=list)


def migrate(source: str, report: AnalysisReport) -> MigrationResult:
    """Apply rule-based migrations to source code.

    Only applies changes with confidence == 1.0.
    Flags everything else as remaining issues for LLM agents.
    """
    result = MigrationResult(code=source)
    lines = source.splitlines(keepends=True)
    all_maps = get_all_mappings()

    # Track which lines were modified to avoid double-edits
    modified_lines: set[int] = set()

    # 1. Apply torch.backends.cudnn replacements (exact line match)
    for usage in report.usages:
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
                    ))
                else:
                    result.remaining.append(RemainingIssue(
                        line=usage.line, symbol=usage.symbol,
                        reason="cudnn backend reference needs manual review",
                    ))

    # 2. Apply import replacements
    for usage in report.usages:
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
                    ))
                else:
                    result.remaining.append(RemainingIssue(
                        line=usage.line, symbol=usage.symbol,
                        reason="Import may need manual migration",
                    ))

    # 3. Apply environment variable replacements
    for usage in report.usages:
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
                    ))

    # 4. Apply CUDA C API replacements (in strings/comments or direct calls)
    for usage in report.usages:
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
                            ))
                else:
                    result.remaining.append(RemainingIssue(
                        line=usage.line, symbol=usage.symbol,
                        reason=f"Low confidence ({confidence}) — needs LLM review",
                    ))
            elif usage.symbol == "<<<...>>>":
                result.remaining.append(RemainingIssue(
                    line=usage.line, symbol=usage.symbol,
                    reason="CUDA kernel launch syntax — needs LLM migration",
                ))
            else:
                result.remaining.append(RemainingIssue(
                    line=usage.line, symbol=usage.symbol,
                    reason="Unknown CUDA API — needs LLM review",
                ))

    # 5. Flag attribute usages that weren't handled
    for usage in report.usages:
        if usage.category == "attribute" and usage.line not in modified_lines:
            result.remaining.append(RemainingIssue(
                line=usage.line, symbol=usage.symbol,
                reason="CUDA attribute reference — needs LLM review",
            ))

    result.code = "".join(lines)

    # 6. Collect optimization suggestions
    result.optimizations = find_matching_optimizations(source)

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
