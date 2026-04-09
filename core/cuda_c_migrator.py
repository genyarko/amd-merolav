"""CUDA C/C++ to HIP migrator — handles .cu/.cuh files and inline CUDA C in Python strings.

Applies high-confidence replacements (headers, types, runtime API) and flags
complex patterns (texture memory, kernel launch syntax) for manual review.
"""

from __future__ import annotations

import re
import shutil
from dataclasses import dataclass, field

from core.logging import get_logger
from knowledge.cuda_c_map import (
    CUDA_C_LAUNCH_MAP,
    CUDA_C_QUALIFIERS_MAP,
    CUDA_C_TYPES_MAP,
    get_all_cuda_c_mappings,
)
from knowledge.cuda_rocm_map import HEADER_MAP, get_all_mappings

logger = get_logger(__name__)


@dataclass
class CudaCChange:
    line: int
    original: str
    replacement: str
    rule: str
    confidence: float = 1.0


@dataclass
class CudaCIssue:
    line: int
    symbol: str
    reason: str
    confidence: float = 0.5


@dataclass
class CudaCMigrationResult:
    code: str
    applied: list[CudaCChange] = field(default_factory=list)
    remaining: list[CudaCIssue] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Header replacement
# ---------------------------------------------------------------------------

_INCLUDE_RE = re.compile(r'(#\s*include\s*)[<"]([^>"]+)[>"]')


def _replace_headers(line: str, line_num: int, result: CudaCMigrationResult) -> str:
    """Replace CUDA headers with HIP equivalents."""
    m = _INCLUDE_RE.search(line)
    if not m:
        return line
    header = m.group(2)
    if header in HEADER_MAP:
        hip_header = HEADER_MAP[header]
        new_line = line[:m.start()] + f'#include <{hip_header}>' + line[m.end():]
        result.applied.append(CudaCChange(
            line=line_num, original=line.rstrip(),
            replacement=new_line.rstrip(),
            rule=f"Header: {header} → {hip_header}",
        ))
        return new_line
    # cub/ → hipcub/ with proper renaming
    if header.startswith("cub/"):
        # Check cuda_c_map for exact match first
        from knowledge.cuda_c_map import CUB_API_MAP
        bracket_header = f"<{header}>"
        if bracket_header in CUB_API_MAP:
            hip_header = CUB_API_MAP[bracket_header][0].strip("<>")
        else:
            hip_header = header.replace("cub/", "hipcub/").replace(".cuh", ".hpp")
        new_line = line[:m.start()] + f'#include <{hip_header}>' + line[m.end():]
        result.applied.append(CudaCChange(
            line=line_num, original=line.rstrip(),
            replacement=new_line.rstrip(),
            rule=f"Header: {header} → {hip_header}",
        ))
        return new_line
    return line


# ---------------------------------------------------------------------------
# Type replacement
# ---------------------------------------------------------------------------

def _replace_types(line: str, line_num: int, result: CudaCMigrationResult) -> str:
    """Replace CUDA C types with HIP equivalents."""
    new_line = line
    for cuda_type, (hip_type, notes, confidence) in CUDA_C_TYPES_MAP.items():
        if cuda_type == hip_type:
            continue  # same in HIP (e.g. dim3)
        if cuda_type in new_line:
            if confidence >= 0.8:
                old = new_line
                new_line = new_line.replace(cuda_type, hip_type)
                if new_line != old:
                    result.applied.append(CudaCChange(
                        line=line_num, original=old.rstrip(),
                        replacement=new_line.rstrip(),
                        rule=f"Type: {cuda_type} → {hip_type}",
                        confidence=confidence,
                    ))
            else:
                result.remaining.append(CudaCIssue(
                    line=line_num, symbol=cuda_type,
                    reason=f"Low-confidence type mapping ({confidence}) — {notes}",
                    confidence=confidence,
                ))
    return new_line


# ---------------------------------------------------------------------------
# Runtime API replacement (in C context)
# ---------------------------------------------------------------------------

_CUDA_API_CALL_RE = re.compile(r"\b(cuda[A-Z]\w+)\b")


def _replace_runtime_apis(
    line: str, line_num: int, result: CudaCMigrationResult, all_maps: dict,
) -> str:
    """Replace CUDA runtime/driver API calls with HIP equivalents."""
    new_line = line
    for m in _CUDA_API_CALL_RE.finditer(line):
        symbol = m.group(1)
        if symbol in all_maps:
            hip_name, notes, confidence = all_maps[symbol]
            if confidence >= 0.9:
                new_line = new_line.replace(symbol, hip_name, 1)
                result.applied.append(CudaCChange(
                    line=line_num, original=line.rstrip(),
                    replacement=new_line.rstrip(),
                    rule=f"{symbol} → {hip_name}",
                    confidence=confidence,
                ))
            else:
                result.remaining.append(CudaCIssue(
                    line=line_num, symbol=symbol,
                    reason=f"Low-confidence API ({confidence:.0%})"
                           + (f" — {notes}" if notes else ""),
                    confidence=confidence,
                ))
        else:
            # Unknown or version-filtered CUDA API — flag for manual review
            result.remaining.append(CudaCIssue(
                line=line_num, symbol=symbol,
                reason="CUDA API not in mapping (may need newer ROCm version or manual review)",
                confidence=0.4,
            ))
    return new_line


# ---------------------------------------------------------------------------
# Kernel launch syntax
# ---------------------------------------------------------------------------

_KERNEL_LAUNCH_RE = re.compile(r"(\w+)\s*<<<\s*(.+?)\s*,\s*(.+?)\s*>>>")
_KERNEL_LAUNCH_SHARED_RE = re.compile(
    r"(\w+)\s*<<<\s*(.+?)\s*,\s*(.+?)\s*,\s*(.+?)\s*>>>"
)
_KERNEL_LAUNCH_FULL_RE = re.compile(
    r"(\w+)\s*<<<\s*(.+?)\s*,\s*(.+?)\s*,\s*(.+?)\s*,\s*(.+?)\s*>>>"
)


def _replace_kernel_launches(
    line: str, line_num: int, result: CudaCMigrationResult,
) -> str:
    """Convert <<<grid, block>>> kernel launch syntax to hipLaunchKernelGGL."""
    if "<<<" not in line:
        return line

    # Try full form: kernel<<<grid, block, shared, stream>>>(...args...)
    m = _KERNEL_LAUNCH_FULL_RE.search(line)
    if m:
        kernel, grid, block, shared, stream = m.group(1), m.group(2), m.group(3), m.group(4), m.group(5)
        # Find the argument list after >>>
        after = line[m.end():]
        args_match = re.match(r"\s*\(([^)]*)\)", after)
        args = args_match.group(1) if args_match else ""
        replacement = f"hipLaunchKernelGGL({kernel}, {grid}, {block}, {shared}, {stream}"
        if args.strip():
            replacement += f", {args.strip()}"
        replacement += ")"
        end_pos = m.end() + (args_match.end() if args_match else 0)
        new_line = line[:m.start()] + replacement + line[end_pos:]
        result.applied.append(CudaCChange(
            line=line_num, original=line.rstrip(),
            replacement=new_line.rstrip(),
            rule="Kernel launch <<<grid,block,shared,stream>>> → hipLaunchKernelGGL",
            confidence=0.7,
        ))
        return new_line

    # Try shared memory form: kernel<<<grid, block, shared>>>
    m = _KERNEL_LAUNCH_SHARED_RE.search(line)
    if m:
        kernel, grid, block, shared = m.group(1), m.group(2), m.group(3), m.group(4)
        after = line[m.end():]
        args_match = re.match(r"\s*\(([^)]*)\)", after)
        args = args_match.group(1) if args_match else ""
        replacement = f"hipLaunchKernelGGL({kernel}, {grid}, {block}, {shared}, 0"
        if args.strip():
            replacement += f", {args.strip()}"
        replacement += ")"
        end_pos = m.end() + (args_match.end() if args_match else 0)
        new_line = line[:m.start()] + replacement + line[end_pos:]
        result.applied.append(CudaCChange(
            line=line_num, original=line.rstrip(),
            replacement=new_line.rstrip(),
            rule="Kernel launch <<<grid,block,shared>>> → hipLaunchKernelGGL",
            confidence=0.6,
        ))
        return new_line

    # Basic form: kernel<<<grid, block>>>
    m = _KERNEL_LAUNCH_RE.search(line)
    if m:
        kernel, grid, block = m.group(1), m.group(2), m.group(3)
        after = line[m.end():]
        args_match = re.match(r"\s*\(([^)]*)\)", after)
        args = args_match.group(1) if args_match else ""
        replacement = f"hipLaunchKernelGGL({kernel}, {grid}, {block}, 0, 0"
        if args.strip():
            replacement += f", {args.strip()}"
        replacement += ")"
        end_pos = m.end() + (args_match.end() if args_match else 0)
        new_line = line[:m.start()] + replacement + line[end_pos:]
        result.applied.append(CudaCChange(
            line=line_num, original=line.rstrip(),
            replacement=new_line.rstrip(),
            rule="Kernel launch <<<grid,block>>> → hipLaunchKernelGGL",
            confidence=0.5,
        ))
        return new_line

    # Couldn't parse the launch syntax — flag for manual review
    result.remaining.append(CudaCIssue(
        line=line_num, symbol="<<<...>>>",
        reason="Complex kernel launch syntax — needs manual conversion to hipLaunchKernelGGL",
        confidence=0.3,
    ))
    return line


# ---------------------------------------------------------------------------
# CUB namespace replacement
# ---------------------------------------------------------------------------

_CUB_NS_RE = re.compile(r"\bcub::")


def _replace_cub_namespace(
    line: str, line_num: int, result: CudaCMigrationResult,
) -> str:
    """Replace cub:: namespace with hipcub::."""
    if "cub::" not in line:
        return line
    new_line = _CUB_NS_RE.sub("hipcub::", line)
    if new_line != line:
        result.applied.append(CudaCChange(
            line=line_num, original=line.rstrip(),
            replacement=new_line.rstrip(),
            rule="Namespace: cub:: → hipcub::",
        ))
    return new_line


# ---------------------------------------------------------------------------
# Inline CUDA C in Python strings
# ---------------------------------------------------------------------------

_TRIPLE_QUOTE_BLOCK_RE = re.compile(
    r'(""")(.*?)(""")', re.DOTALL
)
_SINGLE_TRIPLE_QUOTE_BLOCK_RE = re.compile(
    r"(''')(.*?)(''')", re.DOTALL
)


def _has_cuda_c_indicators(text: str) -> bool:
    """Check if a string block likely contains CUDA C code."""
    indicators = [
        "__global__", "__device__", "__shared__", "<<<",
        "cudaMalloc", "cudaMemcpy", "cudaFree",
        "#include", "threadIdx", "blockIdx", "blockDim",
    ]
    return any(ind in text for ind in indicators)


def migrate_inline_cuda_c(source: str, rocm_version: str = "") -> CudaCMigrationResult:
    """Migrate inline CUDA C code found in Python triple-quoted strings.

    Detects triple-quoted strings containing CUDA C indicators, applies
    header/type/API/launch/namespace replacements within them, and returns
    the full source with those strings patched.
    """
    result = CudaCMigrationResult(code=source)
    all_maps = get_all_mappings(rocm_version=rocm_version or None)

    def _migrate_block(match: re.Match) -> str:
        open_q, content, close_q = match.group(1), match.group(2), match.group(3)
        if not _has_cuda_c_indicators(content):
            return match.group(0)

        lines = content.split("\n")
        new_lines = []
        # Use a rough line offset based on how many newlines precede the match
        base_line = source[:match.start()].count("\n") + 1
        for i, line in enumerate(lines):
            line_num = base_line + i
            line = _replace_headers(line, line_num, result)
            line = _replace_types(line, line_num, result)
            line = _replace_runtime_apis(line, line_num, result, all_maps)
            line = _replace_kernel_launches(line, line_num, result)
            line = _replace_cub_namespace(line, line_num, result)
            new_lines.append(line)
        return open_q + "\n".join(new_lines) + close_q

    migrated = _TRIPLE_QUOTE_BLOCK_RE.sub(_migrate_block, source)
    migrated = _SINGLE_TRIPLE_QUOTE_BLOCK_RE.sub(_migrate_block, migrated)
    result.code = migrated
    return result


# ---------------------------------------------------------------------------
# Full .cu/.cuh file migration
# ---------------------------------------------------------------------------

def migrate_cuda_c_file(source: str, rocm_version: str = "") -> CudaCMigrationResult:
    """Migrate a .cu or .cuh file from CUDA to HIP.

    Applies:
    1. Header replacements
    2. Type replacements
    3. Runtime/driver API replacements
    4. Kernel launch syntax conversion
    5. CUB namespace conversion

    Returns a CudaCMigrationResult with the migrated code and change log.
    """
    result = CudaCMigrationResult(code=source)
    all_maps = get_all_mappings(rocm_version=rocm_version or None)

    lines = source.splitlines(keepends=True)

    logger.info("Starting CUDA C migration (%d lines)", len(lines))

    for i, line in enumerate(lines):
        line_num = i + 1
        line = _replace_headers(line, line_num, result)
        line = _replace_types(line, line_num, result)
        line = _replace_runtime_apis(line, line_num, result, all_maps)
        line = _replace_kernel_launches(line, line_num, result)
        line = _replace_cub_namespace(line, line_num, result)
        lines[i] = line

    result.code = "".join(lines)

    # Add HIPIFY suggestion if there are remaining issues
    if result.remaining:
        hipify_available = shutil.which("hipify-perl") is not None
        if hipify_available:
            result.warnings.append(
                "hipify-perl is available on this system. Consider running it "
                "for a more thorough automated conversion of remaining patterns."
            )
        else:
            result.warnings.append(
                "For complex CUDA C patterns, consider using AMD's hipify-perl "
                "or hipify-clang tools: "
                "https://rocm.docs.amd.com/projects/HIPIFY/en/latest/"
            )

    logger.info(
        "CUDA C migration complete: %d applied, %d remaining",
        len(result.applied), len(result.remaining),
    )
    return result


# ---------------------------------------------------------------------------
# HIPIFY integration (optional)
# ---------------------------------------------------------------------------

def check_hipify_available() -> str | None:
    """Return the path to hipify-perl if available, else None."""
    return shutil.which("hipify-perl")


def run_hipify(file_path: str) -> tuple[str, bool]:
    """Run hipify-perl on a file and return (output, success).

    Returns the hipified source on success, or an error message on failure.
    """
    import subprocess

    hipify = check_hipify_available()
    if not hipify:
        return "hipify-perl not found on PATH", False

    try:
        proc = subprocess.run(
            [hipify, file_path],
            capture_output=True, text=True, timeout=60,
        )
        if proc.returncode == 0:
            return proc.stdout, True
        return f"hipify-perl failed: {proc.stderr}", False
    except subprocess.TimeoutExpired:
        return "hipify-perl timed out after 60s", False
    except Exception as exc:
        return f"hipify-perl error: {exc}", False
