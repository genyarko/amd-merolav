"""Static analysis: scan Python files for CUDA-specific usage."""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field

from core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class CudaUsage:
    line: int
    col: int
    symbol: str
    category: str  # "import", "attribute", "string", "env_var", "api_call", "backend"
    context: str  # the source line for display
    is_false_positive: bool = False  # True if this is likely not a real CUDA API usage


@dataclass
class AnalysisReport:
    file_path: str
    usages: list[CudaUsage] = field(default_factory=list)
    summary: dict[str, int] = field(default_factory=dict)

    def add(self, usage: CudaUsage) -> None:
        self.usages.append(usage)
        self.summary[usage.category] = self.summary.get(usage.category, 0) + 1

    @property
    def total(self) -> int:
        return len(self.usages)

    @property
    def has_cuda(self) -> bool:
        return self.total > 0


# --- AST Visitor ---

class _CudaVisitor(ast.NodeVisitor):
    """Walk the AST looking for CUDA-specific patterns."""

    def __init__(self, source_lines: list[str]) -> None:
        self.source_lines = source_lines
        self.usages: list[CudaUsage] = []

    def _ctx(self, lineno: int) -> str:
        if 1 <= lineno <= len(self.source_lines):
            return self.source_lines[lineno - 1].rstrip()
        return ""

    # --- Imports ---

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            name = alias.name.lower()
            if any(kw in name for kw in ("pycuda", "cupy", "tensorrt", "cuda")):
                self.usages.append(CudaUsage(
                    line=node.lineno, col=node.col_offset,
                    symbol=alias.name, category="import",
                    context=self._ctx(node.lineno),
                ))
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        mod = (node.module or "").lower()
        if any(kw in mod for kw in ("pycuda", "cupy", "tensorrt", "apex")):
            self.usages.append(CudaUsage(
                line=node.lineno, col=node.col_offset,
                symbol=node.module or "", category="import",
                context=self._ctx(node.lineno),
            ))
        self.generic_visit(node)

    # --- Attribute access (torch.cuda.*, torch.backends.cudnn.*) ---

    def visit_Attribute(self, node: ast.Attribute) -> None:
        full = _reconstruct_attr(node)
        if full:
            # torch.backends.cudnn.*
            if "cudnn" in full:
                self.usages.append(CudaUsage(
                    line=node.lineno, col=node.col_offset,
                    symbol=full, category="backend",
                    context=self._ctx(node.lineno),
                ))
            # torch.cuda.nvtx, torch.cuda.nccl
            elif "torch.cuda.nvtx" in full or "torch.cuda.nccl" in full:
                self.usages.append(CudaUsage(
                    line=node.lineno, col=node.col_offset,
                    symbol=full, category="attribute",
                    context=self._ctx(node.lineno),
                ))
        self.generic_visit(node)

    # --- String literals ("cuda", device strings, env vars) ---

    def visit_Constant(self, node: ast.Constant) -> None:
        if isinstance(node.value, str):
            val = node.value
            # Environment variables
            if val in ("CUDA_VISIBLE_DEVICES", "CUDA_LAUNCH_BLOCKING", "CUDA_DEVICE_ORDER"):
                self.usages.append(CudaUsage(
                    line=node.lineno, col=node.col_offset,
                    symbol=val, category="env_var",
                    context=self._ctx(node.lineno),
                ))
        self.generic_visit(node)

    # --- Function calls matching CUDA API names ---

    def visit_Call(self, node: ast.Call) -> None:
        func_name = _get_call_name(node)
        if func_name and func_name.startswith("cuda"):
            self.usages.append(CudaUsage(
                line=node.lineno, col=node.col_offset,
                symbol=func_name, category="api_call",
                context=self._ctx(node.lineno),
            ))
        self.generic_visit(node)


def _reconstruct_attr(node: ast.Attribute) -> str | None:
    """Reconstruct a dotted attribute chain like 'torch.backends.cudnn.benchmark'."""
    parts = []
    current: ast.expr = node
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if isinstance(current, ast.Name):
        parts.append(current.id)
    else:
        return None
    return ".".join(reversed(parts))


def _get_call_name(node: ast.Call) -> str | None:
    """Get the name of a direct function call like cudaMalloc(...)."""
    if isinstance(node.func, ast.Name):
        return node.func.id
    return None


# --- Regex pass for patterns AST can't catch ---

_ENV_VAR_RE = re.compile(
    r"""(CUDA_VISIBLE_DEVICES|CUDA_LAUNCH_BLOCKING|CUDA_DEVICE_ORDER)"""
)
_KERNEL_LAUNCH_RE = re.compile(r"<<<.*?>>>")
_CUDA_C_API_RE = re.compile(r"\b(cuda[A-Z]\w+)\b")

# Patterns for C/C++ CUDA libraries found in inline code strings
_CUB_NS_RE = re.compile(r"\bcub::(\w+)")
_THRUST_NS_RE = re.compile(r"\bthrust::(\w+)")
_COOP_GROUPS_RE = re.compile(r"\bcooperative_groups::(\w+)")
_WARP_INTRINSIC_RE = re.compile(
    r"\b(__shfl(?:_up|_down|_xor)?_sync|__ballot_sync|__all_sync|__any_sync"
    r"|__activemask|__syncwarp|warpSize)\b"
)
_CUDA_GRAPH_RE = re.compile(r"\b(cudaGraph\w+|cudaStreamBeginCapture|cudaStreamEndCapture|cudaStreamIsCapturing)\b")
_TENSORRT_RE = re.compile(r"\b(trt\.Builder|trt\.Runtime|trt\.Logger|ICudaEngine|IExecutionContext)\b")

# CUDA C qualifiers and types
_CUDA_C_QUALIFIER_RE = re.compile(
    r"\b(__global__|__device__|__host__|__shared__|__constant__|__managed__"
    r"|__launch_bounds__|__forceinline__)\b"
)
_CUDA_C_TYPE_RE = re.compile(
    r"\b(cudaStream_t|cudaEvent_t|cudaError_t|cudaDeviceProp|cudaMemcpyKind"
    r"|cudaFuncAttributes|cudaPointerAttributes|cudaGraph_t|cudaGraphExec_t"
    r"|cudaGraphNode_t|cudaArray_t|cudaTextureObject_t|cudaSurfaceObject_t"
    r"|cudaPitchedPtr|cudaExtent|cudaPos|cudaChannelFormatDesc"
    r"|cudaGraphicsResource_t)\b"
)
# Inline CUDA C in Python strings (pycuda SourceModule, etc.)
_SOURCE_MODULE_RE = re.compile(r"SourceModule\s*\(")
# CUDA header includes
_CUDA_HEADER_RE = re.compile(r'#\s*include\s*[<"]([^>"]*(?:cuda|cub/|thrust/)[^>"]*)[>"]')


def _regex_scan(source: str, source_lines: list[str]) -> list[CudaUsage]:
    """Catch CUDA patterns that AST misses (inline strings, comments, etc.)."""
    usages: list[CudaUsage] = []
    for i, line in enumerate(source_lines, 1):
        # Kernel launch syntax in strings
        if _KERNEL_LAUNCH_RE.search(line):
            usages.append(CudaUsage(
                line=i, col=0,
                symbol="<<<...>>>", category="api_call",
                context=line.rstrip(),
            ))
        # CUB namespace references
        for m in _CUB_NS_RE.finditer(line):
            usages.append(CudaUsage(
                line=i, col=m.start(),
                symbol=f"cub::{m.group(1)}", category="api_call",
                context=line.rstrip(),
            ))
        # Thrust namespace references
        for m in _THRUST_NS_RE.finditer(line):
            usages.append(CudaUsage(
                line=i, col=m.start(),
                symbol=f"thrust::{m.group(1)}", category="api_call",
                context=line.rstrip(),
            ))
        # Cooperative groups
        for m in _COOP_GROUPS_RE.finditer(line):
            usages.append(CudaUsage(
                line=i, col=m.start(),
                symbol=f"cooperative_groups::{m.group(1)}", category="api_call",
                context=line.rstrip(),
            ))
        # Warp intrinsics
        for m in _WARP_INTRINSIC_RE.finditer(line):
            usages.append(CudaUsage(
                line=i, col=m.start(),
                symbol=m.group(1), category="api_call",
                context=line.rstrip(),
            ))
        # CUDA Graphs API
        for m in _CUDA_GRAPH_RE.finditer(line):
            usages.append(CudaUsage(
                line=i, col=m.start(),
                symbol=m.group(1), category="api_call",
                context=line.rstrip(),
            ))
        # TensorRT references (in code, not imports — imports handled by AST)
        for m in _TENSORRT_RE.finditer(line):
            usages.append(CudaUsage(
                line=i, col=m.start(),
                symbol=m.group(1), category="api_call",
                context=line.rstrip(),
            ))
        # CUDA C qualifiers (__global__, __device__, __shared__, etc.)
        for m in _CUDA_C_QUALIFIER_RE.finditer(line):
            usages.append(CudaUsage(
                line=i, col=m.start(),
                symbol=m.group(1), category="cuda_c_qualifier",
                context=line.rstrip(),
            ))
        # CUDA C types (cudaStream_t, cudaEvent_t, dim3, etc.)
        for m in _CUDA_C_TYPE_RE.finditer(line):
            usages.append(CudaUsage(
                line=i, col=m.start(),
                symbol=m.group(1), category="cuda_c_type",
                context=line.rstrip(),
            ))
        # CUDA header includes
        for m in _CUDA_HEADER_RE.finditer(line):
            usages.append(CudaUsage(
                line=i, col=m.start(),
                symbol=m.group(1), category="cuda_c_header",
                context=line.rstrip(),
            ))
        # pycuda SourceModule (marks inline CUDA C code)
        if _SOURCE_MODULE_RE.search(line):
            usages.append(CudaUsage(
                line=i, col=0,
                symbol="SourceModule", category="inline_cuda_c",
                context=line.rstrip(),
            ))
    return usages


# --- False-positive detection ---

_COMMENT_RE = re.compile(r"^\s*#")


def _flag_false_positives(report: AnalysisReport, source_lines: list[str]) -> None:
    """Flag usages that are likely false positives.

    Heuristics:
    - CUDA references on comment-only lines
    - env_var / api_call usages inside docstrings (triple-quoted strings)
    - Variable names that happen to start with 'cuda' but aren't API calls
    """
    # Build a set of lines that are inside try/except ImportError blocks
    # (these are guarded imports — still real usages, not false positives)

    for usage in report.usages:
        line_idx = usage.line - 1
        if line_idx < 0 or line_idx >= len(source_lines):
            continue
        line = source_lines[line_idx]

        # 1. Comment-only lines (regex pass may pick these up)
        if _COMMENT_RE.match(line):
            usage.is_false_positive = True
            logger.debug(
                "Flagged false positive at line %d: '%s' is in a comment",
                usage.line,
                usage.symbol,
            )
            continue

        # 2. Env var or api_call in a line that's entirely a string assignment
        #    (e.g., docstrings mentioning CUDA_VISIBLE_DEVICES)
        stripped = line.strip()
        if usage.category in ("env_var", "api_call"):
            if stripped.startswith(('"""', "'''", 'r"""', "r'''")):
                usage.is_false_positive = True
                logger.debug(
                    "Flagged false positive at line %d: '%s' is inside a docstring",
                    usage.line,
                    usage.symbol,
                )
                continue

    fp_count = sum(1 for u in report.usages if u.is_false_positive)
    if fp_count:
        logger.info("Flagged %d likely false positive(s)", fp_count)


# --- Public API ---

def analyze_source(source: str, file_path: str = "<input>") -> AnalysisReport:
    """Analyze Python source code for CUDA-specific usage.

    Returns an AnalysisReport listing every CUDA reference found.
    """
    report = AnalysisReport(file_path=file_path)
    source_lines = source.splitlines()

    logger.info("Analyzing %s (%d lines)", file_path, len(source_lines))

    # AST pass
    try:
        tree = ast.parse(source)
        visitor = _CudaVisitor(source_lines)
        visitor.visit(tree)
        for usage in visitor.usages:
            report.add(usage)
        logger.debug("AST pass found %d CUDA usages", len(visitor.usages))
    except SyntaxError as exc:
        logger.warning(
            "AST parsing failed for %s (line %s: %s) — falling back to regex-only analysis",
            file_path,
            exc.lineno,
            exc.msg,
        )

    # Regex pass for things AST can't catch
    for usage in _regex_scan(source, source_lines):
        # Avoid duplicates (same line + symbol)
        existing = {(u.line, u.symbol) for u in report.usages}
        if (usage.line, usage.symbol) not in existing:
            report.add(usage)

    # False-positive detection pass
    _flag_false_positives(report, source_lines)

    logger.info(
        "Analysis complete for %s: %d CUDA usages found (%s)",
        file_path,
        report.total,
        ", ".join(f"{k}={v}" for k, v in report.summary.items()) or "none",
    )

    return report
