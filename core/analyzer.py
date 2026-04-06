"""Static analysis: scan Python files for CUDA-specific usage."""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field


@dataclass
class CudaUsage:
    line: int
    col: int
    symbol: str
    category: str  # "import", "attribute", "string", "env_var", "api_call", "backend"
    context: str  # the source line for display


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
    return usages


# --- Public API ---

def analyze_source(source: str, file_path: str = "<input>") -> AnalysisReport:
    """Analyze Python source code for CUDA-specific usage.

    Returns an AnalysisReport listing every CUDA reference found.
    """
    report = AnalysisReport(file_path=file_path)
    source_lines = source.splitlines()

    # AST pass
    try:
        tree = ast.parse(source)
        visitor = _CudaVisitor(source_lines)
        visitor.visit(tree)
        for usage in visitor.usages:
            report.add(usage)
    except SyntaxError:
        pass  # Fall through to regex-only

    # Regex pass for things AST can't catch
    for usage in _regex_scan(source, source_lines):
        # Avoid duplicates (same line + symbol)
        existing = {(u.line, u.symbol) for u in report.usages}
        if (usage.line, usage.symbol) not in existing:
            report.add(usage)

    return report
