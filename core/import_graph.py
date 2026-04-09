"""Import dependency graph for multi-file project migration.

Parses all Python files in a project to build a dependency graph,
identifies which modules define CUDA symbols, and determines
migration order (leaf dependencies first).
"""

from __future__ import annotations

import ast
from collections import defaultdict
from pathlib import Path
from typing import Iterator

from core.logging import get_logger

logger = get_logger(__name__)


class ImportGraph:
    """Directed graph of import dependencies between project modules."""

    def __init__(self) -> None:
        # module_name → set of module_names it imports
        self._edges: dict[str, set[str]] = defaultdict(set)
        # module_name → file path
        self._paths: dict[str, Path] = {}
        # module_name → set of CUDA symbols it defines/uses
        self._cuda_symbols: dict[str, set[str]] = defaultdict(set)

    @property
    def modules(self) -> list[str]:
        return list(self._paths.keys())

    def path_for(self, module: str) -> Path | None:
        return self._paths.get(module)

    def dependencies(self, module: str) -> set[str]:
        return self._edges.get(module, set())

    def dependents(self, module: str) -> set[str]:
        """Modules that import the given module."""
        return {m for m, deps in self._edges.items() if module in deps}

    def has_cuda(self, module: str) -> bool:
        return bool(self._cuda_symbols.get(module))

    def cuda_modules(self) -> list[str]:
        return [m for m in self._paths if self.has_cuda(m)]

    def add_module(self, module_name: str, file_path: Path) -> None:
        self._paths[module_name] = file_path
        if module_name not in self._edges:
            self._edges[module_name] = set()

    def add_edge(self, from_module: str, to_module: str) -> None:
        self._edges[from_module].add(to_module)

    def add_cuda_symbol(self, module: str, symbol: str) -> None:
        self._cuda_symbols[module].add(symbol)

    def migration_order(self) -> list[str]:
        """Return modules in topological order (leaf dependencies first).

        Modules with no CUDA symbols are included but placed after CUDA
        modules so cross-file context is available.
        """
        visited: set[str] = set()
        order: list[str] = []

        def _visit(mod: str) -> None:
            if mod in visited:
                return
            visited.add(mod)
            for dep in self._edges.get(mod, set()):
                if dep in self._paths:  # only visit project-internal deps
                    _visit(dep)
            order.append(mod)

        for mod in self._paths:
            _visit(mod)

        return order

    def __repr__(self) -> str:
        return f"ImportGraph({len(self._paths)} modules, {sum(len(v) for v in self._edges.values())} edges)"


# --- CUDA symbol patterns for quick detection ---

_CUDA_IMPORT_PREFIXES = (
    "torch.cuda", "cupy", "pycuda", "numba.cuda",
    "cudnn", "nccl", "cuda",
)

_CUDA_ATTR_PATTERNS = (
    "cuda", "cudnn", "nccl", "gpu", "GPU",
)


def build_import_graph(project_root: Path) -> ImportGraph:
    """Scan all Python files in a project and build the import graph.

    Args:
        project_root: Root directory to scan.

    Returns:
        An ImportGraph with modules, edges, and CUDA symbol annotations.
    """
    graph = ImportGraph()

    # Collect all .py files
    py_files = sorted(project_root.rglob("*.py"))
    if not py_files:
        logger.warning("No Python files found in %s", project_root)
        return graph

    # First pass: register all modules
    for py_file in py_files:
        module_name = _file_to_module(py_file, project_root)
        if module_name:
            graph.add_module(module_name, py_file)

    # Second pass: parse imports and detect CUDA symbols
    for py_file in py_files:
        module_name = _file_to_module(py_file, project_root)
        if not module_name:
            continue

        try:
            source = py_file.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=str(py_file))
        except (SyntaxError, UnicodeDecodeError) as exc:
            logger.warning("Could not parse %s: %s", py_file, exc)
            continue

        _process_module(tree, module_name, source, graph)

    logger.info(
        "Built import graph: %d modules, %d with CUDA symbols",
        len(graph.modules),
        len(graph.cuda_modules()),
    )

    return graph


def _file_to_module(file_path: Path, root: Path) -> str | None:
    """Convert a file path to a Python module name relative to root."""
    try:
        rel = file_path.relative_to(root)
    except ValueError:
        return None

    parts = list(rel.parts)
    if not parts:
        return None

    # Strip .py extension from the last part
    if parts[-1].endswith(".py"):
        parts[-1] = parts[-1][:-3]

    # __init__ becomes the package name
    if parts[-1] == "__init__":
        parts = parts[:-1]

    if not parts:
        return None

    return ".".join(parts)


def _process_module(
    tree: ast.AST, module_name: str, source: str, graph: ImportGraph
) -> None:
    """Extract imports and CUDA symbols from a parsed module."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported = alias.name
                graph.add_edge(module_name, imported)
                if any(imported.startswith(p) for p in _CUDA_IMPORT_PREFIXES):
                    graph.add_cuda_symbol(module_name, imported)

        elif isinstance(node, ast.ImportFrom):
            if node.module:
                graph.add_edge(module_name, node.module)
                if any(node.module.startswith(p) for p in _CUDA_IMPORT_PREFIXES):
                    for alias in (node.names or []):
                        graph.add_cuda_symbol(module_name, f"{node.module}.{alias.name}")

        elif isinstance(node, ast.Attribute):
            attr_str = _get_attribute_string(node)
            if attr_str and any(p in attr_str for p in _CUDA_ATTR_PATTERNS):
                graph.add_cuda_symbol(module_name, attr_str)

        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            if "cuda" in node.value.lower():
                graph.add_cuda_symbol(module_name, f"string:{node.value[:40]}")


def _get_attribute_string(node: ast.Attribute) -> str | None:
    """Reconstruct a dotted attribute chain (e.g. 'torch.cuda.is_available')."""
    parts: list[str] = [node.attr]
    current = node.value
    depth = 0
    while isinstance(current, ast.Attribute) and depth < 10:
        parts.append(current.attr)
        current = current.value
        depth += 1
    if isinstance(current, ast.Name):
        parts.append(current.id)
    else:
        return None
    return ".".join(reversed(parts))
