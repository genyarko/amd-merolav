"""Function-level chunking for large-file migration.

Splits a Python source file into logical chunks (functions, classes,
module-level blocks) so each can be migrated independently within
LLM context limits.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field

from core.logging import get_logger

logger = get_logger(__name__)

# Rough chars-per-token estimate for Python code
_CHARS_PER_TOKEN = 4

# Patterns that indicate CUDA usage (fast regex pre-filter)
_CUDA_PATTERNS = re.compile(
    r"(?:cuda|torch\.cuda|cupy|pycuda|numba\.cuda|cudnn|nccl|nvtx|"
    r"SourceModule|CUDA|gpu|GPU|device\s*=|\.to\(['\"]cuda)"
)


@dataclass
class CodeChunk:
    """A logical unit of source code."""

    name: str  # e.g. "imports", "class:MyModel", "func:train"
    kind: str  # "imports", "module", "function", "class"
    start_line: int  # 1-based inclusive
    end_line: int  # 1-based inclusive
    source: str  # the actual code text
    has_cuda: bool = False  # whether CUDA patterns were detected
    dependencies: list[str] = field(default_factory=list)  # names this chunk references

    @property
    def estimated_tokens(self) -> int:
        return max(1, len(self.source) // _CHARS_PER_TOKEN)


@dataclass
class ChunkResult:
    """Result of chunking a source file."""

    chunks: list[CodeChunk]
    shared_context: str  # imports + global assignments needed by all chunks
    total_tokens: int

    @property
    def cuda_chunks(self) -> list[CodeChunk]:
        return [c for c in self.chunks if c.has_cuda]

    @property
    def clean_chunks(self) -> list[CodeChunk]:
        return [c for c in self.chunks if not c.has_cuda]


def chunk_source(source: str, max_chunk_tokens: int = 4000) -> ChunkResult:
    """Split source into AST-aware chunks.

    Args:
        source: Python source code.
        max_chunk_tokens: Maximum estimated tokens per chunk.

    Returns:
        A ChunkResult with the list of chunks and shared context.
    """
    lines = source.splitlines(keepends=True)
    chunks: list[CodeChunk] = []

    try:
        tree = ast.parse(source)
    except SyntaxError:
        # Can't parse — treat the whole file as one chunk
        logger.warning("Cannot parse source for chunking — using single chunk")
        whole = CodeChunk(
            name="<entire-file>",
            kind="module",
            start_line=1,
            end_line=len(lines),
            source=source,
            has_cuda=bool(_CUDA_PATTERNS.search(source)),
        )
        return ChunkResult(
            chunks=[whole],
            shared_context="",
            total_tokens=whole.estimated_tokens,
        )

    # Collect import lines and top-level assignments as shared context
    import_lines: list[int] = []  # 0-based
    global_assign_lines: list[int] = []

    # Track which line ranges are claimed by top-level nodes
    node_ranges: list[tuple[int, int, ast.AST]] = []

    for node in ast.iter_child_nodes(tree):
        if not hasattr(node, "lineno"):
            continue

        start = node.lineno  # 1-based
        end = _node_end_line(node, lines)

        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for ln in range(start - 1, end):
                import_lines.append(ln)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            node_ranges.append((start, end, node))
        elif isinstance(node, ast.ClassDef):
            node_ranges.append((start, end, node))
        elif isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            for ln in range(start - 1, end):
                global_assign_lines.append(ln)
        else:
            # Other module-level statements (if, try, etc.)
            node_ranges.append((start, end, node))

    # Build shared context
    shared_line_indices = sorted(set(import_lines + global_assign_lines))
    shared_context = "".join(lines[i] for i in shared_line_indices if i < len(lines))

    # Build chunks from node ranges
    for start, end, node in node_ranges:
        chunk_source_text = "".join(lines[start - 1 : end])
        name = _chunk_name(node)
        kind = _chunk_kind(node)
        has_cuda = bool(_CUDA_PATTERNS.search(chunk_source_text))

        chunk = CodeChunk(
            name=name,
            kind=kind,
            start_line=start,
            end_line=end,
            source=chunk_source_text,
            has_cuda=has_cuda,
        )
        chunks.append(chunk)

    # If there are no function/class chunks, treat as single module chunk
    if not chunks:
        whole = CodeChunk(
            name="<module>",
            kind="module",
            start_line=1,
            end_line=len(lines),
            source=source,
            has_cuda=bool(_CUDA_PATTERNS.search(source)),
        )
        return ChunkResult(
            chunks=[whole],
            shared_context=shared_context,
            total_tokens=whole.estimated_tokens,
        )

    # Merge small adjacent chunks to avoid overhead, and split oversized ones
    chunks = _merge_small_chunks(chunks, max_chunk_tokens)
    chunks = _split_large_chunks(chunks, lines, max_chunk_tokens)

    total_tokens = sum(c.estimated_tokens for c in chunks)
    logger.info(
        "Chunked source into %d chunks (%d with CUDA, ~%d total tokens)",
        len(chunks),
        sum(1 for c in chunks if c.has_cuda),
        total_tokens,
    )

    return ChunkResult(
        chunks=chunks,
        shared_context=shared_context,
        total_tokens=total_tokens,
    )


def reassemble_chunks(
    original_source: str,
    chunks: list[CodeChunk],
    migrated_chunks: dict[str, str],
) -> str:
    """Reassemble migrated chunks into a complete file.

    Args:
        original_source: The original full source.
        chunks: The chunk list from chunking.
        migrated_chunks: Map of chunk.name -> migrated source text.

    Returns:
        The reassembled file with migrated chunks replacing originals.
    """
    lines = original_source.splitlines(keepends=True)
    # Process chunks in reverse order so line indices remain valid
    sorted_chunks = sorted(chunks, key=lambda c: c.start_line, reverse=True)

    for chunk in sorted_chunks:
        migrated = migrated_chunks.get(chunk.name)
        if migrated is None:
            continue  # chunk wasn't migrated, keep original
        migrated_lines = migrated.splitlines(keepends=True)
        # Ensure trailing newline
        if migrated_lines and not migrated_lines[-1].endswith("\n"):
            migrated_lines[-1] += "\n"
        lines[chunk.start_line - 1 : chunk.end_line] = migrated_lines

    return "".join(lines)


def _node_end_line(node: ast.AST, lines: list[str]) -> int:
    """Get the end line of an AST node (1-based)."""
    if hasattr(node, "end_lineno") and node.end_lineno is not None:
        return node.end_lineno

    # Fallback: find the last child node's end line
    max_line = node.lineno
    for child in ast.walk(node):
        if hasattr(child, "end_lineno") and child.end_lineno is not None:
            max_line = max(max_line, child.end_lineno)
        elif hasattr(child, "lineno"):
            max_line = max(max_line, child.lineno)
    return max_line


def _chunk_name(node: ast.AST) -> str:
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return f"func:{node.name}"
    if isinstance(node, ast.ClassDef):
        return f"class:{node.name}"
    return f"stmt:L{node.lineno}"


def _chunk_kind(node: ast.AST) -> str:
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return "function"
    if isinstance(node, ast.ClassDef):
        return "class"
    return "module"


def _merge_small_chunks(
    chunks: list[CodeChunk], max_tokens: int
) -> list[CodeChunk]:
    """Merge consecutive small chunks (< 25% of max) into combined chunks."""
    if not chunks:
        return chunks

    threshold = max_tokens // 4
    merged: list[CodeChunk] = []
    acc: list[CodeChunk] = []
    acc_tokens = 0

    for chunk in chunks:
        if chunk.estimated_tokens < threshold and acc_tokens + chunk.estimated_tokens < max_tokens:
            acc.append(chunk)
            acc_tokens += chunk.estimated_tokens
        else:
            if acc:
                merged.append(_combine_chunks(acc))
            acc = [chunk] if chunk.estimated_tokens < threshold else []
            acc_tokens = chunk.estimated_tokens if chunk.estimated_tokens < threshold else 0
            if chunk.estimated_tokens >= threshold:
                merged.append(chunk)

    if acc:
        merged.append(_combine_chunks(acc))

    return merged


def _combine_chunks(chunks: list[CodeChunk]) -> CodeChunk:
    """Combine multiple chunks into one."""
    if len(chunks) == 1:
        return chunks[0]

    names = "+".join(c.name for c in chunks)
    source = "\n".join(c.source.rstrip("\n") for c in chunks) + "\n"
    has_cuda = any(c.has_cuda for c in chunks)

    return CodeChunk(
        name=names,
        kind="module",
        start_line=chunks[0].start_line,
        end_line=chunks[-1].end_line,
        source=source,
        has_cuda=has_cuda,
    )


def _split_large_chunks(
    chunks: list[CodeChunk], lines: list[str], max_tokens: int
) -> list[CodeChunk]:
    """Split oversized chunks by line count."""
    result: list[CodeChunk] = []
    max_chars = max_tokens * _CHARS_PER_TOKEN

    for chunk in chunks:
        if len(chunk.source) <= max_chars:
            result.append(chunk)
            continue

        # Split by lines
        chunk_lines = chunk.source.splitlines(keepends=True)
        lines_per_sub = max(1, max_chars // max(1, len(chunk.source) // len(chunk_lines)))
        part = 0

        for i in range(0, len(chunk_lines), lines_per_sub):
            part += 1
            sub_lines = chunk_lines[i : i + lines_per_sub]
            sub_source = "".join(sub_lines)
            sub = CodeChunk(
                name=f"{chunk.name}:part{part}",
                kind=chunk.kind,
                start_line=chunk.start_line + i,
                end_line=chunk.start_line + i + len(sub_lines) - 1,
                source=sub_source,
                has_cuda=bool(_CUDA_PATTERNS.search(sub_source)),
            )
            result.append(sub)

    return result
