"""Context budget management for LLM calls.

Estimates token usage and trims inputs to fit within the LLM's context
window, prioritizing chunks that contain CUDA patterns.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from core.chunker import CodeChunk, ChunkResult
from core.logging import get_logger

logger = get_logger(__name__)

# Rough chars-per-token for Python code
_CHARS_PER_TOKEN = 4

# Default context limits by backend (leave headroom for output)
_DEFAULT_CONTEXT_LIMITS: dict[str, int] = {
    "mistral": 28000,    # Codestral ~32k, leave room
    "deepseek": 28000,   # DeepSeek-Coder ~32k
    "claude": 180000,    # Claude ~200k
    "self-hosted": 12000,  # Conservative for local models
}

# Fixed overhead for system prompt, plan, etc.
_SYSTEM_OVERHEAD_TOKENS = 2000


@dataclass
class BudgetAllocation:
    """Result of budget allocation for a migration call."""

    chunks_to_migrate: list[CodeChunk]
    shared_context: str
    analysis_summary: str
    knowledge_entries: list[str]
    total_estimated_tokens: int
    was_trimmed: bool = False


@dataclass
class ContextBudget:
    """Manages the token budget for a single LLM call."""

    max_tokens: int
    reserved_output: int = 4000

    @property
    def available(self) -> int:
        return self.max_tokens - self.reserved_output - _SYSTEM_OVERHEAD_TOKENS

    @classmethod
    def for_backend(cls, backend: str, override: int | None = None,
                    settings: "Any | None" = None) -> ContextBudget:
        if override:
            limit = override
        elif settings and getattr(settings, "executor_context_limit", 0) > 0:
            limit = settings.executor_context_limit
        else:
            limit = _DEFAULT_CONTEXT_LIMITS.get(backend, 12000)
        return cls(max_tokens=limit)


def estimate_tokens(text: str) -> int:
    """Estimate the number of tokens in a text string."""
    return max(1, len(text) // _CHARS_PER_TOKEN)


def allocate_budget(
    chunk_result: ChunkResult,
    analysis_text: str,
    knowledge_entries: list[str],
    budget: ContextBudget,
) -> BudgetAllocation:
    """Allocate the available context budget across chunks and context.

    Priority order:
    1. Shared context (imports, globals) — always included
    2. Chunks with CUDA patterns — included first
    3. Analysis report — summarized if too long
    4. Knowledge base entries — only include relevant ones
    5. Clean chunks — included only if budget allows

    Args:
        chunk_result: The chunked source file.
        analysis_text: The full analysis report as text.
        knowledge_entries: Relevant knowledge base entries.
        budget: The context budget to work within.

    Returns:
        A BudgetAllocation with what fits in the budget.
    """
    available = budget.available
    used = 0
    was_trimmed = False

    # 1. Shared context — always included
    shared = chunk_result.shared_context
    used += estimate_tokens(shared)

    # 2. CUDA chunks — prioritized
    cuda_chunks = sorted(chunk_result.cuda_chunks, key=lambda c: c.estimated_tokens)
    clean_chunks = chunk_result.clean_chunks
    selected_chunks: list[CodeChunk] = []

    for chunk in cuda_chunks:
        if used + chunk.estimated_tokens <= available:
            selected_chunks.append(chunk)
            used += chunk.estimated_tokens
        else:
            was_trimmed = True
            logger.warning(
                "Chunk '%s' (~%d tokens) exceeds remaining budget — skipping",
                chunk.name,
                chunk.estimated_tokens,
            )

    # 3. Analysis — summarize if too large
    analysis_tokens = estimate_tokens(analysis_text)
    if used + analysis_tokens <= available:
        summary = analysis_text
        used += analysis_tokens
    else:
        # Summarize: first 20 lines + count
        analysis_lines = analysis_text.splitlines()
        if len(analysis_lines) > 20:
            summary = "\n".join(analysis_lines[:20])
            summary += f"\n... ({len(analysis_lines) - 20} more lines omitted)"
            was_trimmed = True
        else:
            summary = analysis_text
        used += estimate_tokens(summary)

    # 4. Knowledge entries — include what fits
    selected_knowledge: list[str] = []
    for entry in knowledge_entries:
        entry_tokens = estimate_tokens(entry)
        if used + entry_tokens <= available:
            selected_knowledge.append(entry)
            used += entry_tokens
        else:
            was_trimmed = True
            break

    # 5. Clean chunks — include if budget allows
    for chunk in clean_chunks:
        if used + chunk.estimated_tokens <= available:
            selected_chunks.append(chunk)
            used += chunk.estimated_tokens

    # Sort selected chunks by start line to preserve order
    selected_chunks.sort(key=lambda c: c.start_line)

    if was_trimmed:
        logger.info(
            "Context was trimmed to fit budget: %d/%d tokens used, "
            "%d/%d chunks included",
            used,
            budget.available,
            len(selected_chunks),
            len(chunk_result.chunks),
        )

    return BudgetAllocation(
        chunks_to_migrate=selected_chunks,
        shared_context=shared,
        analysis_summary=summary,
        knowledge_entries=selected_knowledge,
        total_estimated_tokens=used,
        was_trimmed=was_trimmed,
    )


def needs_chunking(source: str, backend: str, chunk_size: int | None = None,
                   settings: "Any | None" = None) -> bool:
    """Check whether a source file needs chunking for the given backend."""
    budget = ContextBudget.for_backend(backend, override=chunk_size, settings=settings)
    source_tokens = estimate_tokens(source)
    # If the source + overhead fits in context, no need to chunk
    return source_tokens > (budget.available - _SYSTEM_OVERHEAD_TOKENS)
