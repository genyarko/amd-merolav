"""Orchestrator — wires Coder, Reviewer, and Tester into an AutoGen GroupChat."""

from __future__ import annotations

import re
import time
from typing import Any

from autogen import ConversableAgent, GroupChat, GroupChatManager
from autogen.oai import OpenAIWrapper as _OAIWrapper

from core.logging import get_logger

logger = get_logger(__name__)

# Mistral and vLLM reject the 'name' field on messages that ag2 adds for
# multi-agent tracking. Strip it globally before every API call.
_orig_create = _OAIWrapper.create

def _create_strip_names(self, **config):
    if "messages" in config:
        config["messages"] = [
            {k: v for k, v in m.items() if k != "name"}
            for m in config["messages"]
        ]
    return _orig_create(self, **config)

_OAIWrapper.create = _create_strip_names

from agents.coder import CODER_SYSTEM_PROMPT, format_coder_message
from agents.planner import run_planner
from agents.reviewer import REVIEWER_SYSTEM_PROMPT
from agents.tester import run_validation
from config.model_profiles import get_config_list, get_executor_config
from config.settings import Settings
from core.analyzer import AnalysisReport
from core.chunker import chunk_source, reassemble_chunks, CodeChunk
from core.context_budget import ContextBudget, allocate_budget, estimate_tokens, needs_chunking
from core.migrator import MigrationResult


def run_migration_agents(
    original_code: str,
    migration_result: MigrationResult,
    report: AnalysisReport,
    settings: Settings,
    verbose: bool = False,
    console: "Any | None" = None,
    cache: "Any | None" = None,
    chunk_size: int | None = None,
    on_chunk_complete: "Any | None" = None,
) -> str:
    """Run the two-phase planner/executor migration pipeline.

    Flow:
      Phase 1 — Planner (DeepSeek-R1): reasons through the migration,
                 produces a step-by-step plan.
      Phase 2 — GroupChat: Executor (Qwen2.5-Coder) follows the plan,
                 Reviewer validates, Tester runs AST checks.
                 Terminates early if Tester outputs "ALL_TESTS_PASSED".

    For large files, the source is split into chunks and each CUDA-containing
    chunk is migrated independently. Clean chunks are passed through unchanged.

    Args:
        original_code: The user's original CUDA source code.
        migration_result: Output from the rule-based migrator.
        report: The analyzer's CUDA usage report.
        settings: Application settings (backend, max_rounds, etc.).
        chunk_size: Override max tokens per chunk (default 4000).
        on_chunk_complete: Optional callback(chunk_name, index, total) for progress.

    Returns:
        The final migrated Python code as a string.
    """
    from rich.console import Console as RichConsole
    from rich.panel import Panel
    con = console or RichConsole()

    logger.info(
        "Starting agent migration pipeline (backend=%s, max_rounds=%d)",
        settings.default_backend,
        settings.max_rounds,
    )

    # --- Check if chunking is needed ---
    effective_chunk_size = chunk_size or 4000
    if needs_chunking(original_code, settings.default_backend, chunk_size):
        return _run_chunked_migration(
            original_code=original_code,
            migration_result=migration_result,
            report=report,
            settings=settings,
            verbose=verbose,
            console=con,
            cache=cache,
            chunk_size=effective_chunk_size,
            on_chunk_complete=on_chunk_complete,
        )

    # --- Phase 1: Planning ---

    plan = run_planner(original_code, migration_result, report, settings,
                       verbose=verbose, console=con, cache=cache)

    # --- Phase 2: Execution GroupChat ---

    # Use executor config (Qwen2.5-Coder) for both Executor and Reviewer
    try:
        if settings.default_backend == "self-hosted":
            llm_config = get_executor_config(settings)
        else:
            config_list = get_config_list(settings.default_backend, settings)
            llm_config: dict[str, Any] = {"config_list": config_list, "temperature": 0.1}
        logger.debug("Executor LLM config loaded for backend=%s", settings.default_backend)
    except Exception as exc:
        logger.error("Failed to load LLM config for backend=%s: %s", settings.default_backend, exc)
        raise

    # --- Verbose observer hook ---

    AGENT_STYLES = {
        "Executor": ("bold green", "Executor (Qwen2.5-Coder)"),
        "Reviewer": ("bold yellow", "Reviewer"),
        "Tester":   ("bold magenta", "Tester"),
    }

    def _make_observer(agent_name: str):
        style, label = AGENT_STYLES.get(agent_name, ("white", agent_name))

        def _observer(recipient, messages, sender, config):
            if verbose and messages:
                content = messages[-1].get("content", "")
                # Truncate long code blocks for display
                preview = content if len(content) <= 1200 else content[:1200] + "\n... [truncated]"
                con.print(Panel(
                    preview,
                    title=f"[{style}]{label}[/{style}]",
                    style=style.split()[-1],  # use just the color
                ))
            return False, None  # observe only, don't intercept

        return _observer

    # --- Create agents ---

    coder = ConversableAgent(
        name="Executor",
        system_message=CODER_SYSTEM_PROMPT,
        llm_config=llm_config,
        human_input_mode="NEVER",
    )
    coder.register_reply(ConversableAgent, _make_observer("Executor"), position=0)

    reviewer = ConversableAgent(
        name="Reviewer",
        system_message=REVIEWER_SYSTEM_PROMPT,
        llm_config=llm_config,
        human_input_mode="NEVER",
    )
    reviewer.register_reply(ConversableAgent, _make_observer("Reviewer"), position=0)

    tester = ConversableAgent(
        name="Tester",
        system_message=(
            "You are the Tester. When you receive code, run the validation "
            "function and report results. You do not write code."
        ),
        llm_config=False,
        human_input_mode="NEVER",
        is_termination_msg=lambda msg: "ALL_TESTS_PASSED" in msg.get("content", ""),
    )

    def _tester_reply(
        recipient: ConversableAgent,
        messages: list[dict],
        sender: ConversableAgent,
        config: Any,
    ) -> tuple[bool, str]:
        """Extract code from the most recent code block in conversation history."""
        # Search backward — Reviewer may say "APPROVED" without repeating code
        code = None
        for msg in reversed(messages or []):
            code = _extract_code_block(msg.get("content", ""))
            if code:
                break
        result = run_validation(code) if code else (
            "VALIDATION FAILED\n\n"
            "  [FAIL] No code block found in the previous message.\n"
            "  Please provide the migrated code in a ```python code block."
        )
        if verbose:
            passed = "ALL_TESTS_PASSED" in result
            style = "bold green" if passed else "bold red"
            con.print(Panel(result, title=f"[{style}]Tester[/{style}]",
                            style="green" if passed else "red"))
        return True, result

    tester.register_reply(ConversableAgent, _tester_reply, position=0)

    # --- Build the group chat ---

    groupchat = GroupChat(
        agents=[coder, reviewer, tester],
        messages=[],
        max_round=settings.max_rounds,
        speaker_selection_method="round_robin",  # Coder→Reviewer→Tester, no LLM needed
    )

    manager = GroupChatManager(
        groupchat=groupchat,
        llm_config=False,  # round_robin needs no LLM for speaker selection
        is_termination_msg=lambda msg: "ALL_TESTS_PASSED" in msg.get("content", ""),
    )

    # --- Build the seed message ---

    remaining_issues = [
        {"line": iss.line, "symbol": iss.symbol, "reason": iss.reason}
        for iss in migration_result.remaining
    ]

    seed_message = format_coder_message(
        original_code=original_code,
        pre_migrated_code=migration_result.code,
        remaining_issues=remaining_issues,
    )

    if plan:
        seed_message = (
            f"## Migration Plan (from Planner)\n\n{plan}\n\n---\n\n{seed_message}"
        )

    # --- Run the conversation ---

    logger.info("Initiating agent group chat (%d max rounds)", settings.max_rounds)
    start_time = time.monotonic()

    try:
        coder.initiate_chat(manager, message=seed_message)
    except Exception as exc:
        elapsed = time.monotonic() - start_time
        logger.error(
            "Agent group chat failed after %.1fs: %s",
            elapsed,
            exc,
        )
        # Graceful fallback: return the rule-based result instead of crashing
        logger.warning(
            "Falling back to rule-based migration result (agent pipeline failed)"
        )
        return migration_result.code

    elapsed = time.monotonic() - start_time
    n_messages = len(groupchat.messages)
    logger.info(
        "Agent group chat completed in %.1fs (%d messages exchanged)",
        elapsed,
        n_messages,
    )

    # --- Extract final code from conversation ---

    final_code = _extract_final_code(groupchat.messages)
    if not final_code:
        logger.warning("No code block found in agent conversation — returning empty result")
    else:
        logger.debug("Extracted final code (%d chars)", len(final_code))

    return final_code


def _run_chunked_migration(
    original_code: str,
    migration_result: MigrationResult,
    report: AnalysisReport,
    settings: Settings,
    verbose: bool,
    console: Any,
    cache: Any | None,
    chunk_size: int,
    on_chunk_complete: Any | None,
) -> str:
    """Migrate a large file chunk by chunk.

    Splits the pre-migrated code into AST-aware chunks, sends only
    CUDA-containing chunks through the agent pipeline, and reassembles.
    """
    from rich.panel import Panel

    con = console
    code_to_chunk = migration_result.code  # start from rule-based result

    chunk_result = chunk_source(code_to_chunk, max_chunk_tokens=chunk_size)
    cuda_chunks = chunk_result.cuda_chunks
    total_chunks = len(chunk_result.chunks)

    con.print(
        f"  [cyan]Large file detected — split into {total_chunks} chunks "
        f"({len(cuda_chunks)} with CUDA patterns)[/cyan]"
    )

    if not cuda_chunks:
        con.print("  [green]No CUDA patterns found in any chunk after rule-based pass.[/green]")
        return code_to_chunk

    # Budget allocation
    budget = ContextBudget.for_backend(settings.default_backend)
    analysis_text = "\n".join(
        f"  L{u.line} [{u.category}] {u.symbol}" for u in report.usages
    )
    allocation = allocate_budget(
        chunk_result=chunk_result,
        analysis_text=analysis_text,
        knowledge_entries=[],  # knowledge is already baked into the rule-based pass
        budget=budget,
    )

    if allocation.was_trimmed:
        con.print(
            f"  [yellow]Context trimmed to fit budget: "
            f"{len(allocation.chunks_to_migrate)}/{total_chunks} chunks included[/yellow]"
        )

    # Migrate each CUDA chunk through the agent pipeline
    migrated_chunks: dict[str, str] = {}

    for idx, chunk in enumerate(allocation.chunks_to_migrate):
        if not chunk.has_cuda:
            continue  # skip clean chunks

        chunk_num = idx + 1
        con.print(
            f"  [cyan]Migrating chunk {chunk_num}/{len(cuda_chunks)}: "
            f"{chunk.name} (L{chunk.start_line}-{chunk.end_line}, "
            f"~{chunk.estimated_tokens} tokens)...[/cyan]"
        )

        # Check cache for this chunk — use source hash for stable key across edits
        chunk_cache_key = f"__chunk__{cache.hash_source(chunk.source, settings.default_backend)[:16]}"
        if cache and cache.is_unchanged(chunk_cache_key, chunk.source, settings.default_backend):
            cached = cache.get_migration(chunk.source, settings.default_backend)
            if cached:
                migrated_chunks[chunk.name] = cached.get("migrated_code", chunk.source)
                con.print(f"    [dim]Cached — skipping.[/dim]")
                if on_chunk_complete:
                    on_chunk_complete(chunk.name, chunk_num, len(cuda_chunks))
                continue

        # Build a mini migration context for this chunk
        chunk_context = (
            f"## Shared Context (imports & globals)\n\n"
            f"```python\n{chunk_result.shared_context}```\n\n"
            f"## Code to Migrate\n\n"
            f"```python\n{chunk.source}```\n\n"
            f"## Analysis\n\n{allocation.analysis_summary}\n"
        )

        # Run single-chunk through a simplified agent call
        try:
            chunk_code = _migrate_single_chunk(
                chunk_context=chunk_context,
                chunk_source=chunk.source,
                shared_context=chunk_result.shared_context,
                settings=settings,
                verbose=verbose,
                console=con,
            )
            if chunk_code:
                migrated_chunks[chunk.name] = chunk_code
                con.print(f"    [green]Done.[/green]")

                # Cache the chunk result
                if cache:
                    cache.put_migration(
                        file_path=chunk_cache_key,
                        source=chunk.source,
                        backend=settings.default_backend,
                        migrated_code=chunk_code,
                        applied=[],
                        remaining=[],
                    )
            else:
                con.print(f"    [yellow]No code returned — keeping original.[/yellow]")
        except Exception as exc:
            con.print(f"    [red]Error: {exc} — keeping original.[/red]")
            logger.error("Chunk migration failed for '%s': %s", chunk.name, exc)

        if on_chunk_complete:
            on_chunk_complete(chunk.name, chunk_num, len(cuda_chunks))

    # Reassemble
    if migrated_chunks:
        final_code = reassemble_chunks(code_to_chunk, chunk_result.chunks, migrated_chunks)
        con.print(
            f"  [green]Reassembled {len(migrated_chunks)} migrated chunk(s) "
            f"into final output.[/green]"
        )
        return final_code

    return code_to_chunk


def _migrate_single_chunk(
    chunk_context: str,
    chunk_source: str,
    shared_context: str,
    settings: Settings,
    verbose: bool,
    console: Any,
) -> str | None:
    """Run a single chunk through the agent pipeline.

    Uses a simplified two-agent chat (Executor + Tester) without the full
    GroupChat overhead, since each chunk is small.
    """
    try:
        if settings.default_backend == "self-hosted":
            llm_config = get_executor_config(settings)
        else:
            config_list = get_config_list(settings.default_backend, settings)
            llm_config: dict[str, Any] = {"config_list": config_list, "temperature": 0.1}
    except Exception:
        raise

    coder = ConversableAgent(
        name="ChunkExecutor",
        system_message=(
            CODER_SYSTEM_PROMPT
            + "\n\nYou are migrating a SINGLE CHUNK of a larger file. "
            "Output ONLY the migrated chunk code in a ```python block. "
            "Do NOT include imports or other code outside this chunk."
        ),
        llm_config=llm_config,
        human_input_mode="NEVER",
        max_consecutive_auto_reply=1,
    )

    tester = ConversableAgent(
        name="ChunkTester",
        system_message="Validate the migrated chunk.",
        llm_config=False,
        human_input_mode="NEVER",
        is_termination_msg=lambda msg: True,  # always terminate after one round
    )

    def _chunk_tester_reply(recipient, messages, sender, config):
        code = None
        for msg in reversed(messages or []):
            code = _extract_code_block(msg.get("content", ""))
            if code:
                break
        result = run_validation(
            shared_context + "\n" + code
        ) if code else "VALIDATION FAILED\nNo code block found."
        return True, result

    tester.register_reply(ConversableAgent, _chunk_tester_reply, position=0)

    coder.initiate_chat(tester, message=chunk_context)

    # Extract code from conversation
    for msg in reversed(coder.chat_messages.get(tester, [])):
        code = _extract_code_block(msg.get("content", ""))
        if code:
            return code

    return None


def _extract_code_block(text: str) -> str | None:
    """Extract the last Python code block from a message."""
    # Accept ```python, ```Python, or ``` with any trailing chars before newline
    pattern = r"```(?:[Pp]ython)?[^\n]*\n(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL)
    return matches[-1].strip() if matches else None


def _extract_final_code(messages: list[dict]) -> str:
    """Walk messages in reverse to find the last code block from the Coder."""
    # First, try to find the last code block from the Coder
    for msg in reversed(messages):
        if msg.get("name") in ("Executor", "Coder") or msg.get("role") == "assistant":
            code = _extract_code_block(msg.get("content", ""))
            if code:
                return code

    # Fallback: find any code block in any message
    for msg in reversed(messages):
        code = _extract_code_block(msg.get("content", ""))
        if code:
            return code

    return ""
