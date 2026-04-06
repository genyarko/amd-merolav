"""Orchestrator — wires Coder, Reviewer, and Tester into an AutoGen GroupChat."""

from __future__ import annotations

import re
from typing import Any

from autogen import ConversableAgent, GroupChat, GroupChatManager
from autogen.oai import OpenAIWrapper as _OAIWrapper

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
from core.migrator import MigrationResult


def run_migration_agents(
    original_code: str,
    migration_result: MigrationResult,
    report: AnalysisReport,
    settings: Settings,
    verbose: bool = False,
    console: "Any | None" = None,
) -> str:
    """Run the two-phase planner/executor migration pipeline.

    Flow:
      Phase 1 — Planner (DeepSeek-R1): reasons through the migration,
                 produces a step-by-step plan.
      Phase 2 — GroupChat: Executor (Qwen2.5-Coder) follows the plan,
                 Reviewer validates, Tester runs AST checks.
                 Terminates early if Tester outputs "ALL_TESTS_PASSED".

    Args:
        original_code: The user's original CUDA source code.
        migration_result: Output from the rule-based migrator.
        report: The analyzer's CUDA usage report.
        settings: Application settings (backend, max_rounds, etc.).

    Returns:
        The final migrated Python code as a string.
    """
    from rich.console import Console as RichConsole
    from rich.panel import Panel
    con = console or RichConsole()

    # --- Phase 1: Planning ---

    plan = run_planner(original_code, migration_result, report, settings,
                       verbose=verbose, console=con)

    # --- Phase 2: Execution GroupChat ---

    # Use executor config (Qwen2.5-Coder) for both Executor and Reviewer
    if settings.default_backend == "self-hosted":
        llm_config = get_executor_config(settings)
    else:
        config_list = get_config_list(settings.default_backend, settings)
        llm_config: dict[str, Any] = {"config_list": config_list, "temperature": 0.1}

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

    coder.initiate_chat(manager, message=seed_message)

    # --- Extract final code from conversation ---

    return _extract_final_code(groupchat.messages)


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
