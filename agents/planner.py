"""Planner agent — uses DeepSeek-R1 to reason through a migration plan."""

from __future__ import annotations

import time

from autogen.oai import OpenAIWrapper

from config.model_profiles import get_planner_config
from config.settings import Settings
from core.analyzer import AnalysisReport
from core.logging import get_logger
from core.migrator import MigrationResult

logger = get_logger(__name__)

PLANNER_SYSTEM_PROMPT = """\
You are a senior GPU software architect specializing in CUDA-to-ROCm migrations.

Your job is to REASON and PLAN — not to write code.

Given:
- The original CUDA source code
- The static analysis report (what CUDA APIs/patterns were found)
- A partial rule-based migration (high-confidence changes already applied)
- The remaining issues that need LLM attention

You must produce a numbered, step-by-step migration plan that the Executor (a coding agent) will follow exactly.

For each step:
1. Identify the specific CUDA construct to address (line numbers if known)
2. State the correct ROCm/HIP equivalent
3. Flag any gotchas or ROCm-specific behavior differences
4. Indicate priority: CRITICAL / IMPORTANT / OPTIONAL

Key ROCm facts to reason with:
- torch.device("cuda") is CORRECT on ROCm — do NOT change to "rocm" or "hip"
- torch.cuda.is_available() returns True on ROCm — do NOT change
- torch.cuda.amp works — prefer torch.amp.autocast("cuda") on PyTorch >= 2.0
- torch.backends.cudnn.* → torch.backends.miopen.*
- NCCL → RCCL (API-compatible, keep backend="nccl")
- CUDA_VISIBLE_DEVICES → HIP_VISIBLE_DEVICES
- pycuda → hip Python bindings
- Custom CUDA kernels (<<<...>>>) require HIPIFY or manual HIP porting

End your plan with:
PLAN COMPLETE — N steps for the Executor.
"""


def format_planner_message(
    original_code: str,
    migration_result: MigrationResult,
    report: AnalysisReport,
) -> str:
    issues = "\n".join(
        f"  Line {i.line}: {i.symbol} — {i.reason}"
        for i in migration_result.remaining
    ) or "  (none)"

    applied = "\n".join(
        f"  Line {c.line}: {c.original!r} → {c.replacement!r}"
        for c in migration_result.applied
    ) or "  (none)"

    summary_lines = "\n".join(
        f"  {cat}: {count}" for cat, count in report.summary.items()
    )

    return f"""\
## Migration Task

### CUDA Usage Summary
{summary_lines}

### Already Applied (rule-based, high-confidence)
{applied}

### Remaining Issues (need your plan)
{issues}

### Original Source Code
```python
{original_code}
```

### Partially Migrated Code (after rule-based pass)
```python
{migration_result.code}
```

Produce a step-by-step migration plan for the Executor to implement.
"""


_MAX_RETRIES = 3
_RETRY_BACKOFF = (2, 5, 10)  # seconds between retries


def run_planner(
    original_code: str,
    migration_result: MigrationResult,
    report: AnalysisReport,
    settings: Settings,
    verbose: bool = False,
    console: "Console | None" = None,
    cache: "MigrationCache | None" = None,
) -> str:
    """Run the Planner agent and return its migration plan as a string.

    Retries up to 3 times with exponential backoff on transient errors.
    Returns an empty string if all retries are exhausted.

    If a cache is provided, checks for a cached plan before calling the LLM.
    """
    from rich.console import Console as RichConsole
    from rich.panel import Panel
    con = console or RichConsole()

    logger.info(
        "Starting Planner (model=%s, url=%s)",
        settings.planner_model,
        settings.planner_base_url,
    )

    # --- Check planner cache ---
    remaining_issues = [
        {"line": i.line, "symbol": i.symbol, "reason": i.reason}
        for i in migration_result.remaining
    ]
    if cache is not None:
        cached_plan = cache.get_planner(original_code, remaining_issues)
        if cached_plan is not None:
            logger.info("Using cached planner output (%d chars)", len(cached_plan))
            if verbose:
                con.print(Panel(
                    "[bold cyan]Planner[/bold cyan] — using cached plan",
                    style="cyan",
                ))
                con.print(Panel(cached_plan, title="[bold cyan]Planner Output (cached)[/bold cyan]", style="cyan"))
            return cached_plan

    if verbose:
        con.print(Panel(
            f"[bold cyan]Planner[/bold cyan] — DeepSeek-R1 reasoning...\n"
            f"[dim]{settings.planner_model}  @  {settings.planner_base_url}[/dim]",
            style="cyan"
        ))

    llm_config = get_planner_config(settings)
    client = OpenAIWrapper(config_list=llm_config["config_list"])

    message = format_planner_message(original_code, migration_result, report)
    messages = [
        {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
        {"role": "user", "content": message},
    ]

    last_error: Exception | None = None
    for attempt in range(_MAX_RETRIES):
        try:
            logger.debug("Planner attempt %d/%d", attempt + 1, _MAX_RETRIES)
            response = client.create(
                messages=messages,
                model=settings.planner_model,
                temperature=llm_config.get("temperature", 0.6),
            )

            texts = client.extract_text_or_completion_object(response)
            if not texts:
                plan = ""
            elif isinstance(texts[0], str):
                plan = texts[0]
            else:
                plan = getattr(texts[0], "content", "") or ""

            logger.info("Planner produced plan (%d chars)", len(plan))

            # Cache the plan for future runs
            if cache is not None and plan:
                cache.put_planner(original_code, remaining_issues, plan)

            if verbose and plan:
                con.print(Panel(plan, title="[bold cyan]Planner Output[/bold cyan]", style="cyan"))

            return plan

        except Exception as exc:
            last_error = exc
            backoff = _RETRY_BACKOFF[min(attempt, len(_RETRY_BACKOFF) - 1)]
            logger.warning(
                "Planner attempt %d/%d failed: %s. Retrying in %ds...",
                attempt + 1,
                _MAX_RETRIES,
                exc,
                backoff,
            )
            if attempt < _MAX_RETRIES - 1:
                time.sleep(backoff)

    logger.error(
        "Planner failed after %d attempts. Last error: %s. "
        "Proceeding without a plan — the Executor will work from the remaining issues list.",
        _MAX_RETRIES,
        last_error,
    )
    con.print(
        f"[yellow]  Planner unavailable ({last_error}). "
        f"Proceeding without a plan.[/yellow]"
    )
    return ""
