"""CLI entry point for the CUDA-to-ROCm migration agent."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table

from config.settings import Settings
from core.analyzer import analyze_source
from core.differ import generate_diff
from core.file_io import collect_python_files, read_source_file, write_output_file
from core.migrator import migrate

app = typer.Typer(
    name="rocm-migrate",
    help="Migrate CUDA Python/PyTorch code to AMD ROCm.",
    add_completion=False,
)
console = Console()


@app.command()
def main(
    input_path: str = typer.Argument(
        ..., help="Path to .py file or directory of .py files"
    ),
    backend: str = typer.Option(
        "mistral",
        help="Model backend: self-hosted | mistral | deepseek | claude",
    ),
    planner_url: Optional[str] = typer.Option(
        None,
        help="Planner vLLM server URL (overrides .env PLANNER_BASE_URL)",
    ),
    max_rounds: int = typer.Option(
        6, help="Max agent refinement rounds"
    ),
    output: str = typer.Option(
        "./rocm_output/", help="Output directory for migrated files"
    ),
    diff_only: bool = typer.Option(
        False, "--diff-only", help="Only show diff, don't write files"
    ),
    no_agent: bool = typer.Option(
        False, "--no-agent", help="Skip LLM agents, only apply rule-based migration"
    ),
    force_agents: bool = typer.Option(
        False, "--force-agents", help="Always run LLM agents even if rule-based pass resolved everything"
    ),
    no_test: bool = typer.Option(
        False, "--no-test", help="Skip validation step"
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Show detailed output"
    ),
) -> None:
    """Migrate CUDA Python code to ROCm."""
    # Build settings
    settings = Settings()
    settings.default_backend = backend
    if planner_url is not None:
        settings.planner_base_url = planner_url
    settings.max_rounds = max_rounds

    # Collect files
    try:
        files = collect_python_files(input_path)
    except (FileNotFoundError, ValueError) as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)

    console.print(
        Panel(f"ROCm Migration Agent — {len(files)} file(s) to process",
              style="bold blue")
    )

    for file_path in files:
        _process_file(
            file_path=file_path,
            settings=settings,
            output_dir=Path(output),
            diff_only=diff_only,
            no_agent=no_agent,
            force_agents=force_agents,
            no_test=no_test,
            verbose=verbose,
        )


def _process_file(
    file_path: Path,
    settings: Settings,
    output_dir: Path,
    diff_only: bool,
    no_agent: bool,
    force_agents: bool,
    no_test: bool,
    verbose: bool,
) -> None:
    """Process a single file through the migration pipeline."""
    console.rule(f"[bold]{file_path.name}[/bold]")

    # 1. Read source
    source = read_source_file(file_path)
    console.print(f"  Read {len(source.splitlines())} lines")

    # 2. Analyze
    console.print("  [cyan]Analyzing CUDA usage...[/cyan]")
    report = analyze_source(source, str(file_path))

    if not report.has_cuda:
        console.print("  [green]No CUDA-specific code detected. Skipping.[/green]")
        return

    # Show analysis summary
    _print_analysis_summary(report)

    # 3. Rule-based migration
    console.print("  [cyan]Applying rule-based migrations...[/cyan]")
    result = migrate(source, report)

    if verbose:
        for change in result.applied:
            console.print(f"    [green]Line {change.line}:[/green] {change.rule}")

    console.print(
        f"  Applied [green]{len(result.applied)}[/green] automatic changes, "
        f"[yellow]{len(result.remaining)}[/yellow] remaining for LLM"
    )

    # 4. LLM agent loop (optional)
    final_code = result.code
    if not no_agent and (result.remaining or force_agents):
        console.print(f"  [cyan]Running agent loop (backend={settings.default_backend})...[/cyan]")
        try:
            from agents.orchestrator import run_migration_agents
            final_code = run_migration_agents(
                original_code=source,
                migration_result=result,
                report=report,
                settings=settings,
                verbose=verbose,
                console=console,
            )
            if final_code:
                console.print("  [green]Agent migration complete.[/green]")
            else:
                console.print("  [yellow]Agent returned no code, using rule-based result.[/yellow]")
                final_code = result.code
        except Exception as e:
            console.print(f"  [red]Agent error: {e}[/red]")
            console.print("  [yellow]Falling back to rule-based result.[/yellow]")
            final_code = result.code
    elif no_agent:
        console.print("  [dim]Skipping agent loop (--no-agent)[/dim]")

    # 5. Validation (optional)
    if not no_test:
        console.print("  [cyan]Validating migrated code...[/cyan]")
        from agents.tester import run_validation
        validation = run_validation(final_code)
        if "ALL_TESTS_PASSED" in validation:
            console.print("  [green]All validation checks passed.[/green]")
        else:
            console.print(f"  [yellow]Validation issues:[/yellow]")
            for line in validation.splitlines():
                if "FAIL" in line:
                    console.print(f"    [red]{line.strip()}[/red]")
                elif line.strip().startswith("-"):
                    console.print(f"    [yellow]{line.strip()}[/yellow]")

    # 6. Show diff
    diff = generate_diff(source, final_code, file_path.name)
    if diff:
        console.print()
        console.print(Panel("Diff", style="bold"))
        console.print(Syntax(diff, "diff", theme="monokai"))
    else:
        console.print("  [dim]No changes made.[/dim]")

    # 7. Show optimization suggestions
    if result.optimizations:
        _print_optimizations(result.optimizations)

    # 8. Write output
    if not diff_only and diff:
        out_path = output_dir / file_path.name
        write_output_file(out_path, final_code)
        console.print(f"\n  [green]Written to {out_path}[/green]")


def _print_analysis_summary(report) -> None:
    """Print a table of detected CUDA usages."""
    table = Table(title="CUDA Usage Detected", show_lines=False)
    table.add_column("Category", style="cyan")
    table.add_column("Count", justify="right", style="bold")
    for cat, count in sorted(report.summary.items()):
        table.add_column
        table.add_row(cat, str(count))
    console.print(table)


def _print_optimizations(optimizations) -> None:
    """Print AMD-specific optimization suggestions."""
    console.print()
    console.print(Panel("AMD Optimization Suggestions", style="bold green"))
    for opt in optimizations:
        priority_color = {1: "red", 2: "yellow", 3: "dim"}.get(opt.priority, "white")
        console.print(f"  [{priority_color}]P{opt.priority}[/{priority_color}] "
                      f"[bold]{opt.category}[/bold]: {opt.suggestion}")
        if opt.url:
            console.print(f"      [dim]{opt.url}[/dim]")


if __name__ == "__main__":
    app()
