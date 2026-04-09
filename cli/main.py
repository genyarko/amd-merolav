"""CLI entry point for the CUDA-to-ROCm migration agent."""

from __future__ import annotations

import ast
import glob as globmod
import json
import time
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.syntax import Syntax
from rich.table import Table

from config.settings import Settings
from core.analyzer import analyze_source
from core.differ import generate_diff
from core.file_io import collect_python_files, read_source_file, write_output_file
from core.audit import append_audit_entry, format_history, read_audit_log
from core.cache import MigrationCache
from core.logging import get_logger, setup_logging
from core.cuda_c_migrator import migrate_cuda_c_file, migrate_inline_cuda_c, check_hipify_available, run_hipify
from core.migrator import migrate, AppliedChange
from core.quality import build_quality_report, generate_review_checklist

app = typer.Typer(
    name="rocm-migrate",
    help="Migrate CUDA Python/PyTorch code to AMD ROCm.",
    add_completion=True,
)
console = Console()
logger = get_logger(__name__)


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
    quiet: bool = typer.Option(
        False, "--quiet", "-q", help="Suppress all output except errors"
    ),
    log_level: Optional[str] = typer.Option(
        None, "--log-level", help="Log level: DEBUG | INFO | WARNING | ERROR"
    ),
    log_file: Optional[str] = typer.Option(
        None, "--log-file", help="Path to a log file (enables file logging)"
    ),
    planner_timeout: Optional[int] = typer.Option(
        None, "--planner-timeout", help="Planner LLM timeout in seconds"
    ),
    test_timeout: Optional[int] = typer.Option(
        None, "--test-timeout", help="Validation test timeout in seconds"
    ),
    no_cache: bool = typer.Option(
        False, "--no-cache", help="Skip cache, force fresh migration"
    ),
    clear_cache: bool = typer.Option(
        False, "--clear-cache", help="Clear all cached data and exit"
    ),
    show_history: Optional[str] = typer.Option(
        None, "--show-history", help="Show migration history for a file (or all if empty string)"
    ),
    rocm_version: Optional[str] = typer.Option(
        None, "--rocm-version", help="Target ROCm version (e.g. 6.0) — filters out unsupported mappings"
    ),
    use_hipify: bool = typer.Option(
        False, "--use-hipify", help="Use hipify-perl as first pass for .cu/.cuh files (if available)"
    ),
    validate_on_gpu: bool = typer.Option(
        False, "--validate-on-gpu", help="Run migrated code on real ROCm GPU for validation (requires AMD hardware)"
    ),
    interactive: bool = typer.Option(
        False, "--interactive", "-i", help="Review each proposed change interactively (accept/reject/edit)"
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Show migration summary table without writing files (enhanced --diff-only)"
    ),
    watch: bool = typer.Option(
        False, "--watch", "-w", help="Watch input files for changes and re-migrate automatically"
    ),
    include_pattern: Optional[str] = typer.Option(
        None, "--include", help="Include only files matching glob pattern (e.g. '*.py')"
    ),
    exclude_pattern: Optional[str] = typer.Option(
        None, "--exclude", help="Exclude files matching glob pattern (e.g. 'test_*')"
    ),
    output_format: str = typer.Option(
        "diff", "--format", help="Output format: diff | json | markdown | patch"
    ),
    chunk_size: Optional[int] = typer.Option(
        None, "--chunk-size", help="Max tokens per chunk for large-file migration (default: 4000)"
    ),
) -> None:
    """Migrate CUDA Python code to ROCm."""
    # Build settings
    settings = Settings()
    settings.default_backend = backend
    if planner_url is not None:
        settings.planner_base_url = planner_url
    settings.max_rounds = max_rounds
    if planner_timeout is not None:
        settings.planner_timeout = planner_timeout
    if test_timeout is not None:
        settings.test_timeout = test_timeout
    if rocm_version is not None:
        # Validate version string format (e.g. "5.7", "6.0", "6.2.1")
        import re as _re
        if not _re.match(r"^\d+\.\d+(\.\d+)?$", rocm_version):
            console.print(
                f"[red]Error:[/red] Invalid --rocm-version '{rocm_version}'. "
                f"Expected format: MAJOR.MINOR (e.g. '6.0', '5.7.1')"
            )
            raise typer.Exit(1)
        settings.rocm_version = rocm_version

    # Resolve effective log level: --verbose → DEBUG, --quiet → WARNING, --log-level overrides all
    if log_level:
        effective_level = log_level.upper()
    elif verbose:
        effective_level = "DEBUG"
    elif quiet:
        effective_level = "ERROR"
    else:
        effective_level = settings.log_level
    settings.log_level = effective_level
    if log_file:
        settings.log_file = log_file

    setup_logging(level=effective_level, log_file=settings.log_file or None)
    logger.info("CLI started (backend=%s, log_level=%s)", backend, effective_level)

    # dry-run implies diff-only
    if dry_run:
        diff_only = True

    # Validate output format
    valid_formats = ("diff", "json", "markdown", "patch")
    if output_format not in valid_formats:
        console.print(f"[red]Error:[/red] Invalid format '{output_format}'. Choose from: {', '.join(valid_formats)}")
        raise typer.Exit(1)

    # --- Early-exit commands ---

    if clear_cache:
        cache = MigrationCache(settings.cache_dir)
        removed = cache.clear()
        console.print(f"[green]Cache cleared:[/green] {removed} file(s) removed.")
        raise typer.Exit(0)

    if show_history is not None:
        file_filter = show_history if show_history else None
        entries = read_audit_log(file_filter=file_filter)
        console.print(format_history(entries))
        raise typer.Exit(0)

    # --- Cache setup ---

    cache = None if no_cache else MigrationCache(settings.cache_dir)

    # --- Input validation ---
    input_p = Path(input_path)
    if not input_p.exists():
        console.print(f"[red]Error:[/red] Path not found: {input_path}")
        raise typer.Exit(1)

    _SUPPORTED_EXTS = {".py", ".cu", ".cuh"}
    if input_p.is_file():
        if input_p.suffix not in _SUPPORTED_EXTS:
            console.print(f"[red]Error:[/red] Expected a .py/.cu/.cuh file, got: {input_p.suffix}")
            raise typer.Exit(1)
        # Quick check if .py file is valid Python
        if input_p.suffix == ".py":
            source_text = input_p.read_text(encoding="utf-8")
            try:
                ast.parse(source_text)
            except SyntaxError as exc:
                console.print(
                    f"[yellow]Warning:[/yellow] {input_p.name} has a syntax error "
                    f"(line {exc.lineno}): {exc.msg}. Analysis will use regex-only mode."
                )
                logger.warning("Input file %s has syntax error at line %s: %s", input_p, exc.lineno, exc.msg)

    output_p = Path(output)
    if not diff_only:
        try:
            output_p.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            console.print(f"[red]Error:[/red] Cannot create output directory {output}: {exc}")
            raise typer.Exit(1)

    # --- Validate API key is set for selected backend ---
    if not no_agent:
        _check_backend_key(settings, backend)
        _check_backend_health(settings, backend)

    # Collect files — support glob patterns in input_path
    try:
        if any(c in input_path for c in ("*", "?", "[")):
            # Glob pattern provided
            matched = sorted(Path(p) for p in globmod.glob(input_path, recursive=True))
            files = [f for f in matched if f.is_file() and f.suffix in {".py", ".cu", ".cuh"}]
            if not files:
                console.print(f"[red]Error:[/red] No supported files matched pattern: {input_path}")
                raise typer.Exit(1)
        else:
            files = collect_python_files(input_path)
    except (FileNotFoundError, ValueError) as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)

    # Apply --include / --exclude filters
    if include_pattern:
        files = [f for f in files if f.match(include_pattern)]
    if exclude_pattern:
        files = [f for f in files if not f.match(exclude_pattern)]

    if not files:
        console.print("[red]Error:[/red] No files remaining after --include/--exclude filters.")
        raise typer.Exit(1)

    console.print(
        Panel(f"ROCm Migration Agent — {len(files)} file(s) to process",
              style="bold blue")
    )

    # Project-level analysis for multi-file runs
    if len(files) > 1 and input_p.is_dir():
        from core.project_migrator import ProjectAnalyzer
        project_analyzer = ProjectAnalyzer(input_p)
        py_files = [f for f in files if f.suffix == ".py"]
        if py_files:
            project_report = project_analyzer.analyze(py_files)
            if project_report.import_graph:
                cuda_mods = project_report.import_graph.cuda_modules()
                if cuda_mods:
                    console.print(
                        f"  [cyan]Import graph: {len(project_report.import_graph.modules)} modules, "
                        f"{len(cuda_mods)} with CUDA symbols[/cyan]"
                    )
                    order = project_report.import_graph.migration_order()
                    # Reorder files by dependency order
                    path_order = {}
                    for idx, mod in enumerate(order):
                        p = project_report.import_graph.path_for(mod)
                        if p:
                            path_order[str(p)] = idx
                    files = sorted(files, key=lambda f: path_order.get(str(f), 999))

    # Check hipify availability upfront
    if use_hipify:
        hipify_path = check_hipify_available()
        if hipify_path:
            console.print(f"  [green]hipify-perl found: {hipify_path}[/green]")
        else:
            console.print("  [yellow]Warning: --use-hipify set but hipify-perl not found on PATH[/yellow]")
            use_hipify = False

    def _run_migration_pass() -> None:
        """Run one migration pass over all files."""
        from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, MofNCompleteColumn

        skipped = 0
        processed = 0
        all_results: list[dict] = []  # for dry-run / json output

        use_progress = len(files) > 1 and not verbose

        with Progress(
            SpinnerColumn(),
            TextColumn("[bold]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            console=console,
            disable=not use_progress,
        ) as progress:
            task_id = progress.add_task("Migrating files...", total=len(files))

            for file_path in files:
                progress.update(task_id, description=f"[cyan]{file_path.name}[/cyan]")

                if file_path.suffix in (".cu", ".cuh"):
                    was_skipped = _process_cu_file(
                        file_path=file_path,
                        settings=settings,
                        output_dir=Path(output),
                        diff_only=diff_only,
                        verbose=verbose,
                        use_hipify=use_hipify,
                        cache=cache,
                        output_format=output_format,
                    )
                else:
                    was_skipped = _process_file(
                        file_path=file_path,
                        settings=settings,
                        output_dir=Path(output),
                        diff_only=diff_only,
                        no_agent=no_agent,
                        force_agents=force_agents,
                        no_test=no_test,
                        verbose=verbose,
                        cache=cache,
                        validate_on_gpu=validate_on_gpu,
                        interactive=interactive,
                        output_format=output_format,
                        dry_run=dry_run,
                        all_results=all_results,
                        chunk_size=chunk_size,
                    )
                if was_skipped:
                    skipped += 1
                else:
                    processed += 1
                progress.advance(task_id)

        # Final summary
        if skipped or processed:
            console.print(
                f"\n[bold]Summary:[/bold] {processed} file(s) processed"
                + (f", {skipped} unchanged (cached)" if skipped else "")
            )

        # Dry-run summary table
        if dry_run and all_results:
            _print_dry_run_summary(all_results)

        # JSON output mode: dump all results
        if output_format == "json" and all_results:
            console.print(json.dumps(all_results, indent=2))

    # --- Watch mode ---
    if watch:
        console.print("[cyan]Watch mode enabled — monitoring files for changes (Ctrl+C to stop)...[/cyan]")
        try:
            from watchdog.observers import Observer
            from watchdog.events import FileSystemEventHandler

            class _MigrationHandler(FileSystemEventHandler):
                def __init__(self):
                    self._last_trigger = 0.0

                def on_modified(self, event):
                    if event.is_directory:
                        return
                    p = Path(event.src_path)
                    if p.suffix not in {".py", ".cu", ".cuh"}:
                        return
                    if p not in files:
                        return
                    # Debounce: ignore events within 2 seconds of last trigger
                    now = time.time()
                    if now - self._last_trigger < 2.0:
                        return
                    self._last_trigger = now
                    console.print(f"\n[cyan]Change detected: {p.name} — re-migrating...[/cyan]")
                    # Only re-process the changed file, not all files
                    if p.suffix in (".cu", ".cuh"):
                        _process_cu_file(
                            file_path=p, settings=settings,
                            output_dir=Path(output), diff_only=diff_only,
                            verbose=verbose, use_hipify=use_hipify,
                            cache=cache, output_format=output_format,
                        )
                    else:
                        _process_file(
                            file_path=p, settings=settings,
                            output_dir=Path(output), diff_only=diff_only,
                            no_agent=no_agent, force_agents=force_agents,
                            no_test=no_test, verbose=verbose, cache=cache,
                            validate_on_gpu=validate_on_gpu,
                            interactive=False,  # no interactive in watch mode
                            output_format=output_format,
                            dry_run=False, chunk_size=chunk_size,
                        )

            observer = Observer()
            handler = _MigrationHandler()
            watch_paths = set()
            for f in files:
                watch_paths.add(str(f.parent.resolve()))
            for wp in watch_paths:
                observer.schedule(handler, wp, recursive=False)

            # Run once immediately
            _run_migration_pass()
            observer.start()
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                observer.stop()
            observer.join()
        except ImportError:
            console.print("[red]Error:[/red] watchdog package required for --watch mode. Install with: pip install watchdog")
            raise typer.Exit(1)
    else:
        _run_migration_pass()


def _process_file(
    file_path: Path,
    settings: Settings,
    output_dir: Path,
    diff_only: bool,
    no_agent: bool,
    force_agents: bool,
    no_test: bool,
    verbose: bool,
    cache: "MigrationCache | None" = None,
    validate_on_gpu: bool = False,
    interactive: bool = False,
    output_format: str = "diff",
    dry_run: bool = False,
    all_results: list | None = None,
    chunk_size: int | None = None,
) -> bool:
    """Process a single file through the migration pipeline.

    Returns True if the file was skipped (unchanged in cache), False otherwise.
    """
    # 1. Read source
    source = read_source_file(file_path)

    # Incremental skip: if source is unchanged since last cached migration, skip
    if cache and cache.is_unchanged(str(file_path), source, settings.default_backend):
        console.print(f"  [dim]{file_path.name} — unchanged (cached), skipping.[/dim]")
        logger.info("Skipping unchanged file: %s", file_path)
        return True

    console.rule(f"[bold]{file_path.name}[/bold]")
    console.print(f"  Read {len(source.splitlines())} lines")

    # 2. Analyze
    console.print("  [cyan]Analyzing CUDA usage...[/cyan]")
    report = analyze_source(source, str(file_path))

    if not report.has_cuda:
        console.print("  [green]No CUDA-specific code detected. Skipping.[/green]")
        return False

    # Show analysis summary
    _print_analysis_summary(report)

    # 3. Rule-based migration
    console.print("  [cyan]Applying rule-based migrations...[/cyan]")
    result = migrate(source, report, rocm_version=settings.rocm_version)

    # 3b. Inline CUDA C migration (if SourceModule or similar detected)
    has_inline_cuda = any(u.category == "inline_cuda_c" for u in report.usages)
    if has_inline_cuda:
        console.print("  [cyan]Migrating inline CUDA C code...[/cyan]")
        inline_result = migrate_inline_cuda_c(result.code, rocm_version=settings.rocm_version)
        if inline_result.applied:
            result.code = inline_result.code
            console.print(f"  Applied [green]{len(inline_result.applied)}[/green] inline CUDA C changes")

    if verbose:
        for change in result.applied:
            console.print(f"    [green]Line {change.line}:[/green] {change.rule}")

    console.print(
        f"  Applied [green]{len(result.applied)}[/green] automatic changes, "
        f"[yellow]{len(result.remaining)}[/yellow] remaining for LLM"
    )

    # 3c. Interactive review of changes
    if interactive and result.applied:
        result = _interactive_review(source, result)

    # 4. LLM agent loop (optional)
    final_code = result.code
    if not no_agent and (result.remaining or force_agents):
        console.print(f"  [cyan]Running agent loop (backend={settings.default_backend})...[/cyan]")
        try:
            from agents.orchestrator import run_migration_agents
            def _on_chunk_done(name, idx, total):
                console.print(f"    [dim]Chunk {idx}/{total} complete: {name}[/dim]")

            final_code = run_migration_agents(
                original_code=source,
                migration_result=result,
                report=report,
                settings=settings,
                verbose=verbose,
                console=console,
                cache=cache,
                chunk_size=chunk_size,
                on_chunk_complete=_on_chunk_done,
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
    validation_passed = False
    if not no_test:
        console.print("  [cyan]Validating migrated code...[/cyan]")
        from agents.tester import run_validation
        validation = run_validation(
            final_code,
            original_code=source,
            run_sandbox=not no_agent,  # only sandbox if agents were available
            test_timeout=settings.test_timeout,
        )
        if "ALL_TESTS_PASSED" in validation:
            console.print("  [green]All validation checks passed.[/green]")
            validation_passed = True
        else:
            console.print(f"  [yellow]Validation issues:[/yellow]")
        # Show structured output with color coding
        for line in validation.splitlines():
            stripped = line.strip()
            if not stripped or stripped in ("ALL_TESTS_PASSED", "VALIDATION FAILED"):
                continue
            if stripped.startswith("[ERROR]"):
                console.print(f"    [red]{stripped}[/red]")
            elif stripped.startswith("[WARNING]"):
                console.print(f"    [yellow]{stripped}[/yellow]")
            elif stripped.startswith("[INFO]"):
                console.print(f"    [dim]{stripped}[/dim]")
            elif stripped.startswith("[FAIL]"):
                console.print(f"    [red]{stripped}[/red]")
            elif stripped.startswith("[PASS]"):
                console.print(f"    [green]{stripped}[/green]")
            elif stripped.startswith("→") or stripped.startswith("- "):
                console.print(f"      [dim]{stripped}[/dim]")

    # 5a. GPU validation (optional, requires AMD hardware)
    if not no_test and validate_on_gpu:
        console.print("  [cyan]Validating on ROCm GPU...[/cyan]")
        from testing.runner import execute_on_rocm
        gpu_result = execute_on_rocm(final_code, timeout=settings.test_timeout)
        if gpu_result.status == "PASS":
            console.print("  [green]GPU validation passed.[/green]")
        elif gpu_result.status == "WARN":
            console.print(f"  [yellow]GPU validation passed with warnings[/yellow]")
        elif gpu_result.status == "ERROR":
            console.print(f"  [red]GPU validation error: {gpu_result.error_summary}[/red]")
        else:
            console.print(f"  [red]GPU validation failed: {gpu_result.error_summary}[/red]")
            if gpu_result.stderr:
                for err_line in gpu_result.stderr.strip().splitlines()[-3:]:
                    console.print(f"    [red]{err_line.strip()}[/red]")

    # 5b. Build and display quality report
    agent_used = not no_agent and (bool(result.remaining) or force_agents)
    quality_report = build_quality_report(
        file_path=str(file_path),
        applied=result.applied,
        remaining=result.remaining,
        agent_used=agent_used and final_code != result.code,
        validation_passed=validation_passed,
    )
    _print_quality_report(quality_report)

    # 5c. Write review checklist if there are items needing review
    if not diff_only and (quality_report.needs_review or quality_report.low_confidence):
        checklist_path = output_dir / (file_path.stem + "_REVIEW_CHECKLIST.md")
        checklist = generate_review_checklist(quality_report)
        checklist_path.parent.mkdir(parents=True, exist_ok=True)
        checklist_path.write_text(checklist, encoding="utf-8")
        console.print(f"  [yellow]Review checklist written to {checklist_path}[/yellow]")

    # 6. Show diff / formatted output
    diff = generate_diff(source, final_code, file_path.name)
    if diff:
        _render_output(diff, source, final_code, file_path, result, output_format, quality_report)
    else:
        console.print("  [dim]No changes made.[/dim]")

    # Collect results for dry-run summary
    if dry_run and all_results is not None:
        conf_dist = {"high": 0, "medium": 0, "low": 0}
        for c in result.applied:
            if c.confidence >= 0.9:
                conf_dist["high"] += 1
            elif c.confidence >= 0.5:
                conf_dist["medium"] += 1
            else:
                conf_dist["low"] += 1
        all_results.append({
            "file": str(file_path),
            "changes": len(result.applied),
            "remaining": len(result.remaining),
            "confidence": conf_dist,
            "overall_score": quality_report.overall_score,
            "complexity": "high" if len(result.remaining) > 5 else "medium" if len(result.remaining) > 0 else "low",
        })

    # 7. Show optimization suggestions
    if result.optimizations:
        _print_optimizations(result.optimizations)

    # 8. Write output
    if not diff_only and diff:
        out_path = output_dir / file_path.name
        write_output_file(out_path, final_code)
        console.print(f"\n  [green]Written to {out_path}[/green]")

    # 9. Cache the migration result
    if cache and diff:
        applied_dicts = [
            {"line": c.line, "rule": c.rule, "confidence": getattr(c, "confidence", 1.0)}
            for c in result.applied
        ]
        remaining_dicts = [
            {"line": r.line, "symbol": r.symbol, "reason": r.reason}
            for r in result.remaining
        ]
        cache.put_migration(
            file_path=str(file_path),
            source=source,
            backend=settings.default_backend,
            migrated_code=final_code,
            applied=applied_dicts,
            remaining=remaining_dicts,
        )

    # 10. Audit log
    append_audit_entry(
        file_path=str(file_path),
        backend=settings.default_backend,
        applied=result.applied,
        remaining=result.remaining,
        optimizations=result.optimizations,
        overall_confidence=quality_report.overall_score,
        validation_passed=validation_passed if not no_test else None,
        agent_used=agent_used and final_code != result.code,
    )

    return False


def _process_cu_file(
    file_path: Path,
    settings: Settings,
    output_dir: Path,
    diff_only: bool,
    verbose: bool,
    use_hipify: bool = False,
    cache: "MigrationCache | None" = None,
    output_format: str = "diff",
) -> bool:
    """Process a .cu or .cuh file through the CUDA C migrator.

    Returns True if skipped (cached), False otherwise.
    """
    source = read_source_file(file_path)

    # Incremental skip
    if cache and cache.is_unchanged(str(file_path), source, "cuda_c"):
        console.print(f"  [dim]{file_path.name} — unchanged (cached), skipping.[/dim]")
        return True

    console.rule(f"[bold]{file_path.name}[/bold] (CUDA C/C++)")
    console.print(f"  Read {len(source.splitlines())} lines")

    # Optional HIPIFY first pass
    if use_hipify:
        console.print("  [cyan]Running hipify-perl...[/cyan]")
        hipified, success = run_hipify(str(file_path))
        if success:
            console.print("  [green]hipify-perl completed successfully[/green]")
            source = hipified  # use hipified source as starting point
        else:
            console.print(f"  [yellow]hipify-perl failed: {hipified}[/yellow]")
            console.print("  [yellow]Continuing with rule-based migration...[/yellow]")

    # Run CUDA C migrator
    console.print("  [cyan]Applying CUDA C → HIP migrations...[/cyan]")
    result = migrate_cuda_c_file(source, rocm_version=settings.rocm_version)

    if verbose:
        for change in result.applied:
            console.print(f"    [green]Line {change.line}:[/green] {change.rule}")

    console.print(
        f"  Applied [green]{len(result.applied)}[/green] changes, "
        f"[yellow]{len(result.remaining)}[/yellow] need manual review"
    )

    # Show warnings
    for warning in result.warnings:
        console.print(f"  [yellow]Note:[/yellow] {warning}")

    # LLM agent refinement for remaining issues in .cu files
    final_code = result.code
    if result.remaining and not diff_only:
        console.print(f"  [cyan]Running LLM agent for {len(result.remaining)} unresolved issues...[/cyan]")
        try:
            from agents.orchestrator import run_migration_agents
            from core.analyzer import AnalysisReport, CudaUsage
            from core.migrator import MigrationResult as MigResult

            # Build a minimal analysis report for the .cu content
            cu_report = AnalysisReport(file_path=str(file_path))
            for issue in result.remaining:
                cu_report.add(CudaUsage(
                    line=issue.line, col=0, symbol=issue.symbol,
                    category="api_call", context=issue.reason,
                ))
            cu_migration = MigResult(
                code=result.code,
                applied=result.applied,
                remaining=result.remaining,
                warnings=result.warnings,
            )

            agent_code = run_migration_agents(
                original_code=source,
                migration_result=cu_migration,
                report=cu_report,
                settings=settings,
                verbose=verbose,
                console=console,
                cache=cache,
            )
            if agent_code:
                final_code = agent_code
                console.print("  [green]Agent refinement complete.[/green]")
            else:
                console.print("  [yellow]Agent returned no code, using rule-based result.[/yellow]")
        except Exception as e:
            console.print(f"  [red]Agent error: {e}[/red]")
            console.print("  [yellow]Falling back to rule-based result.[/yellow]")

    # Show diff
    diff = generate_diff(source, final_code, file_path.name)
    if diff:
        _render_output(diff, source, final_code, file_path, result, output_format, None)
    else:
        console.print("  [dim]No changes made.[/dim]")

    # Show remaining issues
    if result.remaining:
        table = Table(title="Manual Review Required", show_lines=False)
        table.add_column("Line", style="cyan", justify="right")
        table.add_column("Symbol", style="bold")
        table.add_column("Reason", style="yellow")
        for issue in result.remaining:
            table.add_row(str(issue.line), issue.symbol, issue.reason)
        console.print()
        console.print(table)

    # Write output
    if not diff_only and diff:
        # .cu → .hip.cu or .cuh → .hip.cuh
        out_name = file_path.stem + ".hip" + file_path.suffix
        out_path = output_dir / out_name
        write_output_file(out_path, final_code)
        console.print(f"\n  [green]Written to {out_path}[/green]")

    # Cache
    if cache and diff:
        applied_dicts = [
            {"line": c.line, "rule": c.rule, "confidence": c.confidence}
            for c in result.applied
        ]
        remaining_dicts = [
            {"line": r.line, "symbol": r.symbol, "reason": r.reason}
            for r in result.remaining
        ]
        cache.put_migration(
            file_path=str(file_path),
            source=source,
            backend="cuda_c",
            migrated_code=result.code,
            applied=applied_dicts,
            remaining=remaining_dicts,
        )

    # Audit log
    overall_confidence = (
        sum(c.confidence for c in result.applied) / len(result.applied)
        if result.applied else 0.0
    )
    append_audit_entry(
        file_path=str(file_path),
        backend="cuda_c",
        applied=result.applied,
        remaining=result.remaining,
        optimizations=[],
        overall_confidence=overall_confidence,
    )

    return False


def _interactive_review(source: str, result) -> "MigrationResult":
    """Interactively review each proposed change — accept, reject, or edit."""
    from core.migrator import MigrationResult

    source_lines = source.splitlines()
    accepted: list[AppliedChange] = []
    rejected: list[AppliedChange] = []
    accept_all = False

    console.print("\n[bold]Interactive Review[/bold] — reviewing each change:")
    console.print("  [dim]Options: [y]es  [n]o  [e]dit  [a]ccept all  [q]uit[/dim]\n")

    for change in result.applied:
        if accept_all:
            accepted.append(change)
            continue

        # Show context (3 lines before/after)
        line_idx = change.line - 1  # 0-based
        start = max(0, line_idx - 3)
        end = min(len(source_lines), line_idx + 4)
        context_lines = []
        for i in range(start, end):
            marker = ">>>" if i == line_idx else "   "
            context_lines.append(f"{marker} {i + 1:4d} | {source_lines[i]}")

        console.print(Panel(
            "\n".join(context_lines),
            title=f"Change at line {change.line}",
            subtitle=f"[{change.confidence:.0%} confidence] {change.rule}",
        ))
        console.print(f"  [red]- {change.original.strip()}[/red]")
        console.print(f"  [green]+ {change.replacement.strip()}[/green]")

        choice = Prompt.ask(
            "  Accept this change?",
            choices=["y", "n", "e", "a", "q"],
            default="y",
        )

        if choice == "y":
            accepted.append(change)
        elif choice == "n":
            rejected.append(change)
            console.print("  [dim]Rejected.[/dim]")
        elif choice == "e":
            # Let the user type the replacement line
            edited = Prompt.ask("  Enter replacement line", default=change.replacement.strip())
            edited_change = AppliedChange(
                line=change.line,
                original=change.original,
                replacement=edited,
                rule=f"{change.rule} (user-edited)",
                confidence=1.0,
            )
            accepted.append(edited_change)
            console.print(f"  [green]Edited: {edited}[/green]")
        elif choice == "a":
            accept_all = True
            accepted.append(change)
            console.print("  [green]Accepting all remaining changes.[/green]")
        elif choice == "q":
            console.print("  [yellow]Quit — keeping only accepted changes so far.[/yellow]")
            break

    console.print(
        f"\n  Accepted [green]{len(accepted)}[/green], "
        f"rejected [red]{len(rejected)}[/red] changes."
    )

    # Rebuild the code with only accepted changes
    if len(accepted) == len(result.applied):
        return result  # All accepted, no rebuild needed

    # Re-apply only accepted changes from original source
    from core.migrator import migrate as _migrate
    from core.analyzer import analyze_source as _analyze
    # Simplest approach: rebuild from migrated code but revert rejected lines
    migrated_lines = result.code.splitlines(keepends=True)
    original_lines = source.splitlines(keepends=True)
    for change in rejected:
        idx = change.line - 1
        if idx < len(original_lines) and idx < len(migrated_lines):
            migrated_lines[idx] = original_lines[idx]

    new_result = MigrationResult(
        code="".join(migrated_lines),
        applied=accepted,
        remaining=result.remaining,
        warnings=result.warnings,
        optimizations=result.optimizations,
    )
    return new_result


def _render_output(diff: str, source: str, final_code: str, file_path: Path,
                   result, output_format: str, quality_report) -> None:
    """Render migration output in the requested format."""
    if output_format == "diff":
        console.print()
        console.print(Panel("Diff", style="bold"))
        console.print(Syntax(diff, "diff", theme="monokai"))

    elif output_format == "json":
        # JSON output is accumulated and printed at the end; show nothing here per-file
        pass

    elif output_format == "markdown":
        md_lines = [
            f"# Migration Report: {file_path.name}",
            "",
            f"**Overall Confidence:** {quality_report.overall_score:.0%}" if quality_report else "",
            f"**Changes Applied:** {len(result.applied)}",
            f"**Remaining Issues:** {len(result.remaining)}",
            "",
            "## Changes",
            "",
        ]
        for c in result.applied:
            md_lines.append(f"- **Line {c.line}** ({c.confidence:.0%}): {c.rule}")
        if result.remaining:
            md_lines.append("")
            md_lines.append("## Remaining Issues")
            md_lines.append("")
            for r in result.remaining:
                md_lines.append(f"- **Line {r.line}** `{r.symbol}`: {r.reason}")
        if result.optimizations:
            md_lines.append("")
            md_lines.append("## Optimization Suggestions")
            md_lines.append("")
            for opt in result.optimizations:
                md_lines.append(f"- **P{opt.priority} {opt.category}**: {opt.suggestion}")
        md_lines.append("")
        md_lines.append("## Diff")
        md_lines.append("")
        md_lines.append("```diff")
        md_lines.append(diff)
        md_lines.append("```")
        console.print("\n".join(md_lines))

    elif output_format == "patch":
        # Standard patch format — just print raw diff
        console.print(diff)


def _print_dry_run_summary(all_results: list[dict]) -> None:
    """Print a summary table for dry-run mode."""
    console.print()
    table = Table(title="Dry-Run Migration Summary", show_lines=True)
    table.add_column("File", style="cyan")
    table.add_column("Changes", justify="right", style="green")
    table.add_column("Remaining", justify="right", style="yellow")
    table.add_column("High Conf.", justify="right", style="green")
    table.add_column("Med. Conf.", justify="right", style="yellow")
    table.add_column("Low Conf.", justify="right", style="red")
    table.add_column("Score", justify="right", style="bold")
    table.add_column("Complexity", style="dim")

    total_changes = 0
    total_remaining = 0
    total_high = 0
    total_need_review = 0
    total_manual = 0

    for r in all_results:
        score = r["overall_score"]
        score_style = "green" if score >= 0.9 else "yellow" if score >= 0.7 else "red"
        table.add_row(
            Path(r["file"]).name,
            str(r["changes"]),
            str(r["remaining"]),
            str(r["confidence"]["high"]),
            str(r["confidence"]["medium"]),
            str(r["confidence"]["low"]),
            f"[{score_style}]{score:.0%}[/{score_style}]",
            r["complexity"],
        )
        total_changes += r["changes"]
        total_remaining += r["remaining"]
        total_high += r["confidence"]["high"]
        total_need_review += r["confidence"]["medium"]
        total_manual += r["confidence"]["low"]

    console.print(table)
    console.print(
        f"\n  [bold]Total:[/bold] {total_high} high-confidence, "
        f"{total_need_review} need review, {total_manual} need manual work, "
        f"{total_remaining} unresolved"
    )


_BACKEND_KEY_MAP = {
    "mistral": "mistral_api_key",
    "deepseek": "deepseek_api_key",
    "claude": "anthropic_api_key",
}


def _check_backend_key(settings: Settings, backend: str) -> None:
    """Warn if the API key for the selected backend is not set."""
    key_attr = _BACKEND_KEY_MAP.get(backend)
    if key_attr and not getattr(settings, key_attr, ""):
        console.print(
            f"[yellow]Warning:[/yellow] No API key set for backend '{backend}' "
            f"(env var {key_attr.upper()} is empty). LLM agent calls will likely fail."
        )
        logger.warning("API key for backend '%s' is not set (%s)", backend, key_attr.upper())


def _check_backend_health(settings: Settings, backend: str) -> None:
    """Verify that the selected LLM backend is reachable."""
    import urllib.request
    import urllib.error

    _BACKEND_URLS = {
        "self-hosted": settings.planner_base_url,
        "mistral": "https://api.mistral.ai/v1",
        "deepseek": "https://api.deepseek.com/v1",
        "claude": "https://api.anthropic.com/v1",
    }

    base_url = _BACKEND_URLS.get(backend)
    if not base_url:
        return

    models_url = f"{base_url.rstrip('/')}/models"
    try:
        req = urllib.request.Request(models_url, method="GET")
        # Add auth headers for cloud backends
        key_attr = _BACKEND_KEY_MAP.get(backend)
        api_key = getattr(settings, key_attr, "") if key_attr else ""
        if api_key:
            if backend == "claude":
                req.add_header("x-api-key", api_key)
                req.add_header("anthropic-version", "2023-06-01")
            else:
                req.add_header("Authorization", f"Bearer {api_key}")

        urllib.request.urlopen(req, timeout=10)
        logger.info("Backend '%s' health check passed (%s)", backend, models_url)
    except urllib.error.URLError as exc:
        console.print(
            f"[yellow]Warning:[/yellow] Cannot reach backend '{backend}' "
            f"at {models_url}: {exc.reason}. "
            f"LLM agent calls may fail."
        )
        logger.warning("Backend health check failed for '%s': %s", backend, exc)
    except Exception as exc:
        console.print(
            f"[yellow]Warning:[/yellow] Backend health check failed for '{backend}': {exc}"
        )
        logger.warning("Backend health check error for '%s': %s", backend, exc)


def _print_quality_report(quality_report) -> None:
    """Print the migration quality summary as a color-coded table."""
    from core.quality import MigrationQualityReport

    s = quality_report.summary
    score = quality_report.overall_score

    # Overall score color
    if score >= 0.9:
        score_style = "bold green"
    elif score >= 0.7:
        score_style = "bold yellow"
    else:
        score_style = "bold red"

    table = Table(title="Migration Quality", show_lines=False)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right", style="bold")

    table.add_row("Overall confidence", f"[{score_style}]{score:.0%}[/{score_style}]")
    table.add_row("High confidence (>= 90%)", f"[green]{s['high']}[/green]")
    table.add_row("Needs review (50-89%)", f"[yellow]{s['needs_review']}[/yellow]")
    table.add_row("Low confidence (< 50%)", f"[red]{s['low']}[/red]")
    table.add_row("Total changes", str(s["total"]))

    console.print()
    console.print(table)

    # Show individual low-confidence items inline
    for item in quality_report.low_confidence:
        console.print(
            f"    [red]!! Line {item.line}[/red] ({item.confidence:.0%}): "
            f"{item.rule}"
        )
    for item in quality_report.needs_review:
        console.print(
            f"    [yellow]?  Line {item.line}[/yellow] ({item.confidence:.0%}): "
            f"{item.rule}"
        )


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
