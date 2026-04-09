"""Project-level migration — dependency-ordered multi-file migration.

Uses the import graph to determine migration order and passes cross-file
context (migrated symbols, renamed imports) between files.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from core.analyzer import AnalysisReport, analyze_source
from core.file_io import read_source_file
from core.import_graph import ImportGraph, build_import_graph
from core.logging import get_logger
from core.migrator import MigrationResult, migrate

logger = get_logger(__name__)


@dataclass
class ProjectAnalysisReport:
    """Aggregated analysis across an entire project."""

    root: str
    file_reports: dict[str, AnalysisReport] = field(default_factory=dict)
    import_graph: ImportGraph | None = None
    exported_cuda_symbols: dict[str, list[str]] = field(default_factory=dict)

    @property
    def total_usages(self) -> int:
        return sum(r.total for r in self.file_reports.values())

    @property
    def files_with_cuda(self) -> list[str]:
        return [f for f, r in self.file_reports.items() if r.has_cuda]

    @property
    def total_files(self) -> int:
        return len(self.file_reports)


@dataclass
class FileMigrationContext:
    """Cross-file context passed to each file's migration."""

    migrated_symbols: dict[str, str] = field(default_factory=dict)
    renamed_imports: dict[str, str] = field(default_factory=dict)
    already_migrated: list[str] = field(default_factory=list)


@dataclass
class ProjectMigrationResult:
    """Result of migrating an entire project."""

    file_results: dict[str, MigrationResult] = field(default_factory=dict)
    migration_order: list[str] = field(default_factory=list)
    cross_file_context: FileMigrationContext = field(default_factory=FileMigrationContext)

    @property
    def total_applied(self) -> int:
        return sum(len(r.applied) for r in self.file_results.values())

    @property
    def total_remaining(self) -> int:
        return sum(len(r.remaining) for r in self.file_results.values())


class ProjectAnalyzer:
    """Analyzes all files in a project for CUDA usage."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def analyze(self, files: list[Path]) -> ProjectAnalysisReport:
        """Run analysis on all files and build the project report.

        Args:
            files: List of Python files to analyze.

        Returns:
            A ProjectAnalysisReport with per-file reports and import graph.
        """
        report = ProjectAnalysisReport(root=str(self.root))

        # Build import graph
        report.import_graph = build_import_graph(self.root)

        # Analyze each file
        for file_path in files:
            try:
                source = read_source_file(file_path)
                file_report = analyze_source(source, str(file_path))
                report.file_reports[str(file_path)] = file_report

                # Track exported CUDA symbols
                if file_report.has_cuda:
                    symbols = [u.symbol for u in file_report.usages]
                    report.exported_cuda_symbols[str(file_path)] = symbols

            except Exception as exc:
                logger.warning("Failed to analyze %s: %s", file_path, exc)

        logger.info(
            "Project analysis: %d files, %d with CUDA, %d total usages",
            report.total_files,
            len(report.files_with_cuda),
            report.total_usages,
        )

        return report


class ProjectMigrator:
    """Migrates a project file-by-file in dependency order."""

    def __init__(
        self,
        project_report: ProjectAnalysisReport,
        rocm_version: str = "",
    ) -> None:
        self.report = project_report
        self.rocm_version = rocm_version
        self.context = FileMigrationContext()

    def migrate_all(
        self,
        files: list[Path],
        on_file_start: Any | None = None,
        on_file_complete: Any | None = None,
    ) -> ProjectMigrationResult:
        """Migrate all files in dependency order.

        Args:
            files: List of files to migrate.
            on_file_start: Optional callback(file_path, index, total).
            on_file_complete: Optional callback(file_path, result, index, total).

        Returns:
            ProjectMigrationResult with per-file results.
        """
        result = ProjectMigrationResult(cross_file_context=self.context)

        # Determine migration order from import graph
        if self.report.import_graph:
            ordered_modules = self.report.import_graph.migration_order()
            # Map modules back to file paths
            ordered_paths: list[Path] = []
            path_set: set[str] = set()
            for mod in ordered_modules:
                mod_path = self.report.import_graph.path_for(mod)
                if mod_path and str(mod_path) not in path_set:
                    ordered_paths.append(mod_path)
                    path_set.add(str(mod_path))

            # Add any files not in the graph (e.g. standalone scripts)
            for f in files:
                if str(f) not in path_set:
                    ordered_paths.append(f)
                    path_set.add(str(f))

            migration_files = ordered_paths
        else:
            migration_files = files

        result.migration_order = [str(f) for f in migration_files]
        total = len(migration_files)

        for idx, file_path in enumerate(migration_files):
            if on_file_start:
                on_file_start(file_path, idx, total)

            file_result = self._migrate_file(file_path)
            result.file_results[str(file_path)] = file_result

            # Update cross-file context with migrated symbols
            self._update_context(file_path, file_result)
            self.context.already_migrated.append(str(file_path))

            if on_file_complete:
                on_file_complete(file_path, file_result, idx, total)

        return result

    def _migrate_file(self, file_path: Path) -> MigrationResult:
        """Migrate a single file using the project context."""
        file_key = str(file_path)
        file_report = self.report.file_reports.get(file_key)

        if not file_report:
            # Not analyzed — read and analyze now
            try:
                source = read_source_file(file_path)
                file_report = analyze_source(source, file_key)
            except Exception as exc:
                logger.error("Cannot read %s: %s", file_path, exc)
                return MigrationResult(code="")

        if not file_report.has_cuda:
            source = read_source_file(file_path)
            return MigrationResult(code=source)

        source = read_source_file(file_path)
        result = migrate(source, file_report, rocm_version=self.rocm_version)
        return result

    def _update_context(self, file_path: Path, result: MigrationResult) -> None:
        """Update cross-file context with symbols migrated in this file."""
        for change in result.applied:
            if change.original != change.replacement:
                self.context.migrated_symbols[change.original.strip()] = change.replacement.strip()
                # Track renamed imports for dependent files
                if "import" in change.rule.lower():
                    self.context.renamed_imports[change.original.strip()] = change.replacement.strip()
