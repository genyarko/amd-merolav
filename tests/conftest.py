"""Shared pytest fixtures for the CUDA-to-ROCm migration test suite."""

from __future__ import annotations

from pathlib import Path

import pytest

from core.analyzer import AnalysisReport, analyze_source
from core.migrator import MigrationResult, migrate

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture()
def fixtures_dir() -> Path:
    return FIXTURES_DIR


def load_fixture(name: str) -> str:
    """Read a fixture file by name."""
    return (FIXTURES_DIR / name).read_text(encoding="utf-8")


def analyze_fixture(name: str) -> AnalysisReport:
    """Run the analyzer on a fixture file."""
    source = load_fixture(name)
    return analyze_source(source, name)


def migrate_fixture(name: str) -> MigrationResult:
    """Run the full analyze → migrate pipeline on a fixture file."""
    source = load_fixture(name)
    report = analyze_source(source, name)
    return migrate(source, report)


def analyze_and_migrate(source: str, filename: str = "<test>") -> MigrationResult:
    """Run the full analyze → migrate pipeline on inline source code."""
    report = analyze_source(source, filename)
    return migrate(source, report)
