"""Property-based tests for core/migrator.py using Hypothesis.

These tests verify invariants that should hold for ANY valid Python input,
not just specific fixtures.
"""

from __future__ import annotations

import ast
import textwrap

import pytest
from hypothesis import given, settings, assume, HealthCheck
from hypothesis import strategies as st

from core.analyzer import analyze_source
from core.migrator import migrate


# --- Strategies ---

# Generate syntactically valid Python snippets containing CUDA patterns
_CUDA_IMPORT_LINES = [
    "import pycuda.autoinit",
    "import pycuda.driver as cuda",
    "import cupy as cp",
    "import tensorrt as trt",
]

_CUDA_ENV_LINES = [
    'os.environ["CUDA_VISIBLE_DEVICES"] = "0"',
    'os.environ["CUDA_LAUNCH_BLOCKING"] = "1"',
]

_CUDA_BACKEND_LINES = [
    "torch.backends.cudnn.benchmark = True",
    "torch.backends.cudnn.deterministic = True",
    "torch.backends.cudnn.enabled = True",
]

_CLEAN_LINES = [
    "import os",
    "import torch",
    "import numpy as np",
    "x = 1",
    "y = torch.randn(10)",
    'device = torch.device("cuda")',
    "model = torch.nn.Linear(10, 2)",
    "print('hello')",
    "pass",
]


@st.composite
def cuda_python_source(draw):
    """Generate a valid Python source string that may contain CUDA patterns."""
    # Pick some clean lines
    n_clean = draw(st.integers(min_value=1, max_value=5))
    lines = [draw(st.sampled_from(_CLEAN_LINES)) for _ in range(n_clean)]

    # Optionally add CUDA patterns
    if draw(st.booleans()):
        lines.insert(0, draw(st.sampled_from(_CUDA_IMPORT_LINES)))
    if draw(st.booleans()):
        lines.append(draw(st.sampled_from(_CUDA_ENV_LINES)))
    if draw(st.booleans()):
        lines.append(draw(st.sampled_from(_CUDA_BACKEND_LINES)))

    source = "\n".join(lines) + "\n"

    # Verify it's valid Python before returning
    try:
        ast.parse(source)
    except SyntaxError:
        assume(False)

    return source


@st.composite
def clean_python_source(draw):
    """Generate valid Python source with no CUDA patterns."""
    n_lines = draw(st.integers(min_value=1, max_value=8))
    lines = [draw(st.sampled_from(_CLEAN_LINES)) for _ in range(n_lines)]
    source = "\n".join(lines) + "\n"
    try:
        ast.parse(source)
    except SyntaxError:
        assume(False)
    return source


# --- Property tests ---


class TestMigratorProperties:

    @given(source=cuda_python_source())
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_migrated_code_is_valid_python(self, source: str):
        """Migrated code should always be parseable Python."""
        report = analyze_source(source, "<prop>")
        result = migrate(source, report)
        try:
            ast.parse(result.code)
        except SyntaxError as exc:
            pytest.fail(
                f"Migrated code has syntax error at line {exc.lineno}: {exc.msg}\n"
                f"Original:\n{source}\n"
                f"Migrated:\n{result.code}"
            )

    @given(source=cuda_python_source())
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_migration_is_idempotent(self, source: str):
        """Migrating already-migrated code should produce no further changes."""
        # First migration pass
        report1 = analyze_source(source, "<idem1>")
        result1 = migrate(source, report1)

        # Second migration pass on the result
        report2 = analyze_source(result1.code, "<idem2>")
        result2 = migrate(result1.code, report2)

        assert result2.code == result1.code, (
            f"Migration is not idempotent.\n"
            f"After 1st pass:\n{result1.code}\n"
            f"After 2nd pass:\n{result2.code}"
        )

    @given(source=cuda_python_source())
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_migration_never_increases_cuda_env_vars(self, source: str):
        """Migration should never introduce new CUDA environment variable references."""
        cuda_env_vars = ["CUDA_VISIBLE_DEVICES", "CUDA_LAUNCH_BLOCKING", "CUDA_DEVICE_ORDER"]

        before_count = sum(source.count(v) for v in cuda_env_vars)

        report = analyze_source(source, "<env>")
        result = migrate(source, report)

        after_count = sum(result.code.count(v) for v in cuda_env_vars)
        assert after_count <= before_count, (
            f"CUDA env var count increased from {before_count} to {after_count}"
        )

    @given(source=cuda_python_source())
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_migration_never_increases_cudnn_refs(self, source: str):
        """Migration should never introduce new cudnn references."""
        before_count = source.count("cudnn")
        report = analyze_source(source, "<cudnn>")
        result = migrate(source, report)
        after_count = result.code.count("cudnn")
        assert after_count <= before_count, (
            f"cuDNN reference count increased from {before_count} to {after_count}"
        )

    @given(source=clean_python_source())
    @settings(max_examples=30, suppress_health_check=[HealthCheck.too_slow])
    def test_clean_code_unchanged(self, source: str):
        """Code with no CUDA patterns should pass through unchanged."""
        report = analyze_source(source, "<clean>")
        result = migrate(source, report)
        assert result.code == source
        assert len(result.applied) == 0

    @given(source=cuda_python_source())
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_applied_plus_remaining_covers_all_usages(self, source: str):
        """Every detected CUDA usage should be either applied or flagged as remaining."""
        report = analyze_source(source, "<coverage>")
        result = migrate(source, report)

        applied_lines = {c.line for c in result.applied}
        remaining_lines = {r.line for r in result.remaining}
        covered_lines = applied_lines | remaining_lines

        # Every usage line with an actionable category should be covered
        actionable_categories = {"backend", "import", "env_var", "api_call", "attribute"}
        for usage in report.usages:
            if usage.category in actionable_categories:
                assert usage.line in covered_lines, (
                    f"Usage at line {usage.line} ({usage.symbol}, {usage.category}) "
                    f"not in applied or remaining"
                )

    @given(source=cuda_python_source())
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_torch_cuda_device_preserved(self, source: str):
        """torch.device("cuda") must never be changed — it's correct on ROCm."""
        report = analyze_source(source, "<device>")
        result = migrate(source, report)

        if 'torch.device("cuda")' in source:
            assert 'torch.device("cuda")' in result.code, (
                'torch.device("cuda") was incorrectly modified by the migrator'
            )
