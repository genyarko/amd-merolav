"""Tests for core/migrator.py — rule-based migration."""

from pathlib import Path

import pytest

from core.analyzer import analyze_source
from core.migrator import MigrationResult, migrate

FIXTURES = Path(__file__).parent / "fixtures"


def _migrate_fixture(name: str) -> MigrationResult:
    source = (FIXTURES / name).read_text()
    report = analyze_source(source, name)
    return migrate(source, report)


class TestMigratorSimple:
    @pytest.fixture()
    def result(self) -> MigrationResult:
        return _migrate_fixture("sample_cuda_simple.py")

    def test_replaces_env_var(self, result: MigrationResult):
        assert "HIP_VISIBLE_DEVICES" in result.code
        assert "CUDA_VISIBLE_DEVICES" not in result.code

    def test_replaces_cudnn_benchmark(self, result: MigrationResult):
        assert "torch.backends.cudnn.benchmark" not in result.code
        # Should have MIOpen comment or replacement
        assert "MIOpen" in result.code or "miopen" in result.code

    def test_replaces_cudnn_deterministic(self, result: MigrationResult):
        assert "miopen.deterministic" in result.code

    def test_has_applied_changes(self, result: MigrationResult):
        assert len(result.applied) >= 3  # env_var + benchmark + deterministic

    def test_preserves_torch_cuda_device(self, result: MigrationResult):
        # torch.device("cuda") is correct on ROCm — should NOT be changed
        assert 'torch.device("cuda"' in result.code

    def test_preserves_cuda_is_available(self, result: MigrationResult):
        assert "torch.cuda.is_available()" in result.code


class TestMigratorMultiGpu:
    @pytest.fixture()
    def result(self) -> MigrationResult:
        return _migrate_fixture("sample_cuda_multi_gpu.py")

    def test_replaces_both_env_vars(self, result: MigrationResult):
        assert "CUDA_VISIBLE_DEVICES" not in result.code
        assert "CUDA_LAUNCH_BLOCKING" not in result.code
        assert "HIP_VISIBLE_DEVICES" in result.code
        assert "HIP_LAUNCH_BLOCKING" in result.code

    def test_replaces_cudnn_enabled(self, result: MigrationResult):
        assert "miopen.enabled" in result.code

    def test_finds_optimization_suggestions(self, result: MigrationResult):
        triggers = [o.trigger for o in result.optimizations]
        assert "DataParallel" in triggers or "torch.cuda.amp" in triggers


class TestMigratorPycuda:
    @pytest.fixture()
    def result(self) -> MigrationResult:
        return _migrate_fixture("sample_cuda_pycuda.py")

    def test_replaces_pycuda_import(self, result: MigrationResult):
        # pycuda imports should be replaced
        applied_rules = [a.rule for a in result.applied]
        assert any("Import" in r for r in applied_rules)

    def test_code_has_hip_reference(self, result: MigrationResult):
        assert "hip" in result.code.lower()


class TestMigratorEdgeCases:
    def test_no_cuda_code_unchanged(self):
        source = "import numpy as np\nx = np.array([1, 2, 3])\n"
        report = analyze_source(source, "<clean>")
        result = migrate(source, report)
        assert result.code == source
        assert len(result.applied) == 0
        assert len(result.remaining) == 0

    def test_applied_changes_have_details(self):
        result = _migrate_fixture("sample_cuda_simple.py")
        for change in result.applied:
            assert change.line > 0
            assert change.original
            assert change.replacement
            assert change.rule
