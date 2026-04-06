"""Tests for core/analyzer.py — CUDA usage detection."""

from pathlib import Path

import pytest

from core.analyzer import AnalysisReport, analyze_source

FIXTURES = Path(__file__).parent / "fixtures"


class TestAnalyzerSimple:
    @pytest.fixture()
    def report(self) -> AnalysisReport:
        source = (FIXTURES / "sample_cuda_simple.py").read_text()
        return analyze_source(source, "sample_cuda_simple.py")

    def test_detects_cuda_usages(self, report: AnalysisReport):
        assert report.has_cuda
        assert report.total > 0

    def test_detects_env_var(self, report: AnalysisReport):
        env_usages = [u for u in report.usages if u.category == "env_var"]
        symbols = [u.symbol for u in env_usages]
        assert "CUDA_VISIBLE_DEVICES" in symbols

    def test_detects_cudnn_backend(self, report: AnalysisReport):
        backend_usages = [u for u in report.usages if u.category == "backend"]
        assert len(backend_usages) >= 2  # benchmark + deterministic
        symbols = [u.symbol for u in backend_usages]
        assert any("cudnn" in s for s in symbols)

    def test_has_context_lines(self, report: AnalysisReport):
        for usage in report.usages:
            assert usage.context, f"Usage at line {usage.line} has no context"


class TestAnalyzerMultiGpu:
    @pytest.fixture()
    def report(self) -> AnalysisReport:
        source = (FIXTURES / "sample_cuda_multi_gpu.py").read_text()
        return analyze_source(source, "sample_cuda_multi_gpu.py")

    def test_detects_multiple_env_vars(self, report: AnalysisReport):
        env_usages = [u for u in report.usages if u.category == "env_var"]
        symbols = {u.symbol for u in env_usages}
        assert "CUDA_VISIBLE_DEVICES" in symbols
        assert "CUDA_LAUNCH_BLOCKING" in symbols

    def test_detects_cudnn_patterns(self, report: AnalysisReport):
        backend_usages = [u for u in report.usages if u.category == "backend"]
        assert len(backend_usages) >= 2  # benchmark + enabled


class TestAnalyzerPycuda:
    @pytest.fixture()
    def report(self) -> AnalysisReport:
        source = (FIXTURES / "sample_cuda_pycuda.py").read_text()
        return analyze_source(source, "sample_cuda_pycuda.py")

    def test_detects_pycuda_import(self, report: AnalysisReport):
        import_usages = [u for u in report.usages if u.category == "import"]
        assert len(import_usages) >= 1
        symbols = [u.symbol for u in import_usages]
        assert any("pycuda" in s for s in symbols)


class TestAnalyzerEdgeCases:
    def test_empty_source(self):
        report = analyze_source("", "<empty>")
        assert not report.has_cuda
        assert report.total == 0

    def test_no_cuda(self):
        source = "import numpy as np\nx = np.array([1, 2, 3])\n"
        report = analyze_source(source, "<clean>")
        assert not report.has_cuda

    def test_syntax_error_falls_to_regex(self):
        source = "this is not valid python {{{"
        # Should not raise, falls through to regex pass
        report = analyze_source(source, "<bad>")
        assert isinstance(report, AnalysisReport)

    def test_kernel_launch_in_string(self):
        source = 'code = "kernel<<<grid, block>>>(args)"\n'
        report = analyze_source(source, "<kernel>")
        api_usages = [u for u in report.usages if u.symbol == "<<<...>>>"]
        assert len(api_usages) == 1
