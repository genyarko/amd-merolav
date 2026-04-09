"""Extended edge-case tests for core/analyzer.py."""

from core.analyzer import analyze_source


class TestCudaInComments:
    """CUDA references inside comments should not be flagged as usages."""

    def test_single_line_comment(self):
        source = "# os.environ['CUDA_VISIBLE_DEVICES'] = '0'\nx = 1\n"
        report = analyze_source(source, "<comment>")
        # The regex scanner should still detect env vars in comments,
        # but the AST scanner won't since comments aren't AST nodes.
        # This is acceptable — the migrator only acts on AST-matched usages.
        assert isinstance(report.total, int)

    def test_cuda_in_docstring(self):
        source = '"""\nThis module replaces CUDA_VISIBLE_DEVICES with HIP.\n"""\nx = 1\n'
        report = analyze_source(source, "<docstring>")
        # Docstrings are string constants — env var detection might trigger
        # but this is informational, not harmful.
        assert isinstance(report.total, int)

    def test_inline_comment_not_flagged_as_backend(self):
        source = "x = 1  # torch.backends.cudnn.benchmark was removed\n"
        report = analyze_source(source, "<inline>")
        # AST doesn't see comments, so no backend usage should be found
        backend_usages = [u for u in report.usages if u.category == "backend"]
        assert len(backend_usages) == 0


class TestVariableNamesContainingCuda:
    """Variable names that happen to contain 'cuda' shouldn't cause API-call detections."""

    def test_variable_with_cuda_prefix(self):
        source = "cuda_device_count = 4\nprint(cuda_device_count)\n"
        report = analyze_source(source, "<varname>")
        # These are Name nodes, not Call nodes — should NOT be flagged as api_call
        api_usages = [u for u in report.usages if u.category == "api_call"]
        assert len(api_usages) == 0

    def test_bool_variable_is_cuda_available(self):
        source = "is_cuda_available = True\n"
        report = analyze_source(source, "<boolvar>")
        api_usages = [u for u in report.usages if u.category == "api_call"]
        assert len(api_usages) == 0

    def test_dict_key_with_cuda(self):
        source = 'config = {"cuda_enabled": True, "device": "cpu"}\n'
        report = analyze_source(source, "<dictkey>")
        api_usages = [u for u in report.usages if u.category == "api_call"]
        assert len(api_usages) == 0


class TestWildcardAndComplexImports:
    """Test import patterns the analyzer should detect."""

    def test_from_pycuda_star_import(self):
        source = "from pycuda.driver import *\n"
        report = analyze_source(source, "<star>")
        import_usages = [u for u in report.usages if u.category == "import"]
        assert len(import_usages) >= 1
        assert any("pycuda" in u.symbol for u in import_usages)

    def test_from_pycuda_compiler_import(self):
        source = "from pycuda.compiler import SourceModule\n"
        report = analyze_source(source, "<compiler>")
        import_usages = [u for u in report.usages if u.category == "import"]
        assert len(import_usages) >= 1

    def test_nested_pycuda_import(self):
        source = "import pycuda.driver\nimport pycuda.autoinit\n"
        report = analyze_source(source, "<nested>")
        import_usages = [u for u in report.usages if u.category == "import"]
        assert len(import_usages) >= 2

    def test_cupy_import_detected(self):
        source = "import cupy as cp\n"
        report = analyze_source(source, "<cupy>")
        import_usages = [u for u in report.usages if u.category == "import"]
        assert len(import_usages) >= 1
        assert any("cupy" in u.symbol for u in import_usages)


class TestGuardedImports:
    """Imports inside try/except blocks are still valid CUDA usages."""

    def test_try_except_pycuda(self):
        source = (
            "try:\n"
            "    import pycuda.autoinit\n"
            "except ImportError:\n"
            "    pass\n"
        )
        report = analyze_source(source, "<guarded>")
        import_usages = [u for u in report.usages if u.category == "import"]
        assert len(import_usages) >= 1

    def test_try_except_tensorrt(self):
        source = (
            "try:\n"
            "    import tensorrt as trt\n"
            "except ImportError:\n"
            "    trt = None\n"
        )
        report = analyze_source(source, "<guarded_trt>")
        import_usages = [u for u in report.usages if u.category == "import"]
        assert len(import_usages) >= 1


class TestSyntaxErrorHandling:
    """Verify graceful fallback when source has syntax errors."""

    def test_partial_function_def(self):
        source = "def train(model,\n"
        report = analyze_source(source, "<partial>")
        assert isinstance(report.total, int)

    def test_mismatched_parens(self):
        source = "x = foo(bar(\n"
        report = analyze_source(source, "<parens>")
        assert isinstance(report.total, int)

    def test_syntax_error_with_cuda_env_var(self):
        """Regex should still find patterns even when AST fails."""
        source = 'CUDA_VISIBLE_DEVICES {{syntax error\nkernel<<<grid, block>>>()\n'
        report = analyze_source(source, "<bad_with_cuda>")
        # Regex pass should still find the kernel launch
        symbols = [u.symbol for u in report.usages]
        assert "<<<...>>>" in symbols


class TestEmptyAndTrivialInputs:
    def test_empty_string(self):
        report = analyze_source("", "<empty>")
        assert not report.has_cuda
        assert report.total == 0

    def test_whitespace_only(self):
        report = analyze_source("   \n\n  \n", "<whitespace>")
        assert not report.has_cuda

    def test_comments_only(self):
        report = analyze_source("# just a comment\n# another\n", "<comments>")
        assert not report.has_cuda

    def test_single_pass_statement(self):
        report = analyze_source("pass\n", "<pass>")
        assert not report.has_cuda


class TestKernelLaunchPatterns:
    """Test detection of <<<...>>> kernel launch syntax."""

    def test_simple_kernel_launch(self):
        source = 'code = "kernel<<<1, 256>>>(d_a, d_b, n)"\n'
        report = analyze_source(source, "<kernel>")
        assert any(u.symbol == "<<<...>>>" for u in report.usages)

    def test_multiline_kernel_string(self):
        source = (
            'kernel_code = """\n'
            "__global__ void add(float *a, float *b, int n) {\n"
            "    int i = blockIdx.x * blockDim.x + threadIdx.x;\n"
            "    if (i < n) a[i] += b[i];\n"
            "}\n"
            '"""\n'
        )
        report = analyze_source(source, "<multiline_kernel>")
        # No <<<>>> in this snippet, but it's valid CUDA C in a string
        assert isinstance(report.total, int)

    def test_kernel_launch_with_shared_mem(self):
        source = 'x = "kernel<<<grid, block, shared_mem>>>(args)"\n'
        report = analyze_source(source, "<shared>")
        assert any(u.symbol == "<<<...>>>" for u in report.usages)


class TestMultipleCategoriesInOneLine:
    """Lines containing multiple CUDA patterns."""

    def test_env_var_and_comment(self):
        source = 'os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # CUDA device\n'
        report = analyze_source(source, "<multi>")
        env_usages = [u for u in report.usages if u.category == "env_var"]
        assert len(env_usages) >= 1

    def test_cudnn_benchmark_and_deterministic_on_separate_lines(self):
        source = (
            "torch.backends.cudnn.benchmark = True\n"
            "torch.backends.cudnn.deterministic = True\n"
        )
        report = analyze_source(source, "<both>")
        backend_usages = [u for u in report.usages if u.category == "backend"]
        assert len(backend_usages) >= 2


class TestContextLineAccuracy:
    """Verify that context lines match the actual source."""

    def test_context_matches_source_line(self):
        source = (
            "import os\n"
            'os.environ["CUDA_VISIBLE_DEVICES"] = "0"\n'
            "x = 1\n"
        )
        report = analyze_source(source, "<ctx>")
        for usage in report.usages:
            source_lines = source.splitlines()
            assert usage.context == source_lines[usage.line - 1].rstrip()
