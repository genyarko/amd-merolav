"""Tests for testing/validators.py and testing/runner.py."""

import pytest

from testing.validators import (
    check_device_strings,
    check_env_vars,
    check_imports,
    check_no_cudnn_refs,
)
from testing.runner import execute_migrated_code


# --- Validators ---


class TestCheckNoCudnnRefs:
    def test_clean_code_passes(self):
        code = "import torch\ntorch.backends.miopen.enabled = True\n"
        assert check_no_cudnn_refs(code).passed

    def test_cudnn_in_code_fails(self):
        code = "torch.backends.cudnn.benchmark = True\n"
        r = check_no_cudnn_refs(code)
        assert not r.passed
        assert len(r.issues) >= 1

    def test_cudnn_in_comment_passes(self):
        code = "# cudnn was replaced with miopen\nx = 1\n"
        assert check_no_cudnn_refs(code).passed

    def test_miopen_deterministic_passes(self):
        code = "torch.backends.miopen.deterministic = True\n"
        assert check_no_cudnn_refs(code).passed


class TestCheckImports:
    def test_clean_imports_pass(self):
        code = "import torch\nimport numpy as np\n"
        assert check_imports(code).passed

    def test_pycuda_fails(self):
        code = "import pycuda.autoinit\n"
        r = check_imports(code)
        assert not r.passed

    def test_tensorrt_fails(self):
        code = "import tensorrt as trt\n"
        r = check_imports(code)
        assert not r.passed

    def test_from_pycuda_fails(self):
        code = "from pycuda.driver import mem_alloc\n"
        r = check_imports(code)
        assert not r.passed

    def test_hip_import_passes(self):
        code = "import hip\n"
        assert check_imports(code).passed

    def test_migraphx_passes(self):
        code = "import migraphx\n"
        assert check_imports(code).passed

    def test_syntax_error_fails(self):
        code = "this is not python {{{"
        r = check_imports(code)
        assert not r.passed


class TestCheckDeviceStrings:
    def test_cuda_device_passes(self):
        code = 'device = torch.device("cuda")\n'
        assert check_device_strings(code).passed

    def test_cuda_colon_passes(self):
        code = 'device = torch.device("cuda:0")\n'
        assert check_device_strings(code).passed

    def test_rocm_device_fails(self):
        code = 'device = torch.device("rocm")\n'
        r = check_device_strings(code)
        assert not r.passed

    def test_hip_device_fails(self):
        code = 'device = torch.device("hip")\n'
        r = check_device_strings(code)
        assert not r.passed

    def test_to_rocm_fails(self):
        code = 'x = tensor.to("rocm")\n'
        r = check_device_strings(code)
        assert not r.passed

    def test_commented_out_passes(self):
        code = '# device = torch.device("rocm")\n'
        assert check_device_strings(code).passed


class TestCheckEnvVars:
    def test_hip_vars_pass(self):
        code = 'os.environ["HIP_VISIBLE_DEVICES"] = "0"\n'
        assert check_env_vars(code).passed

    def test_cuda_visible_devices_fails(self):
        code = 'os.environ["CUDA_VISIBLE_DEVICES"] = "0"\n'
        r = check_env_vars(code)
        assert not r.passed

    def test_cuda_launch_blocking_fails(self):
        code = 'os.environ["CUDA_LAUNCH_BLOCKING"] = "1"\n'
        r = check_env_vars(code)
        assert not r.passed

    def test_cuda_var_in_comment_passes(self):
        code = '# was CUDA_VISIBLE_DEVICES, now HIP\nx = 1\n'
        assert check_env_vars(code).passed


# --- Runner ---


class TestRunner:
    def test_simple_code_passes(self):
        code = "x = 1 + 2\nprint(x)\n"
        result = execute_migrated_code(code, timeout=15)
        assert result.status == "PASS"
        assert "3" in result.stdout

    def test_syntax_error_fails(self):
        code = "def foo(\n"
        result = execute_migrated_code(code, timeout=15)
        assert result.status == "FAIL"
        assert "SyntaxError" in result.stderr or "SyntaxError" in result.error_summary

    def test_runtime_error_fails(self):
        code = "raise ValueError('test error')\n"
        result = execute_migrated_code(code, timeout=15)
        assert result.status == "FAIL"
        assert "ValueError" in result.stderr

    def test_mock_hip_import_works(self):
        code = "import hip\nprint('hip imported')\n"
        result = execute_migrated_code(code, timeout=15)
        assert result.status == "PASS"
        assert "hip imported" in result.stdout

    def test_mock_migraphx_import_works(self):
        code = "import migraphx\nprint('migraphx ok')\n"
        result = execute_migrated_code(code, timeout=15)
        assert result.status == "PASS"

    def test_timeout_handled(self):
        code = "import time\ntime.sleep(60)\n"
        result = execute_migrated_code(code, timeout=2)
        assert result.status == "ERROR"
        assert "timed out" in result.error_summary.lower()
