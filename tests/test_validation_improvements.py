"""Tests for Phase 13 — Validation Improvements.

Covers: new validators, severity levels, structured reports, equivalence checker,
GPU validation (mocked), and updated run_validation output.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from testing.validators import (
    Severity,
    ValidationIssue,
    ValidationReport,
    check_deprecated_rocm_apis,
    check_device_strings,
    check_env_vars,
    check_imports,
    check_incompatible_libraries,
    check_mixed_imports,
    check_no_cudnn_refs,
    check_orphaned_env_vars,
)
from testing.equivalence import EquivalenceResult, check_equivalence
from testing.runner import detect_rocm_runtime, execute_on_rocm, ExecutionResult
from agents.tester import run_validation


# =====================================================================
# Structured ValidationReport
# =====================================================================

class TestValidationReport:
    def test_add_issue_error_sets_passed_false(self):
        r = ValidationReport(name="test", passed=True)
        r.add_issue(1, "bad thing", Severity.ERROR)
        assert r.passed is False
        assert len(r.structured_issues) == 1
        assert len(r.issues) == 1

    def test_add_issue_warning_keeps_passed(self):
        r = ValidationReport(name="test", passed=True)
        r.add_issue(1, "mild thing", Severity.WARNING)
        assert r.passed is True
        assert len(r.structured_issues) == 1

    def test_add_issue_info_keeps_passed(self):
        r = ValidationReport(name="test", passed=True)
        r.add_issue(5, "note", Severity.INFO)
        assert r.passed is True

    def test_structured_issue_has_suggestion(self):
        r = ValidationReport(name="test", passed=True)
        r.add_issue(1, "msg", Severity.WARNING, suggestion="do this instead")
        assert r.structured_issues[0].suggestion == "do this instead"


# =====================================================================
# Existing validators — now with severity
# =====================================================================

class TestExistingValidatorsWithSeverity:
    def test_cudnn_ref_is_error(self):
        code = "torch.backends.cudnn.benchmark = True\n"
        r = check_no_cudnn_refs(code)
        assert not r.passed
        assert r.structured_issues[0].severity == Severity.ERROR

    def test_cuda_import_is_error(self):
        code = "import pycuda\n"
        r = check_imports(code)
        assert not r.passed
        assert r.structured_issues[0].severity == Severity.ERROR

    def test_wrong_device_is_error(self):
        code = 'x = torch.device("rocm")\n'
        r = check_device_strings(code)
        assert not r.passed
        assert r.structured_issues[0].severity == Severity.ERROR
        assert r.structured_issues[0].suggestion  # has fix suggestion

    def test_cuda_env_var_is_error(self):
        code = 'os.environ["CUDA_VISIBLE_DEVICES"] = "0"\n'
        r = check_env_vars(code)
        assert not r.passed
        assert r.structured_issues[0].severity == Severity.ERROR
        assert "HIP_VISIBLE_DEVICES" in r.structured_issues[0].suggestion

    def test_clean_code_passes_all(self):
        code = (
            "import torch\n"
            'device = torch.device("cuda")\n'
            'os.environ["HIP_VISIBLE_DEVICES"] = "0"\n'
        )
        for checker in [check_no_cudnn_refs, check_imports, check_device_strings, check_env_vars]:
            r = checker(code)
            assert r.passed


# =====================================================================
# New validators
# =====================================================================

class TestMixedImports:
    def test_no_mixed(self):
        code = "import migraphx\nimport torch\n"
        r = check_mixed_imports(code)
        assert r.passed

    def test_mixed_cuda_and_rocm(self):
        code = "import pycuda\nimport hip\n"
        r = check_mixed_imports(code)
        assert len(r.structured_issues) == 1
        assert r.structured_issues[0].severity == Severity.WARNING
        assert "Mixed" in r.structured_issues[0].message

    def test_cuda_only_no_flag(self):
        code = "import pycuda\nimport torch\n"
        r = check_mixed_imports(code)
        assert r.passed  # no ROCm imports to conflict with

    def test_rocm_only_no_flag(self):
        code = "import hip\nimport migraphx\n"
        r = check_mixed_imports(code)
        assert r.passed


class TestIncompatibleLibraries:
    def test_no_conflict(self):
        code = "import migraphx\nimport torch\n"
        r = check_incompatible_libraries(code)
        assert r.passed

    def test_tensorrt_and_migraphx(self):
        code = "import tensorrt\nimport migraphx\n"
        r = check_incompatible_libraries(code)
        assert len(r.structured_issues) == 1
        assert r.structured_issues[0].severity == Severity.WARNING

    def test_pycuda_and_hip(self):
        code = "import pycuda\nimport hip\n"
        r = check_incompatible_libraries(code)
        assert len(r.structured_issues) == 1


class TestDeprecatedAPIs:
    def test_no_deprecated(self):
        code = "hipDeviceSynchronize()\nhipSetDevice(0)\n"
        r = check_deprecated_rocm_apis(code)
        assert r.passed

    def test_hipCtxCreate_deprecated(self):
        code = "hipCtxCreate(&ctx, 0, device)\n"
        r = check_deprecated_rocm_apis(code)
        assert len(r.structured_issues) == 1
        assert r.structured_issues[0].severity == Severity.WARNING
        assert "deprecated" in r.structured_issues[0].message

    def test_hipCtxSynchronize_deprecated(self):
        code = "hipCtxSynchronize()\n"
        r = check_deprecated_rocm_apis(code)
        assert len(r.structured_issues) >= 1

    def test_hipMemcpyHtoD_flagged(self):
        code = "hipMemcpyHtoD(d_ptr, h_ptr, size)\n"
        r = check_deprecated_rocm_apis(code)
        assert len(r.structured_issues) == 1

    def test_comment_skipped(self):
        code = "# hipCtxCreate is deprecated\n"
        r = check_deprecated_rocm_apis(code)
        assert r.passed

    def test_multiple_deprecated(self):
        code = "hipCtxCreate(&ctx, 0, dev)\nhipCtxDestroy(ctx)\nhipCtxSynchronize()\n"
        r = check_deprecated_rocm_apis(code)
        assert len(r.structured_issues) == 3


class TestOrphanedEnvVars:
    def test_no_env_vars(self):
        code = "x = 1\n"
        r = check_orphaned_env_vars(code)
        assert r.passed

    def test_used_var_not_flagged(self):
        code = (
            'os.environ["HIP_VISIBLE_DEVICES"] = "0"\n'
            'print(os.environ["HIP_VISIBLE_DEVICES"])\n'
        )
        r = check_orphaned_env_vars(code)
        assert r.passed

    def test_orphaned_var_flagged(self):
        code = 'os.environ["HIP_VISIBLE_DEVICES"] = "0"\nx = 1\n'
        r = check_orphaned_env_vars(code)
        assert len(r.structured_issues) == 1
        assert r.structured_issues[0].severity == Severity.INFO

    def test_non_hip_var_not_tracked(self):
        code = 'os.environ["MY_CUSTOM_VAR"] = "1"\n'
        r = check_orphaned_env_vars(code)
        assert r.passed  # only tracks HIP-related vars


# =====================================================================
# run_validation — structured output
# =====================================================================

class TestRunValidation:
    def test_clean_code_passes(self):
        code = (
            "import torch\n"
            'device = torch.device("cuda")\n'
            "x = torch.randn(10).to(device)\n"
        )
        result = run_validation(code)
        assert "ALL_TESTS_PASSED" in result

    def test_failed_shows_error_severity(self):
        code = "import pycuda\ntorch.backends.cudnn.benchmark = True\n"
        result = run_validation(code)
        assert "VALIDATION FAILED" in result
        assert "[ERROR]" in result

    def test_deprecated_api_shows_warning(self):
        code = (
            "import torch\n"
            "hipCtxCreate(&ctx, 0, dev)\n"
        )
        result = run_validation(code)
        # Code passes (deprecated is WARNING, not ERROR)
        # but should contain warning info
        assert "[WARNING]" in result

    def test_suggestions_in_output(self):
        code = 'os.environ["CUDA_VISIBLE_DEVICES"] = "0"\n'
        result = run_validation(code)
        assert "HIP_VISIBLE_DEVICES" in result  # suggestion

    def test_mixed_imports_warning(self):
        code = "import pycuda\nimport hip\n"
        result = run_validation(code)
        assert "Mixed" in result or "[WARNING]" in result

    def test_validator_summary_on_failure(self):
        code = "import pycuda\n"
        result = run_validation(code)
        assert "Validator Summary" in result


# =====================================================================
# Equivalence checker
# =====================================================================

class TestEquivalenceChecker:
    def test_identical_code_passes(self):
        code = "import torch\nx = torch.tensor([1.0, 2.0, 3.0])\ny = x * 2\n"
        result = check_equivalence(code, code)
        assert result.passed

    def test_different_values_fails(self):
        original = "import torch\nx = torch.tensor([1.0, 2.0, 3.0])\n"
        migrated = "import torch\nx = torch.tensor([9.0, 9.0, 9.0])\n"
        result = check_equivalence(original, migrated)
        assert not result.passed
        assert len(result.issues) > 0

    def test_shape_mismatch_fails(self):
        original = "import torch\nx = torch.zeros(3)\n"
        migrated = "import torch\nx = torch.zeros(5)\n"
        result = check_equivalence(original, migrated)
        assert not result.passed
        assert any("shape" in iss.expected for iss in result.issues)

    def test_syntax_error_reports_failure(self):
        result = check_equivalence("x = 1", "x = !!!")
        assert not result.passed
        assert result.error

    def test_no_tensors_passes(self):
        code = "x = 42\ny = 'hello'\n"
        result = check_equivalence(code, code)
        assert result.passed


# =====================================================================
# GPU validation (mocked — no real hardware needed)
# =====================================================================

class TestGPUValidation:
    def test_detect_rocm_returns_bool(self):
        # On a machine without ROCm, should return False
        result = detect_rocm_runtime()
        assert isinstance(result, bool)

    @patch("testing.runner.detect_rocm_runtime", return_value=False)
    def test_execute_on_rocm_no_runtime(self, mock_detect):
        result = execute_on_rocm("print('hello')")
        assert result.status == "ERROR"
        assert "No ROCm runtime" in result.error_summary

    @patch("testing.runner.detect_rocm_runtime", return_value=True)
    @patch("subprocess.run")
    def test_execute_on_rocm_success(self, mock_run, mock_detect):
        mock_run.return_value = type("Proc", (), {
            "returncode": 0,
            "stdout": "output\n__GPU_EXECUTION_COMPLETE__\n",
            "stderr": "",
        })()
        result = execute_on_rocm("print('hello')")
        assert result.status == "PASS"

    @patch("testing.runner.detect_rocm_runtime", return_value=True)
    @patch("subprocess.run")
    def test_execute_on_rocm_failure(self, mock_run, mock_detect):
        mock_run.return_value = type("Proc", (), {
            "returncode": 1,
            "stdout": "",
            "stderr": "RuntimeError: HIP error\n",
        })()
        result = execute_on_rocm("bad code")
        assert result.status == "FAIL"
        assert "HIP error" in result.error_summary

    @patch("testing.runner.detect_rocm_runtime", return_value=True)
    @patch("subprocess.run", side_effect=__import__("subprocess").TimeoutExpired(cmd="", timeout=60))
    def test_execute_on_rocm_timeout(self, mock_run, mock_detect):
        result = execute_on_rocm("import time; time.sleep(999)")
        assert result.status == "ERROR"
        assert "timed out" in result.error_summary
