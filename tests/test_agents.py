"""Tests for agent modules — prompts, message formatting, code extraction, validation."""

import pytest

from agents.coder import CODER_SYSTEM_PROMPT, format_coder_message
from agents.reviewer import REVIEWER_SYSTEM_PROMPT
from agents.orchestrator import _extract_code_block, _extract_final_code
from agents.tester import run_validation


# --- Coder ---


class TestCoderPrompt:
    def test_prompt_mentions_rocm_device_string(self):
        assert '"cuda"' in CODER_SYSTEM_PROMPT
        assert "rocm" in CODER_SYSTEM_PROMPT.lower() or "ROCm" in CODER_SYSTEM_PROMPT

    def test_prompt_mentions_key_mappings(self):
        assert "MIOpen" in CODER_SYSTEM_PROMPT
        assert "HIP_VISIBLE_DEVICES" in CODER_SYSTEM_PROMPT

    def test_format_message_includes_code(self):
        msg = format_coder_message(
            original_code="import torch\nx = torch.randn(10).cuda()",
            pre_migrated_code="import torch\nx = torch.randn(10).cuda()",
            remaining_issues=[
                {"line": 2, "symbol": ".cuda()", "reason": "needs review"},
            ],
        )
        assert "Original CUDA Code" in msg
        assert "Partially Migrated Code" in msg
        assert "Remaining Issues" in msg
        assert "Line 2" in msg
        assert ".cuda()" in msg

    def test_format_message_no_remaining(self):
        msg = format_coder_message(
            original_code="x = 1",
            pre_migrated_code="x = 1",
            remaining_issues=[],
        )
        assert "none" in msg.lower() or "handled everything" in msg.lower()


# --- Reviewer ---


class TestReviewerPrompt:
    def test_prompt_mentions_device_string_correct(self):
        # Reviewer should know "cuda" is correct on ROCm
        assert "cuda" in REVIEWER_SYSTEM_PROMPT
        assert "CORRECT" in REVIEWER_SYSTEM_PROMPT or "correct" in REVIEWER_SYSTEM_PROMPT

    def test_prompt_mentions_approved(self):
        assert "APPROVED" in REVIEWER_SYSTEM_PROMPT

    def test_prompt_mentions_issues_found(self):
        assert "ISSUES FOUND" in REVIEWER_SYSTEM_PROMPT


# --- Code extraction ---


class TestCodeExtraction:
    def test_extract_python_block(self):
        text = "Here:\n```python\nimport torch\n```\n"
        assert _extract_code_block(text) == "import torch"

    def test_extract_plain_block(self):
        text = "Here:\n```\nimport torch\nx = 1\n```\n"
        assert _extract_code_block(text) == "import torch\nx = 1"

    def test_extract_last_block(self):
        text = "```python\nfirst\n```\n\n```python\nsecond\n```\n"
        assert _extract_code_block(text) == "second"

    def test_extract_none_when_no_block(self):
        assert _extract_code_block("no code here") is None

    def test_extract_final_code_from_messages(self):
        messages = [
            {"name": "Coder", "content": "```python\nv1\n```"},
            {"name": "Reviewer", "content": "ISSUES FOUND\n1. Fix line 3"},
            {"name": "Coder", "content": "```python\nv2_final\n```"},
            {"name": "Tester", "content": "ALL_TESTS_PASSED"},
        ]
        assert _extract_final_code(messages) == "v2_final"

    def test_extract_final_code_empty(self):
        assert _extract_final_code([]) == ""
        assert _extract_final_code([{"content": "no code"}]) == ""


# --- Tester / Validation ---


class TestTesterValidation:
    def test_clean_code_passes(self):
        code = "import torch\nx = torch.randn(10).cuda()\n"
        result = run_validation(code)
        assert "ALL_TESTS_PASSED" in result

    def test_cudnn_ref_fails(self):
        code = "torch.backends.cudnn.benchmark = True\n"
        result = run_validation(code)
        assert "VALIDATION FAILED" in result or "FAIL" in result

    def test_cuda_env_var_fails(self):
        code = 'import os\nos.environ["CUDA_VISIBLE_DEVICES"] = "0"\n'
        result = run_validation(code)
        assert "FAIL" in result

    def test_wrong_device_string_fails(self):
        code = 'device = torch.device("rocm")\n'
        result = run_validation(code)
        assert "FAIL" in result

    def test_hip_device_string_fails(self):
        code = 'device = torch.device("hip")\n'
        result = run_validation(code)
        assert "FAIL" in result

    def test_cuda_device_string_passes(self):
        # "cuda" is correct on ROCm
        code = 'import torch\ndevice = torch.device("cuda")\n'
        result = run_validation(code)
        assert "ALL_TESTS_PASSED" in result

    def test_pycuda_import_fails(self):
        code = "import pycuda.autoinit\n"
        result = run_validation(code)
        assert "FAIL" in result


# --- Executor config ---


class TestExecutorConfig:
    def test_custom_executor_key_takes_precedence(self):
        from config.settings import Settings
        from config.model_profiles import get_executor_config

        s = Settings()
        s.executor_api_key = "custom-key-123"
        s.mistral_api_key = "mistral-key-456"
        config = get_executor_config(s)
        assert config["config_list"][0]["api_key"] == "custom-key-123"

    def test_falls_back_to_mistral_key(self):
        from config.settings import Settings
        from config.model_profiles import get_executor_config

        s = Settings()
        s.executor_api_key = ""
        s.mistral_api_key = "mistral-key-456"
        config = get_executor_config(s)
        assert config["config_list"][0]["api_key"] == "mistral-key-456"

    def test_falls_back_to_empty_key(self):
        from config.settings import Settings
        from config.model_profiles import get_executor_config

        s = Settings()
        s.executor_api_key = ""
        s.mistral_api_key = ""
        config = get_executor_config(s)
        assert config["config_list"][0]["api_key"] == "EMPTY"

    def test_custom_executor_url_and_model(self):
        from config.settings import Settings
        from config.model_profiles import get_executor_config

        s = Settings()
        s.executor_base_url = "http://10.128.0.2:8002/v1"
        s.executor_model = "google/gemma-4-31B-it"
        s.executor_api_key = "EMPTY"
        config = get_executor_config(s)
        assert config["config_list"][0]["model"] == "google/gemma-4-31B-it"
        assert config["config_list"][0]["base_url"] == "http://10.128.0.2:8002/v1"

    def test_context_budget_respects_executor_override(self):
        from config.settings import Settings
        from core.context_budget import ContextBudget

        s = Settings()
        s.executor_context_limit = 50000
        budget = ContextBudget.for_backend("mistral", settings=s)
        assert budget.max_tokens == 50000

    def test_context_budget_fallback_for_unknown_backend(self):
        from core.context_budget import ContextBudget

        budget = ContextBudget.for_backend("some-custom-backend")
        assert budget.max_tokens == 12000  # safe fallback
