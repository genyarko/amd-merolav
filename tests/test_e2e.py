"""End-to-end tests with mocked LLM responses.

These tests run the full pipeline (analyze → migrate → agents → validate)
with mocked OpenAI-compatible API calls, so no real LLM backend is needed.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from agents.tester import run_validation
from core.analyzer import analyze_source
from core.migrator import migrate

FIXTURES = Path(__file__).parent / "fixtures"


class TestRuleOnlyPipeline:
    """End-to-end tests using --no-agent mode (rule-based migration only)."""

    def test_simple_fixture_full_pipeline(self):
        source = (FIXTURES / "sample_cuda_simple.py").read_text()

        # Step 1: Analyze
        report = analyze_source(source, "sample_cuda_simple.py")
        assert report.has_cuda

        # Step 2: Rule-based migration
        result = migrate(source, report)
        assert len(result.applied) >= 3

        # Step 3: Validate
        validation = run_validation(result.code)
        assert "ALL_TESTS_PASSED" in validation

    def test_multi_gpu_fixture_full_pipeline(self):
        source = (FIXTURES / "sample_cuda_multi_gpu.py").read_text()

        report = analyze_source(source, "sample_cuda_multi_gpu.py")
        assert report.has_cuda

        result = migrate(source, report)
        assert "CUDA_VISIBLE_DEVICES" not in result.code
        assert "CUDA_LAUNCH_BLOCKING" not in result.code
        assert "cudnn" not in result.code.split("#")[0]  # Not in code, maybe in comments

        validation = run_validation(result.code)
        assert "ALL_TESTS_PASSED" in validation

    def test_large_fixture_full_pipeline(self):
        source = (FIXTURES / "sample_cuda_large.py").read_text()

        report = analyze_source(source, "sample_cuda_large.py")
        assert report.has_cuda
        assert report.total >= 5

        result = migrate(source, report)
        assert len(result.applied) >= 5

        validation = run_validation(result.code)
        assert "ALL_TESTS_PASSED" in validation

    def test_partially_migrated_full_pipeline(self):
        source = (FIXTURES / "sample_partially_migrated.py").read_text()

        report = analyze_source(source, "sample_partially_migrated.py")
        result = migrate(source, report)

        # Should fix remaining CUDA refs without breaking already-migrated ones
        assert "HIP_VISIBLE_DEVICES" in result.code
        assert "HIP_LAUNCH_BLOCKING" in result.code
        assert "miopen.deterministic" in result.code

        validation = run_validation(result.code)
        assert "ALL_TESTS_PASSED" in validation

    def test_clean_code_no_changes(self):
        source = "import torch\nimport numpy as np\nx = torch.randn(10)\nprint(x)\n"

        report = analyze_source(source, "clean.py")
        assert not report.has_cuda

        result = migrate(source, report)
        assert result.code == source
        assert len(result.applied) == 0


class TestAgentPipelineWithMockedLLM:
    """Test the agent pipeline with mocked LLM API responses."""

    def _mock_llm_response(self, code: str):
        """Create a mock response object mimicking OpenAI chat completion."""
        mock_message = MagicMock()
        mock_message.content = f"```python\n{code}\n```"

        mock_choice = MagicMock()
        mock_choice.message = mock_message

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        return mock_response

    @pytest.mark.requires_llm
    def test_agent_pipeline_with_mock_planner(self):
        """Test that the planner gracefully handles a mocked empty response."""
        from config.settings import Settings
        from agents.planner import run_planner

        source = (FIXTURES / "sample_cuda_simple.py").read_text()
        report = analyze_source(source, "test.py")
        result = migrate(source, report)

        settings = Settings()
        settings.default_backend = "mistral"

        # Mock the OpenAIWrapper.create to return a planner response
        with patch("agents.planner.OpenAIWrapper") as MockWrapper:
            mock_client = MagicMock()
            mock_client.create.return_value = MagicMock()
            mock_client.extract_text_or_completion_object.return_value = [
                "1. Replace remaining cudnn refs\n2. Check env vars\nPLAN COMPLETE — 2 steps."
            ]
            MockWrapper.return_value = mock_client

            plan = run_planner(source, result, report, settings, verbose=False)
            assert "PLAN COMPLETE" in plan

    @pytest.mark.requires_llm
    def test_planner_retry_on_failure(self):
        """Test that the planner retries on transient errors."""
        from config.settings import Settings
        from agents.planner import run_planner

        source = (FIXTURES / "sample_cuda_simple.py").read_text()
        report = analyze_source(source, "test.py")
        result = migrate(source, report)

        settings = Settings()

        with patch("agents.planner.OpenAIWrapper") as MockWrapper:
            mock_client = MagicMock()
            # First two calls fail, third succeeds
            mock_client.create.side_effect = [
                ConnectionError("timeout"),
                ConnectionError("timeout"),
                MagicMock(),
            ]
            mock_client.extract_text_or_completion_object.return_value = ["Plan here."]
            MockWrapper.return_value = mock_client

            with patch("agents.planner.time.sleep"):  # Don't actually sleep
                plan = run_planner(source, result, report, settings, verbose=False)
            assert plan == "Plan here."

    @pytest.mark.requires_llm
    def test_planner_returns_empty_after_exhausted_retries(self):
        """Test that the planner returns empty string after all retries fail."""
        from config.settings import Settings
        from agents.planner import run_planner

        source = (FIXTURES / "sample_cuda_simple.py").read_text()
        report = analyze_source(source, "test.py")
        result = migrate(source, report)

        settings = Settings()

        with patch("agents.planner.OpenAIWrapper") as MockWrapper:
            mock_client = MagicMock()
            mock_client.create.side_effect = ConnectionError("always fails")
            MockWrapper.return_value = mock_client

            with patch("agents.planner.time.sleep"):
                plan = run_planner(source, result, report, settings, verbose=False)
            assert plan == ""


class TestValidationIntegration:
    """Test the validation step catches real issues in migrated code."""

    def test_incomplete_migration_caught(self):
        """Code with leftover cudnn refs should fail validation."""
        code = (
            "import torch\n"
            "torch.backends.cudnn.benchmark = True\n"
            'device = torch.device("cuda")\n'
        )
        result = run_validation(code)
        assert "FAIL" in result

    def test_wrong_device_string_caught(self):
        """Code with 'rocm' device string should fail validation."""
        code = (
            "import torch\n"
            'device = torch.device("rocm")\n'
        )
        result = run_validation(code)
        assert "FAIL" in result

    def test_fully_migrated_code_passes(self):
        """Properly migrated code should pass all checks."""
        code = (
            "import os\n"
            "import torch\n"
            "import torch.nn as nn\n"
            "\n"
            'os.environ["HIP_VISIBLE_DEVICES"] = "0"\n'
            "torch.backends.miopen.deterministic = True\n"
            "\n"
            'device = torch.device("cuda" if torch.cuda.is_available() else "cpu")\n'
            "model = nn.Linear(10, 2).cuda()\n"
        )
        result = run_validation(code)
        assert "ALL_TESTS_PASSED" in result
