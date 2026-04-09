"""Tests for agent interaction patterns — message flow, termination, edge cases."""

from __future__ import annotations

import pytest

from agents.coder import CODER_SYSTEM_PROMPT, format_coder_message
from agents.reviewer import REVIEWER_SYSTEM_PROMPT
from agents.tester import run_validation
from agents.orchestrator import _extract_code_block, _extract_final_code


class TestCodeBlockExtraction:
    """Test code extraction from various message formats agents might produce."""

    def test_extract_from_python_block(self):
        text = "Here is the migrated code:\n```python\nimport torch\nx = 1\n```\n"
        assert _extract_code_block(text) == "import torch\nx = 1"

    def test_extract_from_plain_block(self):
        text = "```\nimport torch\n```"
        assert _extract_code_block(text) == "import torch"

    def test_extract_from_Python_capitalized(self):
        text = "```Python\nimport torch\n```"
        assert _extract_code_block(text) == "import torch"

    def test_extract_last_of_multiple_blocks(self):
        text = (
            "First attempt:\n```python\nv1\n```\n"
            "Fixed version:\n```python\nv2\n```\n"
        )
        assert _extract_code_block(text) == "v2"

    def test_no_code_block_returns_none(self):
        assert _extract_code_block("APPROVED — code looks good") is None

    def test_empty_code_block(self):
        text = "```python\n\n```"
        result = _extract_code_block(text)
        assert result == "" or result is None

    def test_code_block_with_extra_language_tag(self):
        """Some LLMs add extra info after 'python' like 'python3'."""
        text = "```python3\nimport torch\n```"
        result = _extract_code_block(text)
        assert result == "import torch"

    def test_nested_backticks_in_code(self):
        """Code containing triple backtick strings shouldn't break extraction."""
        text = '```python\nx = """hello"""\ny = 1\n```'
        result = _extract_code_block(text)
        assert "y = 1" in result


class TestFinalCodeExtraction:
    """Test _extract_final_code which searches message history for the last code block."""

    def test_finds_last_coder_message(self):
        messages = [
            {"name": "Executor", "content": "```python\nv1\n```"},
            {"name": "Reviewer", "content": "ISSUES FOUND\n1. Fix line 3"},
            {"name": "Executor", "content": "```python\nv2_final\n```"},
            {"name": "Tester", "content": "ALL_TESTS_PASSED"},
        ]
        assert _extract_final_code(messages) == "v2_final"

    def test_falls_back_to_any_code_block(self):
        """If no Coder/Executor message has code, find any code block."""
        messages = [
            {"name": "Reviewer", "content": "Here's the fix:\n```python\nfixed\n```"},
        ]
        assert _extract_final_code(messages) == "fixed"

    def test_empty_messages_returns_empty(self):
        assert _extract_final_code([]) == ""

    def test_no_code_blocks_returns_empty(self):
        messages = [
            {"name": "Executor", "content": "I'll work on this."},
            {"name": "Reviewer", "content": "APPROVED"},
        ]
        assert _extract_final_code(messages) == ""

    def test_handles_missing_name_field(self):
        messages = [
            {"role": "assistant", "content": "```python\ncode\n```"},
        ]
        assert _extract_final_code(messages) == "code"

    def test_handles_missing_content_field(self):
        messages = [
            {"name": "Executor"},
        ]
        assert _extract_final_code(messages) == ""


class TestTesterTermination:
    """Test the Tester's termination signal behavior."""

    def test_all_tests_passed_signal(self):
        code = "import torch\nx = torch.randn(10).cuda()\n"
        result = run_validation(code)
        assert "ALL_TESTS_PASSED" in result

    def test_validation_failed_signal(self):
        code = "torch.backends.cudnn.benchmark = True\n"
        result = run_validation(code)
        assert "VALIDATION FAILED" in result
        assert "ALL_TESTS_PASSED" not in result

    def test_multiple_failures_listed(self):
        code = (
            'import pycuda.autoinit\n'
            'os.environ["CUDA_VISIBLE_DEVICES"] = "0"\n'
            'torch.backends.cudnn.benchmark = True\n'
        )
        result = run_validation(code)
        assert "FAIL" in result
        # Should report multiple issues
        fail_count = result.count("[FAIL]")
        assert fail_count >= 2

    def test_pass_report_lists_all_checks(self):
        code = "import torch\nx = 1\n"
        result = run_validation(code)
        assert "ALL_TESTS_PASSED" in result
        assert "[PASS]" in result
        # Should list all 8 validators (4 core + 4 pattern-aware)
        pass_count = result.count("[PASS]")
        assert pass_count == 8


class TestReviewerCoderFeedbackLoop:
    """Test that message formats are compatible between agents."""

    def test_coder_message_is_well_formed(self):
        msg = format_coder_message(
            original_code="import torch\nx = torch.randn(10).cuda()",
            pre_migrated_code="import torch\nx = torch.randn(10).cuda()",
            remaining_issues=[
                {"line": 2, "symbol": ".cuda()", "reason": "needs review"},
            ],
        )
        # Message should contain all three sections
        assert "Original CUDA Code" in msg
        assert "Partially Migrated Code" in msg
        assert "Remaining Issues" in msg

    def test_coder_message_with_plan_prefix(self):
        """When a plan exists, it's prepended to the coder message."""
        msg = format_coder_message(
            original_code="x = 1",
            pre_migrated_code="x = 1",
            remaining_issues=[],
        )
        plan = "1. Check imports\n2. Verify device strings\nPLAN COMPLETE"
        combined = f"## Migration Plan (from Planner)\n\n{plan}\n\n---\n\n{msg}"
        assert "Migration Plan" in combined
        assert "PLAN COMPLETE" in combined
        assert "Original CUDA Code" in combined

    def test_reviewer_prompt_compatible_with_coder_output(self):
        """Reviewer's expected input format matches what coder produces."""
        # Reviewer expects to see code — coder returns code in ```python blocks
        assert "code" in REVIEWER_SYSTEM_PROMPT.lower()
        # Reviewer knows to say APPROVED or ISSUES FOUND
        assert "APPROVED" in REVIEWER_SYSTEM_PROMPT
        assert "ISSUES FOUND" in REVIEWER_SYSTEM_PROMPT

    def test_coder_prompt_mentions_code_block_format(self):
        """Coder is instructed to return code in a code block."""
        assert "```python" in CODER_SYSTEM_PROMPT


class TestMalformedLLMResponses:
    """Test handling of unexpected/malformed responses from LLMs."""

    def test_extract_code_from_response_with_explanation(self):
        """LLM might add explanation before/after code."""
        text = (
            "I've fixed the migration issues:\n\n"
            "```python\nimport torch\nx = 1\n```\n\n"
            "The main changes were to replace cudnn with miopen."
        )
        assert _extract_code_block(text) == "import torch\nx = 1"

    def test_extract_code_from_response_with_no_language_tag(self):
        text = "```\nimport torch\n```"
        assert _extract_code_block(text) == "import torch"

    def test_handle_response_with_only_text(self):
        text = "I understand the task. Let me work on the migration."
        assert _extract_code_block(text) is None

    def test_handle_response_with_inline_code_only(self):
        """Inline backticks (single) should not match."""
        text = "Use `import hip` instead of `import pycuda`."
        assert _extract_code_block(text) is None

    def test_handle_response_with_truncated_code_block(self):
        """LLM might get cut off mid-response."""
        text = "```python\nimport torch\nx = 1"
        # No closing ```, so should return None
        assert _extract_code_block(text) is None
