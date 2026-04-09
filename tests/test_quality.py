"""Tests for core/quality.py — confidence scoring and quality reports."""

from __future__ import annotations

import pytest

from core.analyzer import analyze_source
from core.migrator import AppliedChange, MigrationResult, RemainingIssue, migrate
from core.quality import (
    MigrationQualityReport,
    ScoredChange,
    build_quality_report,
    confidence_for_remaining,
    confidence_for_rule,
    generate_review_checklist,
)


# --- ScoredChange ---


class TestScoredChange:
    def test_high_confidence_level(self):
        c = ScoredChange(line=1, original="a", replacement="b", rule="r", confidence=0.95, source="rule")
        assert c.level == "high"

    def test_needs_review_level(self):
        c = ScoredChange(line=1, original="a", replacement="b", rule="r", confidence=0.7, source="rule")
        assert c.level == "needs_review"

    def test_low_confidence_level(self):
        c = ScoredChange(line=1, original="a", replacement="b", rule="r", confidence=0.3, source="rule")
        assert c.level == "low"

    def test_boundary_high(self):
        c = ScoredChange(line=1, original="a", replacement="b", rule="r", confidence=0.9, source="rule")
        assert c.level == "high"

    def test_boundary_needs_review(self):
        c = ScoredChange(line=1, original="a", replacement="b", rule="r", confidence=0.5, source="rule")
        assert c.level == "needs_review"

    def test_boundary_low(self):
        c = ScoredChange(line=1, original="a", replacement="b", rule="r", confidence=0.49, source="rule")
        assert c.level == "low"


# --- MigrationQualityReport ---


class TestMigrationQualityReport:
    def test_empty_report_score_is_1(self):
        r = MigrationQualityReport(file_path="test.py")
        assert r.overall_score == 1.0

    def test_all_high_confidence(self):
        r = MigrationQualityReport(file_path="test.py", changes=[
            ScoredChange(1, "a", "b", "r", 1.0, "rule"),
            ScoredChange(2, "c", "d", "r", 0.95, "rule"),
        ])
        assert len(r.high_confidence) == 2
        assert len(r.needs_review) == 0
        assert len(r.low_confidence) == 0
        assert r.overall_score == pytest.approx(0.975)

    def test_mixed_confidence(self):
        r = MigrationQualityReport(file_path="test.py", changes=[
            ScoredChange(1, "a", "b", "r", 1.0, "rule"),
            ScoredChange(2, "c", "d", "r", 0.7, "agent"),
            ScoredChange(3, "e", "f", "r", 0.3, "unfixed"),
        ])
        assert len(r.high_confidence) == 1
        assert len(r.needs_review) == 1
        assert len(r.low_confidence) == 1
        assert r.overall_score == pytest.approx(2.0 / 3)

    def test_summary_counts(self):
        r = MigrationQualityReport(file_path="test.py", changes=[
            ScoredChange(1, "a", "b", "r", 1.0, "rule"),
            ScoredChange(2, "c", "d", "r", 0.6, "agent"),
        ])
        s = r.summary
        assert s["high"] == 1
        assert s["needs_review"] == 1
        assert s["low"] == 0
        assert s["total"] == 2


# --- Confidence helpers ---


class TestConfidenceForRule:
    def test_backend_replacement(self):
        assert confidence_for_rule("torch.backends.cudnn → miopen") == 1.0

    def test_env_var_replacement(self):
        assert confidence_for_rule("Environment variable CUDA → HIP") == 1.0

    def test_import_replacement(self):
        assert confidence_for_rule("Import replacement") == 0.95

    def test_api_mapping(self):
        assert confidence_for_rule("cudaMalloc → hipMalloc") == 1.0

    def test_unknown_rule_gets_default(self):
        assert confidence_for_rule("some new rule") == 0.9


class TestConfidenceForRemaining:
    def test_kernel_launch(self):
        c = confidence_for_remaining("CUDA kernel launch syntax — needs LLM migration")
        assert c <= 0.4

    def test_unknown_api(self):
        c = confidence_for_remaining("Unknown CUDA API — needs LLM review")
        assert c <= 0.5

    def test_low_confidence_api(self):
        c = confidence_for_remaining("Low confidence (0.5) — needs LLM review")
        assert 0.4 <= c <= 0.6

    def test_manual_review(self):
        c = confidence_for_remaining("cudnn backend reference needs manual review")
        assert 0.4 <= c <= 0.6


# --- build_quality_report ---


class TestBuildQualityReport:
    def test_rule_only_all_applied(self):
        applied = [
            AppliedChange(1, "old1", "new1", "Environment variable CUDA → HIP", 1.0),
            AppliedChange(2, "old2", "new2", "torch.backends.cudnn → miopen", 1.0),
        ]
        report = build_quality_report("test.py", applied, [], agent_used=False)
        assert report.overall_score >= 0.9
        assert len(report.high_confidence) == 2
        assert all(c.source == "rule" for c in report.changes)

    def test_with_remaining_issues(self):
        applied = [
            AppliedChange(1, "old", "new", "Environment variable CUDA → HIP", 1.0),
        ]
        remaining = [
            RemainingIssue(5, "<<<...>>>", "CUDA kernel launch syntax — needs LLM migration", 0.3),
        ]
        report = build_quality_report("test.py", applied, remaining, agent_used=False)
        assert len(report.changes) == 2
        assert len(report.low_confidence) >= 1
        assert report.overall_score < 1.0

    def test_agent_boost(self):
        remaining = [
            RemainingIssue(5, "sym", "Unknown CUDA API — needs LLM review", 0.4),
        ]
        report_no_agent = build_quality_report("t.py", [], remaining, agent_used=False)
        report_with_agent = build_quality_report("t.py", [], remaining, agent_used=True)

        no_agent_conf = report_no_agent.changes[0].confidence
        with_agent_conf = report_with_agent.changes[0].confidence
        assert with_agent_conf > no_agent_conf

    def test_validation_boost(self):
        remaining = [
            RemainingIssue(5, "sym", "Unknown CUDA API — needs LLM review", 0.4),
        ]
        report_no_val = build_quality_report("t.py", [], remaining, agent_used=True, validation_passed=False)
        report_val = build_quality_report("t.py", [], remaining, agent_used=True, validation_passed=True)

        no_val_conf = report_no_val.changes[0].confidence
        val_conf = report_val.changes[0].confidence
        assert val_conf > no_val_conf

    def test_empty_migration(self):
        report = build_quality_report("test.py", [], [])
        assert report.overall_score == 1.0
        assert report.summary["total"] == 0


# --- generate_review_checklist ---


class TestReviewChecklist:
    def test_all_high_confidence_no_review(self):
        report = MigrationQualityReport(file_path="test.py", changes=[
            ScoredChange(1, "a", "b", "r", 1.0, "rule"),
        ])
        checklist = generate_review_checklist(report)
        assert "No manual review needed" in checklist

    def test_low_confidence_items_listed(self):
        report = MigrationQualityReport(file_path="test.py", changes=[
            ScoredChange(1, "a", "b", "r", 1.0, "rule"),
            ScoredChange(5, "<<<...>>>", "(manual)", "kernel launch", 0.3, "unfixed"),
        ])
        checklist = generate_review_checklist(report)
        assert "Line 5" in checklist
        assert "30%" in checklist
        assert "kernel launch" in checklist

    def test_needs_review_items_listed(self):
        report = MigrationQualityReport(file_path="test.py", changes=[
            ScoredChange(3, "sym", "repl", "some reason", 0.7, "agent"),
        ])
        checklist = generate_review_checklist(report)
        assert "Line 3" in checklist
        assert "70%" in checklist

    def test_checklist_is_markdown(self):
        report = MigrationQualityReport(file_path="test.py", changes=[
            ScoredChange(1, "a", "b", "rule1", 0.4, "unfixed"),
        ])
        checklist = generate_review_checklist(report)
        assert checklist.startswith("# Migration Review Checklist")
        assert "**" in checklist  # markdown bold
        assert "- [" in checklist  # checklist items

    def test_items_sorted_by_confidence_ascending(self):
        report = MigrationQualityReport(file_path="test.py", changes=[
            ScoredChange(1, "a", "b", "medium", 0.7, "agent"),
            ScoredChange(2, "c", "d", "low", 0.3, "unfixed"),
            ScoredChange(3, "e", "f", "high", 0.95, "rule"),
        ])
        checklist = generate_review_checklist(report)
        # Line 2 (0.3) should appear before line 1 (0.7)
        pos_line2 = checklist.index("Line 2")
        pos_line1 = checklist.index("Line 1")
        assert pos_line2 < pos_line1


# --- Integration: analyzer + migrator + quality ---


class TestQualityIntegration:
    def test_simple_fixture_quality(self):
        from pathlib import Path
        FIXTURES = Path(__file__).parent / "fixtures"
        source = (FIXTURES / "sample_cuda_simple.py").read_text()

        report = analyze_source(source, "sample_cuda_simple.py")
        result = migrate(source, report)

        quality = build_quality_report(
            file_path="sample_cuda_simple.py",
            applied=result.applied,
            remaining=result.remaining,
        )
        # Simple fixture should have high overall confidence (all rule-based)
        assert quality.overall_score >= 0.9
        assert quality.summary["total"] >= 3

    def test_pycuda_fixture_has_lower_confidence(self):
        from pathlib import Path
        FIXTURES = Path(__file__).parent / "fixtures"
        source = (FIXTURES / "sample_cuda_pycuda.py").read_text()

        report = analyze_source(source, "sample_cuda_pycuda.py")
        result = migrate(source, report)

        quality = build_quality_report(
            file_path="sample_cuda_pycuda.py",
            applied=result.applied,
            remaining=result.remaining,
        )
        # pycuda has complex patterns, so should have some lower-confidence items
        assert quality.summary["total"] >= 1

    def test_applied_changes_carry_confidence(self):
        from pathlib import Path
        FIXTURES = Path(__file__).parent / "fixtures"
        source = (FIXTURES / "sample_cuda_simple.py").read_text()

        report = analyze_source(source, "sample_cuda_simple.py")
        result = migrate(source, report)

        # Verify AppliedChange objects now have confidence
        for change in result.applied:
            assert hasattr(change, "confidence")
            assert 0.0 <= change.confidence <= 1.0

    def test_remaining_issues_carry_confidence(self):
        from pathlib import Path
        FIXTURES = Path(__file__).parent / "fixtures"
        source = (FIXTURES / "sample_cuda_pycuda.py").read_text()

        report = analyze_source(source, "sample_cuda_pycuda.py")
        result = migrate(source, report)

        for issue in result.remaining:
            assert hasattr(issue, "confidence")
            assert 0.0 <= issue.confidence <= 1.0


# --- False-positive detection ---


class TestFalsePositiveDetection:
    def test_comment_line_flagged(self):
        source = "# CUDA_VISIBLE_DEVICES should be replaced\nx = 1\n"
        report = analyze_source(source, "<comment>")
        # If any usage was detected on the comment line, it should be flagged
        for usage in report.usages:
            if usage.line == 1:
                assert usage.is_false_positive

    def test_real_env_var_not_flagged(self):
        source = 'os.environ["CUDA_VISIBLE_DEVICES"] = "0"\n'
        report = analyze_source(source, "<real>")
        for usage in report.usages:
            if usage.symbol == "CUDA_VISIBLE_DEVICES":
                assert not usage.is_false_positive

    def test_real_backend_not_flagged(self):
        source = "torch.backends.cudnn.benchmark = True\n"
        report = analyze_source(source, "<backend>")
        for usage in report.usages:
            if usage.category == "backend":
                assert not usage.is_false_positive

    def test_code_usage_not_flagged(self):
        source = "import pycuda.autoinit\n"
        report = analyze_source(source, "<import>")
        for usage in report.usages:
            assert not usage.is_false_positive
