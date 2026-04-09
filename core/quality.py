"""Migration quality metrics — confidence scoring and review checklist."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ScoredChange:
    """A migration change annotated with a confidence score."""

    line: int
    original: str
    replacement: str
    rule: str
    confidence: float  # 0.0–1.0
    source: str  # "rule" or "agent"

    @property
    def level(self) -> str:
        if self.confidence >= 0.9:
            return "high"
        if self.confidence >= 0.5:
            return "needs_review"
        return "low"


@dataclass
class MigrationQualityReport:
    """Aggregated quality metrics for a migration run."""

    file_path: str
    changes: list[ScoredChange] = field(default_factory=list)

    @property
    def high_confidence(self) -> list[ScoredChange]:
        return [c for c in self.changes if c.confidence >= 0.9]

    @property
    def needs_review(self) -> list[ScoredChange]:
        return [c for c in self.changes if 0.5 <= c.confidence < 0.9]

    @property
    def low_confidence(self) -> list[ScoredChange]:
        return [c for c in self.changes if c.confidence < 0.5]

    @property
    def overall_score(self) -> float:
        """Weighted average confidence across all changes. Returns 1.0 if no changes."""
        if not self.changes:
            return 1.0
        return sum(c.confidence for c in self.changes) / len(self.changes)

    @property
    def summary(self) -> dict[str, int]:
        return {
            "high": len(self.high_confidence),
            "needs_review": len(self.needs_review),
            "low": len(self.low_confidence),
            "total": len(self.changes),
        }


# --- Confidence assignment helpers ---

# Rule-based changes have fixed confidence based on the type of rule applied.
_RULE_CONFIDENCE: dict[str, float] = {
    "torch.backends.cudnn → miopen": 1.0,
    "Environment variable CUDA → HIP": 1.0,
    "Import replacement": 0.95,  # slight risk with complex import patterns
}


def confidence_for_rule(rule: str) -> float:
    """Return a confidence score for a rule-based change."""
    # Exact match first
    if rule in _RULE_CONFIDENCE:
        return _RULE_CONFIDENCE[rule]
    # API mapping (e.g., "cudaMalloc → hipMalloc") — check if it's a direct mapping
    if "→" in rule:
        return 1.0
    return 0.9  # default for known rule-based changes


def confidence_for_remaining(reason: str) -> float:
    """Return a confidence estimate for an issue flagged as remaining.

    These haven't been fixed by the rule engine, so confidence represents
    how likely the LLM agent (or manual reviewer) will handle it correctly.
    """
    reason_lower = reason.lower()
    if "kernel launch" in reason_lower:
        return 0.3  # complex, needs manual review
    if "unknown cuda api" in reason_lower:
        return 0.4
    if "low confidence" in reason_lower:
        return 0.5
    if "manual review" in reason_lower:
        return 0.5
    if "needs llm review" in reason_lower:
        return 0.6
    return 0.5


# --- Quality report builders ---

def build_quality_report(
    file_path: str,
    applied: list,  # list[AppliedChange]
    remaining: list,  # list[RemainingIssue]
    agent_used: bool = False,
    validation_passed: bool = False,
) -> MigrationQualityReport:
    """Build a MigrationQualityReport from migration results.

    Args:
        file_path: The source file path.
        applied: Applied changes from the rule-based migrator.
        remaining: Issues that were not auto-fixed.
        agent_used: Whether the LLM agent pipeline was used.
        validation_passed: Whether all validation checks passed.
    """
    report = MigrationQualityReport(file_path=file_path)

    # Score applied (rule-based) changes
    for change in applied:
        conf = getattr(change, "confidence", None)
        if conf is None:
            conf = confidence_for_rule(change.rule)
        report.changes.append(ScoredChange(
            line=change.line,
            original=change.original,
            replacement=change.replacement,
            rule=change.rule,
            confidence=conf,
            source="rule",
        ))

    # Score remaining issues — these weren't auto-fixed
    for issue in remaining:
        base_conf = confidence_for_remaining(issue.reason)

        # Boost if agent was used (it attempted a fix)
        if agent_used:
            base_conf = min(base_conf + 0.15, 0.95)

        # Boost if validation passed (the fix works)
        if validation_passed:
            base_conf = min(base_conf + 0.1, 0.95)

        report.changes.append(ScoredChange(
            line=issue.line,
            original=issue.symbol,
            replacement="(handled by LLM agent)" if agent_used else "(needs manual review)",
            rule=issue.reason,
            confidence=base_conf,
            source="agent" if agent_used else "unfixed",
        ))

    logger.info(
        "Quality report for %s: overall=%.0f%%, high=%d, review=%d, low=%d",
        file_path,
        report.overall_score * 100,
        len(report.high_confidence),
        len(report.needs_review),
        len(report.low_confidence),
    )

    return report


def generate_review_checklist(report: MigrationQualityReport) -> str:
    """Generate a markdown review checklist from a quality report.

    Items that need review are listed with context and the reason for
    low confidence, so the developer knows exactly what to check.
    """
    lines: list[str] = []
    lines.append(f"# Migration Review Checklist — {report.file_path}")
    lines.append("")
    lines.append(
        f"Overall confidence: **{report.overall_score:.0%}** "
        f"({report.summary['high']} high, "
        f"{report.summary['needs_review']} review, "
        f"{report.summary['low']} low)"
    )
    lines.append("")

    items_to_review = report.needs_review + report.low_confidence
    if not items_to_review:
        lines.append("All changes are high-confidence. No manual review needed.")
        return "\n".join(lines)

    items_to_review.sort(key=lambda c: c.confidence)

    lines.append("## Items Requiring Review")
    lines.append("")

    for item in items_to_review:
        conf_pct = f"{item.confidence:.0%}"
        icon = "!!" if item.confidence < 0.5 else "?"
        lines.append(f"- [{icon}] **Line {item.line}** (confidence: {conf_pct}, source: {item.source})")
        lines.append(f"  - Original: `{item.original}`")
        lines.append(f"  - Migrated: `{item.replacement}`")
        lines.append(f"  - Reason: {item.rule}")
        lines.append("")

    return "\n".join(lines)
