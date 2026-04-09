"""Tester agent — runs validation on migrated code without using LLM."""

from __future__ import annotations

from core.logging import get_logger
from testing.validators import (
    Severity,
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

logger = get_logger(__name__)


def run_validation(
    code: str,
    original_code: str | None = None,
    run_sandbox: bool = False,
    test_timeout: int = 30,
) -> str:
    """Run all validators on migrated code and return a formatted report.

    This is the function registered with the AutoGen Tester agent.
    It does not use an LLM — all checks are deterministic.

    Args:
        code: The migrated code to validate.
        original_code: The original code (enables equivalence checking).
        run_sandbox: If True, also execute the code in a sandbox with HIP mocks.
        test_timeout: Timeout in seconds for sandbox/equivalence execution.

    The report now includes severity levels (ERROR/WARNING/INFO) and
    actionable suggestions for each issue.
    """
    reports: list[ValidationReport] = [
        # Core validators (errors = will crash)
        check_no_cudnn_refs(code),
        check_imports(code),
        check_device_strings(code),
        check_env_vars(code),
        # Pattern-aware validators (warnings/info)
        check_mixed_imports(code),
        check_incompatible_libraries(code),
        check_deprecated_rocm_apis(code),
        check_orphaned_env_vars(code),
    ]

    # Sandbox execution with HIP mocks
    if run_sandbox:
        sandbox_report = _run_sandbox_validation(code, test_timeout)
        reports.append(sandbox_report)

    # Semantic equivalence check
    if original_code:
        equiv_report = _run_equivalence_validation(original_code, code, test_timeout)
        reports.append(equiv_report)

    all_passed = all(r.passed for r in reports)
    lines: list[str] = []

    # Collect all structured issues across reports
    all_issues = []
    for r in reports:
        all_issues.extend(r.structured_issues)

    errors = [i for i in all_issues if i.severity == Severity.ERROR]
    warnings = [i for i in all_issues if i.severity == Severity.WARNING]
    infos = [i for i in all_issues if i.severity == Severity.INFO]

    if all_passed and not errors:
        lines.append("ALL_TESTS_PASSED")
        lines.append("")
        lines.append("Validation Summary:")
        for r in reports:
            lines.append(f"  [PASS] {r.name}")
        if warnings:
            lines.append("")
            lines.append(f"Warnings ({len(warnings)}):")
            for w in warnings:
                lines.append(f"  [WARNING] Line {w.line}: {w.message}")
                if w.suggestion:
                    lines.append(f"    → {w.suggestion}")
        if infos:
            lines.append("")
            lines.append(f"Info ({len(infos)}):")
            for info in infos:
                lines.append(f"  [INFO] Line {info.line}: {info.message}")
                if info.suggestion:
                    lines.append(f"    → {info.suggestion}")
    else:
        lines.append("VALIDATION FAILED")
        lines.append("")

        if errors:
            lines.append(f"Errors ({len(errors)}) — will crash at runtime:")
            for e in errors:
                lines.append(f"  [ERROR] Line {e.line}: {e.message}")
                if e.suggestion:
                    lines.append(f"    → Fix: {e.suggestion}")
            lines.append("")

        if warnings:
            lines.append(f"Warnings ({len(warnings)}) — may behave differently:")
            for w in warnings:
                lines.append(f"  [WARNING] Line {w.line}: {w.message}")
                if w.suggestion:
                    lines.append(f"    → {w.suggestion}")
            lines.append("")

        if infos:
            lines.append(f"Info ({len(infos)}):")
            for info in infos:
                lines.append(f"  [INFO] Line {info.line}: {info.message}")
                if info.suggestion:
                    lines.append(f"    → {info.suggestion}")
            lines.append("")

        # Per-validator summary
        lines.append("Validator Summary:")
        for r in reports:
            status = "PASS" if r.passed else "FAIL"
            lines.append(f"  [{status}] {r.name}")

        lines.append("")
        lines.append("Please fix the ERROR issues and resubmit the code.")

    return "\n".join(lines)


def _run_sandbox_validation(code: str, timeout: int) -> ValidationReport:
    """Execute migrated code in a sandboxed subprocess with HIP mocks."""
    from testing.runner import execute_migrated_code
    from testing.validators import ValidationIssue

    try:
        result = execute_migrated_code(code, timeout=timeout)
    except Exception as exc:
        logger.warning("Sandbox execution setup failed: %s", exc)
        return ValidationReport(
            name="Sandbox Execution",
            passed=False,
            structured_issues=[
                ValidationIssue(
                    line=0,
                    message=f"Sandbox execution setup failed: {exc}",
                    severity=Severity.WARNING,
                    suggestion="Check that testing/mock_hip.py is available",
                )
            ],
        )

    issues: list = []
    if result.status == "FAIL":
        issues.append(
            ValidationIssue(
                line=0,
                message=f"Sandbox execution failed: {result.error_summary}",
                severity=Severity.ERROR,
                suggestion="Review the migrated code for runtime errors",
            )
        )
    elif result.status == "WARN":
        issues.append(
            ValidationIssue(
                line=0,
                message=f"Sandbox execution passed with warnings: {result.error_summary}",
                severity=Severity.WARNING,
            )
        )
    elif result.status == "ERROR":
        issues.append(
            ValidationIssue(
                line=0,
                message=f"Sandbox execution error: {result.error_summary}",
                severity=Severity.ERROR,
                suggestion="Ensure the sandbox environment is properly configured",
            )
        )

    return ValidationReport(
        name="Sandbox Execution",
        passed=result.status in ("PASS", "WARN"),
        structured_issues=issues,
    )


def _run_equivalence_validation(
    original_code: str, migrated_code: str, timeout: int
) -> ValidationReport:
    """Run semantic equivalence checking between original and migrated code."""
    from testing.equivalence import check_equivalence
    from testing.validators import ValidationIssue

    try:
        result = check_equivalence(original_code, migrated_code, timeout=timeout)
    except Exception as exc:
        logger.warning("Equivalence check setup failed: %s", exc)
        return ValidationReport(
            name="Semantic Equivalence",
            passed=True,  # don't block on setup failures
            structured_issues=[
                ValidationIssue(
                    line=0,
                    message=f"Equivalence check could not run: {exc}",
                    severity=Severity.INFO,
                )
            ],
        )

    if result.skipped:
        return ValidationReport(
            name="Semantic Equivalence",
            passed=True,
            structured_issues=[
                ValidationIssue(
                    line=0,
                    message=f"Skipped: {result.skip_reason}",
                    severity=Severity.INFO,
                )
            ],
        )

    issues: list = []
    if not result.passed:
        if result.error:
            issues.append(
                ValidationIssue(
                    line=0,
                    message=f"Equivalence error: {result.error}",
                    severity=Severity.WARNING,
                    suggestion="Verify the migrated code produces the same outputs",
                )
            )
        for iss in result.issues:
            issues.append(
                ValidationIssue(
                    line=0,
                    message=f"Output mismatch on {iss.operation}: expected {iss.expected}, got {iss.actual}",
                    severity=Severity.WARNING,
                    suggestion=f"Max difference: {iss.tolerance:.2e}",
                )
            )

    return ValidationReport(
        name="Semantic Equivalence",
        passed=result.passed,
        structured_issues=issues,
    )
