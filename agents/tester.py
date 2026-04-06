"""Tester agent — runs validation on migrated code without using LLM."""

from __future__ import annotations

from testing.validators import (
    ValidationReport,
    check_device_strings,
    check_env_vars,
    check_imports,
    check_no_cudnn_refs,
)


def run_validation(code: str) -> str:
    """Run all validators on migrated code and return a formatted report.

    This is the function registered with the AutoGen Tester agent.
    It does not use an LLM — all checks are deterministic.
    """
    reports: list[ValidationReport] = [
        check_no_cudnn_refs(code),
        check_imports(code),
        check_device_strings(code),
        check_env_vars(code),
    ]

    all_passed = all(r.passed for r in reports)
    lines: list[str] = []

    if all_passed:
        lines.append("ALL_TESTS_PASSED")
        lines.append("")
        lines.append("Validation Summary:")
        for r in reports:
            lines.append(f"  [PASS] {r.name}")
    else:
        lines.append("VALIDATION FAILED")
        lines.append("")
        for r in reports:
            status = "PASS" if r.passed else "FAIL"
            lines.append(f"  [{status}] {r.name}")
            for issue in r.issues:
                lines.append(f"    - {issue}")
        lines.append("")
        lines.append("Please fix the above issues and resubmit the code.")

    return "\n".join(lines)
