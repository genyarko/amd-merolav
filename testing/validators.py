"""AST-based validators for migrated ROCm code."""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field


@dataclass
class ValidationReport:
    name: str
    passed: bool
    issues: list[str] = field(default_factory=list)


def check_no_cudnn_refs(code: str) -> ValidationReport:
    """Check that no torch.backends.cudnn references remain."""
    report = ValidationReport(name="No cuDNN references", passed=True)

    # Check for cudnn in attribute access
    for i, line in enumerate(code.splitlines(), 1):
        stripped = line.strip()
        if stripped.startswith("#"):
            continue  # skip comments
        if "cudnn" in line and "# " not in line.split("cudnn")[0].split("\n")[-1]:
            # Avoid flagging comments that mention cudnn
            # But flag actual code references
            if re.search(r'\bcudnn\b', line):
                # Check it's not in a comment
                code_part = line.split("#")[0]
                if "cudnn" in code_part:
                    report.passed = False
                    report.issues.append(
                        f"Line {i}: cuDNN reference found: {stripped}"
                    )

    return report


def check_imports(code: str) -> ValidationReport:
    """Check that CUDA-only imports have been replaced."""
    report = ValidationReport(name="Import validation", passed=True)

    cuda_imports = ["pycuda", "tensorrt"]

    try:
        tree = ast.parse(code)
    except SyntaxError:
        report.issues.append("Code has syntax errors — cannot validate imports")
        report.passed = False
        return report

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                for bad in cuda_imports:
                    if bad in alias.name.lower():
                        report.passed = False
                        report.issues.append(
                            f"Line {node.lineno}: CUDA-only import '{alias.name}' still present"
                        )
        elif isinstance(node, ast.ImportFrom):
            mod = (node.module or "").lower()
            for bad in cuda_imports:
                if bad in mod:
                    report.passed = False
                    report.issues.append(
                        f"Line {node.lineno}: CUDA-only import 'from {node.module}' still present"
                    )

    return report


def check_device_strings(code: str) -> ValidationReport:
    """Verify device strings are ROCm-compatible.

    On ROCm, "cuda" IS the correct device string.
    This validator checks for things that are NOT correct.
    """
    report = ValidationReport(name="Device string validation", passed=True)

    # "rocm" or "hip" as device strings would be wrong
    wrong_devices = [
        (r'torch\.device\(\s*["\']rocm', "rocm"),
        (r'torch\.device\(\s*["\']hip', "hip"),
        (r'\.to\(\s*["\']rocm', "rocm"),
        (r'\.to\(\s*["\']hip', "hip"),
    ]

    for i, line in enumerate(code.splitlines(), 1):
        if line.strip().startswith("#"):
            continue
        for pattern, device in wrong_devices:
            if re.search(pattern, line):
                report.passed = False
                report.issues.append(
                    f'Line {i}: Wrong device string "{device}" — '
                    f'use "cuda" on ROCm (PyTorch HIP backend uses "cuda")'
                )

    return report


def check_env_vars(code: str) -> ValidationReport:
    """Check that CUDA environment variables have been replaced."""
    report = ValidationReport(name="Environment variable validation", passed=True)

    cuda_env_vars = [
        "CUDA_VISIBLE_DEVICES",
        "CUDA_LAUNCH_BLOCKING",
        "CUDA_DEVICE_ORDER",
    ]

    for i, line in enumerate(code.splitlines(), 1):
        if line.strip().startswith("#"):
            continue
        code_part = line.split("#")[0]  # ignore comments
        for var in cuda_env_vars:
            if var in code_part:
                report.passed = False
                report.issues.append(
                    f"Line {i}: CUDA env var '{var}' still present — "
                    f"replace with HIP equivalent"
                )

    return report
