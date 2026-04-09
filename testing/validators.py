"""AST-based validators for migrated ROCm code."""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from enum import Enum


class Severity(Enum):
    ERROR = "ERROR"      # Will crash at runtime
    WARNING = "WARNING"  # May behave differently
    INFO = "INFO"        # Cosmetic / informational


@dataclass
class ValidationIssue:
    line: int
    message: str
    severity: Severity
    suggestion: str = ""


@dataclass
class ValidationReport:
    name: str
    passed: bool
    issues: list[str] = field(default_factory=list)
    structured_issues: list[ValidationIssue] = field(default_factory=list)

    def add_issue(
        self,
        line: int,
        message: str,
        severity: Severity,
        suggestion: str = "",
    ) -> None:
        """Add a structured issue and the legacy string form."""
        self.structured_issues.append(ValidationIssue(
            line=line, message=message,
            severity=severity, suggestion=suggestion,
        ))
        self.issues.append(f"Line {line}: {message}")
        if severity == Severity.ERROR:
            self.passed = False


def check_no_cudnn_refs(code: str) -> ValidationReport:
    """Check that no torch.backends.cudnn references remain."""
    report = ValidationReport(name="No cuDNN references", passed=True)

    for i, line in enumerate(code.splitlines(), 1):
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        if "cudnn" in line:
            if re.search(r'\bcudnn\b', line):
                code_part = line.split("#")[0]
                if "cudnn" in code_part:
                    report.add_issue(
                        line=i,
                        message=f"cuDNN reference found: {stripped}",
                        severity=Severity.ERROR,
                        suggestion="Replace torch.backends.cudnn with torch.backends.miopen",
                    )

    if not report.structured_issues:
        report.passed = True
    return report


def check_imports(code: str) -> ValidationReport:
    """Check that CUDA-only imports have been replaced."""
    report = ValidationReport(name="Import validation", passed=True)

    cuda_imports = ["pycuda", "tensorrt"]

    try:
        tree = ast.parse(code)
    except SyntaxError:
        report.add_issue(
            line=0, message="Code has syntax errors — cannot validate imports",
            severity=Severity.ERROR,
        )
        return report

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                for bad in cuda_imports:
                    if bad in alias.name.lower():
                        report.add_issue(
                            line=node.lineno,
                            message=f"CUDA-only import '{alias.name}' still present",
                            severity=Severity.ERROR,
                            suggestion=f"Replace with ROCm equivalent (e.g., hip, migraphx)",
                        )
        elif isinstance(node, ast.ImportFrom):
            mod = (node.module or "").lower()
            for bad in cuda_imports:
                if bad in mod:
                    report.add_issue(
                        line=node.lineno,
                        message=f"CUDA-only import 'from {node.module}' still present",
                        severity=Severity.ERROR,
                        suggestion=f"Replace with ROCm equivalent",
                    )

    return report


def check_device_strings(code: str) -> ValidationReport:
    """Verify device strings are ROCm-compatible.

    On ROCm, "cuda" IS the correct device string.
    This validator checks for things that are NOT correct.
    """
    report = ValidationReport(name="Device string validation", passed=True)

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
                report.add_issue(
                    line=i,
                    message=f'Wrong device string "{device}"',
                    severity=Severity.ERROR,
                    suggestion='Use "cuda" on ROCm — PyTorch HIP backend uses "cuda" as the device string',
                )

    return report


def check_env_vars(code: str) -> ValidationReport:
    """Check that CUDA environment variables have been replaced."""
    report = ValidationReport(name="Environment variable validation", passed=True)

    cuda_env_vars = {
        "CUDA_VISIBLE_DEVICES": "HIP_VISIBLE_DEVICES",
        "CUDA_LAUNCH_BLOCKING": "HIP_LAUNCH_BLOCKING",
        "CUDA_DEVICE_ORDER": "HIP_DEVICE_ORDER",
    }

    for i, line in enumerate(code.splitlines(), 1):
        if line.strip().startswith("#"):
            continue
        code_part = line.split("#")[0]
        for cuda_var, hip_var in cuda_env_vars.items():
            if cuda_var in code_part:
                report.add_issue(
                    line=i,
                    message=f"CUDA env var '{cuda_var}' still present",
                    severity=Severity.ERROR,
                    suggestion=f"Replace with {hip_var}",
                )

    return report


# =====================================================================
# New pattern-aware validators (Phase 13)
# =====================================================================

def check_mixed_imports(code: str) -> ValidationReport:
    """Detect mixed CUDA and ROCm imports (partially migrated state)."""
    report = ValidationReport(name="Mixed import detection", passed=True)

    cuda_imports = set()
    rocm_imports = set()

    cuda_markers = ["pycuda", "tensorrt", "cupy"]
    rocm_markers = ["hip", "migraphx", "miopen", "rocm"]

    try:
        tree = ast.parse(code)
    except SyntaxError:
        return report

    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            mod = ""
            if isinstance(node, ast.Import):
                mod = " ".join(a.name.lower() for a in node.names)
            else:
                mod = (node.module or "").lower()

            for m in cuda_markers:
                if m in mod:
                    cuda_imports.add((node.lineno, mod))
            for m in rocm_markers:
                if m in mod:
                    rocm_imports.add((node.lineno, mod))

    if cuda_imports and rocm_imports:
        cuda_lines = ", ".join(str(ln) for ln, _ in sorted(cuda_imports))
        rocm_lines = ", ".join(str(ln) for ln, _ in sorted(rocm_imports))
        report.add_issue(
            line=min(ln for ln, _ in cuda_imports),
            message=f"Mixed CUDA imports (lines {cuda_lines}) and ROCm imports (lines {rocm_lines})",
            severity=Severity.WARNING,
            suggestion="Migration appears incomplete — remove remaining CUDA-only imports",
        )

    return report


def check_orphaned_env_vars(code: str) -> ValidationReport:
    """Detect environment variables that are set but never used in the code."""
    report = ValidationReport(name="Orphaned environment variable check", passed=True)

    env_set_re = re.compile(
        r"""os\.environ\[['"](\w+)['"]\]\s*=""" "|"
        r"""os\.environ\.setdefault\(\s*['"](\w+)['"]"""
    )
    env_get_re = re.compile(
        r"""os\.environ(?:\.get)?\[?\(?\s*['"](\w+)['"]""" "|"
        r"""os\.getenv\(\s*['"](\w+)['"]"""
    )

    hip_vars = {"HIP_VISIBLE_DEVICES", "HIP_LAUNCH_BLOCKING", "HIP_DEVICE_ORDER",
                "MIOPEN_FIND_MODE", "MIOPEN_USER_DB_PATH", "ROCR_VISIBLE_DEVICES",
                "PYTORCH_HIP_ALLOC_CONF", "NCCL_SOCKET_IFNAME"}

    lines = code.splitlines()
    set_vars: dict[str, int] = {}  # var_name -> line_number

    for i, line in enumerate(lines, 1):
        if line.strip().startswith("#"):
            continue
        for m in env_set_re.finditer(line):
            var = m.group(1) or m.group(2)
            if var and var in hip_vars:
                set_vars[var] = i

    # Check if any set vars are actually read elsewhere
    for var, line_num in set_vars.items():
        use_count = 0
        for i, line in enumerate(lines, 1):
            if i == line_num:
                continue
            if var in line:
                use_count += 1
        if use_count == 0:
            report.add_issue(
                line=line_num,
                message=f"Environment variable '{var}' is set but never referenced elsewhere",
                severity=Severity.INFO,
                suggestion="Consider removing if unused, or verify it's consumed by an external tool",
            )

    return report


def check_incompatible_libraries(code: str) -> ValidationReport:
    """Detect incompatible library combinations in migrated code."""
    report = ValidationReport(name="Incompatible library check", passed=True)

    incompatible_pairs = [
        ("tensorrt", "migraphx",
         "Both TensorRT (CUDA) and MIGraphX (ROCm) are imported — use only MIGraphX on AMD"),
        ("pycuda", "hip",
         "Both pycuda (CUDA) and hip (ROCm) are imported — use only hip-python on AMD"),
    ]

    try:
        tree = ast.parse(code)
    except SyntaxError:
        return report

    imported_modules: set[str] = set()
    import_lines: dict[str, int] = {}

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                mod = alias.name.lower().split(".")[0]
                imported_modules.add(mod)
                import_lines[mod] = node.lineno
        elif isinstance(node, ast.ImportFrom):
            mod = (node.module or "").lower().split(".")[0]
            if mod:
                imported_modules.add(mod)
                import_lines[mod] = node.lineno

    for lib_a, lib_b, message in incompatible_pairs:
        if lib_a in imported_modules and lib_b in imported_modules:
            report.add_issue(
                line=import_lines.get(lib_a, 0),
                message=message,
                severity=Severity.WARNING,
                suggestion=f"Remove the CUDA-specific library ({lib_a}) import",
            )

    return report


def check_deprecated_rocm_apis(code: str) -> ValidationReport:
    """Check for deprecated ROCm/HIP APIs that may be removed in future versions."""
    report = ValidationReport(name="Deprecated ROCm API check", passed=True)

    deprecated_patterns = [
        (r"\bhipCtxCreate\b", "hipCtxCreate is deprecated in ROCm 6.0+ — use hipSetDevice instead"),
        (r"\bhipCtxDestroy\b", "hipCtxDestroy is deprecated in ROCm 6.0+ — contexts are implicit"),
        (r"\bhipCtxSetCurrent\b", "hipCtxSetCurrent is deprecated — use hipSetDevice"),
        (r"\bhipCtxGetCurrent\b", "hipCtxGetCurrent is deprecated — use hipGetDevice"),
        (r"\bhipCtxSynchronize\b", "hipCtxSynchronize is deprecated — use hipDeviceSynchronize"),
        (r"\bhipMemcpyHtoD\b", "hipMemcpyHtoD is driver API — prefer hipMemcpy with hipMemcpyHostToDevice"),
        (r"\bhipMemcpyDtoH\b", "hipMemcpyDtoH is driver API — prefer hipMemcpy with hipMemcpyDeviceToHost"),
    ]

    for i, line in enumerate(code.splitlines(), 1):
        if line.strip().startswith("#"):
            continue
        code_part = line.split("#")[0]
        for pattern, message in deprecated_patterns:
            if re.search(pattern, code_part):
                report.add_issue(
                    line=i,
                    message=message,
                    severity=Severity.WARNING,
                    suggestion="Update to the recommended API",
                )

    return report
