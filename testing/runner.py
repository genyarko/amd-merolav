"""Sandboxed execution of migrated code for validation."""

from __future__ import annotations

import subprocess
import sys
import tempfile
import textwrap
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ExecutionResult:
    status: str  # "PASS", "FAIL", "WARN", "ERROR"
    stdout: str
    stderr: str
    return_code: int
    error_summary: str = ""


# Wrapper script that sets up mocks before running the user's code
_WRAPPER_TEMPLATE = textwrap.dedent("""\
    import sys
    import os

    # Add project root to path so testing.mock_hip is importable
    sys.path.insert(0, {project_root!r})

    from testing.mock_hip import mock_hip_environment

    with mock_hip_environment():
        # Read and exec the user's migrated code
        code_path = {code_path!r}
        with open(code_path, "r", encoding="utf-8") as f:
            source = f.read()

        # Compile first to get better error messages
        compiled = compile(source, code_path, "exec")
        exec(compiled, {{"__name__": "__main__", "__file__": code_path}})

    print("\\n__EXECUTION_COMPLETE__")
""")

# Project root is the Rocm/ directory
_PROJECT_ROOT = str(Path(__file__).parent.parent)


def execute_migrated_code(
    code: str,
    timeout: int = 30,
) -> ExecutionResult:
    """Execute migrated code in a subprocess with HIP mocks.

    The code runs in an isolated Python process with mock_hip_environment
    active, so imports like `hip`, `miopen`, `migraphx` resolve to mocks.

    Args:
        code: The migrated Python source code to execute.
        timeout: Maximum execution time in seconds.

    Returns:
        ExecutionResult with status, stdout, stderr, and return code.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        # Write the user's code to a temp file
        code_file = Path(tmpdir) / "migrated_code.py"
        code_file.write_text(code, encoding="utf-8")

        # Write the wrapper script
        wrapper_file = Path(tmpdir) / "wrapper.py"
        wrapper_source = _WRAPPER_TEMPLATE.format(
            project_root=_PROJECT_ROOT,
            code_path=str(code_file),
        )
        wrapper_file.write_text(wrapper_source, encoding="utf-8")

        try:
            proc = subprocess.run(
                [sys.executable, str(wrapper_file)],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=tmpdir,
            )
        except subprocess.TimeoutExpired:
            return ExecutionResult(
                status="ERROR",
                stdout="",
                stderr="",
                return_code=-1,
                error_summary=f"Execution timed out after {timeout}s",
            )

        stdout = proc.stdout
        stderr = proc.stderr
        rc = proc.returncode

        if rc == 0 and "__EXECUTION_COMPLETE__" in stdout:
            # Check stderr for warnings
            if stderr.strip():
                return ExecutionResult(
                    status="WARN",
                    stdout=stdout,
                    stderr=stderr,
                    return_code=rc,
                    error_summary="Completed with warnings",
                )
            return ExecutionResult(
                status="PASS",
                stdout=stdout,
                stderr=stderr,
                return_code=rc,
            )
        else:
            # Extract the most useful error line
            error_lines = stderr.strip().splitlines()
            summary = error_lines[-1] if error_lines else "Unknown error"
            return ExecutionResult(
                status="FAIL",
                stdout=stdout,
                stderr=stderr,
                return_code=rc,
                error_summary=summary,
            )
