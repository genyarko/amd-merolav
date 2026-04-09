"""Sandboxed execution of migrated code for validation."""

from __future__ import annotations

import subprocess
import sys
import tempfile
import textwrap
from dataclasses import dataclass
from pathlib import Path

from core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ExecutionResult:
    status: str  # "PASS", "FAIL", "WARN", "ERROR"
    stdout: str
    stderr: str
    return_code: int
    error_summary: str = ""


def detect_rocm_runtime() -> bool:
    """Check if a real ROCm/HIP runtime is available on this system.

    Returns True if ``torch.cuda.is_available()`` succeeds on a ROCm build.
    """
    try:
        proc = subprocess.run(
            [sys.executable, "-c",
             "import torch; print(torch.cuda.is_available()); "
             "print(getattr(torch.version, 'hip', 'none'))"],
            capture_output=True, text=True, timeout=15,
        )
        if proc.returncode != 0:
            return False
        lines = proc.stdout.strip().splitlines()
        gpu_available = lines[0].strip() == "True" if lines else False
        has_hip = len(lines) > 1 and lines[1].strip() != "none"
        return gpu_available and has_hip
    except Exception:
        return False


def execute_on_rocm(
    code: str,
    timeout: int = 60,
) -> ExecutionResult:
    """Execute migrated code on a real ROCm GPU in a subprocess.

    Unlike ``execute_migrated_code`` which uses mocks, this function runs
    with the real HIP runtime.  Requires AMD hardware + ROCm stack.

    Args:
        code: The migrated Python source to run.
        timeout: Max execution time in seconds.

    Returns:
        ExecutionResult with status, stdout, stderr, and return code.
    """
    if not detect_rocm_runtime():
        return ExecutionResult(
            status="ERROR",
            stdout="",
            stderr="",
            return_code=-1,
            error_summary="No ROCm runtime detected — cannot validate on GPU",
        )

    with tempfile.TemporaryDirectory() as tmpdir:
        code_file = Path(tmpdir) / "migrated_code.py"
        code_file.write_text(code, encoding="utf-8")

        wrapper = textwrap.dedent(f"""\
            import sys, traceback
            try:
                with open({str(code_file)!r}, "r") as f:
                    source = f.read()
                compiled = compile(source, {str(code_file)!r}, "exec")
                exec(compiled, {{"__name__": "__main__"}})
                print("\\n__GPU_EXECUTION_COMPLETE__")
            except Exception:
                traceback.print_exc()
                sys.exit(1)
        """)
        wrapper_file = Path(tmpdir) / "gpu_wrapper.py"
        wrapper_file.write_text(wrapper, encoding="utf-8")

        logger.info("Executing migrated code on ROCm GPU (timeout=%ds)", timeout)

        try:
            proc = subprocess.run(
                [sys.executable, str(wrapper_file)],
                capture_output=True, text=True,
                timeout=timeout, cwd=tmpdir,
            )
        except subprocess.TimeoutExpired:
            return ExecutionResult(
                status="ERROR", stdout="", stderr="", return_code=-1,
                error_summary=f"GPU execution timed out after {timeout}s",
            )

        if proc.returncode == 0 and "__GPU_EXECUTION_COMPLETE__" in proc.stdout:
            status = "WARN" if proc.stderr.strip() else "PASS"
            return ExecutionResult(
                status=status, stdout=proc.stdout, stderr=proc.stderr,
                return_code=proc.returncode,
                error_summary="Completed with warnings" if status == "WARN" else "",
            )
        else:
            error_lines = proc.stderr.strip().splitlines()
            summary = error_lines[-1] if error_lines else "Unknown GPU error"
            return ExecutionResult(
                status="FAIL", stdout=proc.stdout, stderr=proc.stderr,
                return_code=proc.returncode,
                error_summary=summary,
            )


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

        logger.info("Executing migrated code in sandbox (timeout=%ds)", timeout)

        try:
            proc = subprocess.run(
                [sys.executable, str(wrapper_file)],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=tmpdir,
            )
        except subprocess.TimeoutExpired:
            logger.error("Sandbox execution timed out after %ds", timeout)
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
                logger.info("Sandbox execution completed with warnings")
                logger.debug("Sandbox stderr:\n%s", stderr.strip())
                return ExecutionResult(
                    status="WARN",
                    stdout=stdout,
                    stderr=stderr,
                    return_code=rc,
                    error_summary="Completed with warnings",
                )
            logger.info("Sandbox execution passed")
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
            logger.warning(
                "Sandbox execution failed (rc=%d): %s", rc, summary,
            )
            logger.debug("Sandbox stderr:\n%s", stderr.strip())
            return ExecutionResult(
                status="FAIL",
                stdout=stdout,
                stderr=stderr,
                return_code=rc,
                error_summary=summary,
            )
