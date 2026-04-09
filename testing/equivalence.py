"""Semantic equivalence checking — compare original and migrated code on CPU.

Runs both versions with identical inputs on CPU and compares outputs using
``torch.allclose`` to catch behavioral differences introduced by migration.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
import textwrap
from dataclasses import dataclass, field
from pathlib import Path

from core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class EquivalenceIssue:
    operation: str
    expected: str
    actual: str
    tolerance: float


@dataclass
class EquivalenceResult:
    passed: bool
    issues: list[EquivalenceIssue] = field(default_factory=list)
    error: str = ""
    skipped: bool = False
    skip_reason: str = ""


_EQUIV_RUNNER = textwrap.dedent("""\
    import sys, os, json, traceback, io

    original_path = sys.argv[1]
    migrated_path = sys.argv[2]

    results = {"passed": True, "issues": [], "error": ""}

    try:
        import torch
    except ImportError:
        results["passed"] = True
        results["skipped"] = True
        results["skip_reason"] = "torch not available — skipping equivalence check"
        print(json.dumps(results))
        sys.exit(0)

    import re as _re

    # Helper: capture tensor outputs from exec'd code
    def run_code(path):
        # Seed RNG for deterministic comparison between original and migrated
        torch.manual_seed(42)
        if hasattr(torch, 'use_deterministic_algorithms'):
            try:
                torch.use_deterministic_algorithms(True, warn_only=True)
            except Exception:
                pass

        ns = {"__name__": "__main__", "torch": torch}
        with open(path, "r") as f:
            source = f.read()

        # Force CPU execution to avoid GPU dependency
        source = source.replace('.cuda()', '.cpu()')
        source = source.replace("device='cuda'", "device='cpu'")
        source = source.replace('device="cuda"', 'device="cpu"')
        source = source.replace("torch.device('cuda')", "torch.device('cpu')")
        source = source.replace('torch.device("cuda")', 'torch.device("cpu")')
        source = source.replace("'cuda:0'", "'cpu'")
        source = source.replace('"cuda:0"', '"cpu"')
        source = source.replace("'cuda'", "'cpu'")  # last resort
        source = source.replace('"cuda"', '"cpu"')

        # Stub out torch.cuda GPU-only utility calls that fail on CPU
        source = _re.sub(
            r"torch\.cuda\.memory_allocated\([^)]*\)", "0", source)
        source = _re.sub(
            r"torch\.cuda\.memory_reserved\([^)]*\)", "0", source)
        source = _re.sub(
            r"torch\.cuda\.max_memory_allocated\([^)]*\)", "0", source)
        source = _re.sub(
            r"torch\.cuda\.reset_peak_memory_stats\([^)]*\)", "None", source)
        source = _re.sub(
            r"torch\.cuda\.synchronize\([^)]*\)", "None", source)
        source = _re.sub(
            r"torch\.cuda\.empty_cache\([^)]*\)", "None", source)

        # Suppress stdout from exec'd code so print() calls
        # don't pollute the JSON output on stdout
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        try:
            compiled = compile(source, path, "exec")
            exec(compiled, ns)
        except Exception as e:
            return None, str(e)
        finally:
            sys.stdout = old_stdout

        # Collect all tensor values from the namespace
        tensors = {}
        for k, v in ns.items():
            if k.startswith("_"):
                continue
            if isinstance(v, torch.Tensor):
                tensors[k] = v.detach().cpu()
        return tensors, None

    # Run original
    orig_tensors, orig_err = run_code(original_path)
    if orig_err:
        results["error"] = f"Original code failed: {orig_err}"
        results["passed"] = False
        print(json.dumps(results))
        sys.exit(0)

    # Run migrated
    mig_tensors, mig_err = run_code(migrated_path)
    if mig_err:
        results["error"] = f"Migrated code failed: {mig_err}"
        results["passed"] = False
        print(json.dumps(results))
        sys.exit(0)

    if orig_tensors is None or mig_tensors is None:
        results["error"] = "Could not collect tensor outputs"
        results["passed"] = False
        print(json.dumps(results))
        sys.exit(0)

    # Compare shared tensor variables
    atol = 1e-5
    rtol = 1e-4
    shared_keys = set(orig_tensors.keys()) & set(mig_tensors.keys())
    for key in sorted(shared_keys):
        orig_t = orig_tensors[key]
        mig_t = mig_tensors[key]
        if orig_t.shape != mig_t.shape:
            results["passed"] = False
            results["issues"].append({
                "operation": f"tensor '{key}'",
                "expected": f"shape {list(orig_t.shape)}",
                "actual": f"shape {list(mig_t.shape)}",
                "tolerance": 0.0,
            })
        elif not torch.allclose(orig_t.float(), mig_t.float(), atol=atol, rtol=rtol):
            max_diff = (orig_t.float() - mig_t.float()).abs().max().item()
            results["passed"] = False
            results["issues"].append({
                "operation": f"tensor '{key}'",
                "expected": f"values from original (sample: {orig_t.flatten()[:3].tolist()})",
                "actual": f"values from migrated (sample: {mig_t.flatten()[:3].tolist()})",
                "tolerance": max_diff,
            })

    print(json.dumps(results))
""")


def check_equivalence(
    original_code: str,
    migrated_code: str,
    timeout: int = 30,
) -> EquivalenceResult:
    """Run original and migrated code on CPU, compare tensor outputs.

    Both code snippets are exec'd with all device references forced to CPU.
    Tensor variables in the global namespace are compared with torch.allclose.

    Args:
        original_code: The original CUDA Python code.
        migrated_code: The migrated ROCm Python code.
        timeout: Max execution time in seconds.

    Returns:
        EquivalenceResult indicating whether outputs match.
    """
    logger.info("Running semantic equivalence check")

    with tempfile.TemporaryDirectory() as tmpdir:
        orig_file = Path(tmpdir) / "original.py"
        mig_file = Path(tmpdir) / "migrated.py"
        runner_file = Path(tmpdir) / "equiv_runner.py"

        orig_file.write_text(original_code, encoding="utf-8")
        mig_file.write_text(migrated_code, encoding="utf-8")
        runner_file.write_text(_EQUIV_RUNNER, encoding="utf-8")

        try:
            proc = subprocess.run(
                [sys.executable, str(runner_file),
                 str(orig_file), str(mig_file)],
                capture_output=True, text=True,
                timeout=timeout, cwd=tmpdir,
            )
        except subprocess.TimeoutExpired:
            return EquivalenceResult(
                passed=False,
                error=f"Equivalence check timed out after {timeout}s",
            )

        if proc.returncode != 0:
            return EquivalenceResult(
                passed=False,
                error=f"Runner failed: {proc.stderr.strip().splitlines()[-1] if proc.stderr.strip() else 'unknown'}",
            )

        try:
            import json
            # The runner prints JSON as the last line, but the executed code
            # may also produce print() output (e.g. "Memory allocated: 0").
            # Try parsing the full output first; if that fails, scan lines
            # in reverse for the first valid JSON object.
            raw = proc.stdout.strip()
            data = None
            try:
                data = json.loads(raw)
            except (json.JSONDecodeError, ValueError):
                for line in reversed(raw.splitlines()):
                    line = line.strip()
                    if line.startswith("{"):
                        try:
                            data = json.loads(line)
                            break
                        except (json.JSONDecodeError, ValueError):
                            continue
            if data is None:
                return EquivalenceResult(
                    passed=False,
                    error=f"Could not parse runner output: {raw[:200]}",
                )
        except Exception:
            return EquivalenceResult(
                passed=False,
                error=f"Could not parse runner output: {proc.stdout[:200]}",
            )

        if data.get("skipped"):
            return EquivalenceResult(
                passed=True, skipped=True,
                skip_reason=data.get("skip_reason", ""),
            )

        issues = [
            EquivalenceIssue(
                operation=iss["operation"],
                expected=iss["expected"],
                actual=iss["actual"],
                tolerance=iss.get("tolerance", 0.0),
            )
            for iss in data.get("issues", [])
        ]

        return EquivalenceResult(
            passed=data.get("passed", False),
            issues=issues,
            error=data.get("error", ""),
        )
