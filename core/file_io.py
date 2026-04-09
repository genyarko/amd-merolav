"""Safe file I/O utilities."""

from __future__ import annotations

from pathlib import Path


def read_source_file(path: str | Path) -> str:
    """Read a Python source file and return its contents."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Source file not found: {p}")
    if p.suffix not in (".py", ".cu", ".cuh"):
        raise ValueError(f"Expected a .py/.cu/.cuh file, got: {p.suffix}")
    return p.read_text(encoding="utf-8")


def write_output_file(path: str | Path, content: str) -> Path:
    """Write migrated code to an output file."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return p


_SUPPORTED_EXTENSIONS = {".py", ".cu", ".cuh"}


def collect_python_files(path: str | Path) -> list[Path]:
    """Collect all supported files (.py, .cu, .cuh) from a file path or directory."""
    p = Path(path)
    if p.is_file():
        if p.suffix not in _SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Expected a {'/'.join(_SUPPORTED_EXTENSIONS)} file, got: {p.suffix}"
            )
        return [p]
    if p.is_dir():
        files: list[Path] = []
        for ext in _SUPPORTED_EXTENSIONS:
            files.extend(p.rglob(f"*{ext}"))
        return sorted(files)
    raise FileNotFoundError(f"Path not found: {p}")
