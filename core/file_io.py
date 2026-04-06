"""Safe file I/O utilities."""

from __future__ import annotations

from pathlib import Path


def read_source_file(path: str | Path) -> str:
    """Read a Python source file and return its contents."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Source file not found: {p}")
    if not p.suffix == ".py":
        raise ValueError(f"Expected a .py file, got: {p.suffix}")
    return p.read_text(encoding="utf-8")


def write_output_file(path: str | Path, content: str) -> Path:
    """Write migrated code to an output file."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return p


def collect_python_files(path: str | Path) -> list[Path]:
    """Collect all .py files from a file path or directory."""
    p = Path(path)
    if p.is_file():
        if p.suffix != ".py":
            raise ValueError(f"Expected a .py file, got: {p.suffix}")
        return [p]
    if p.is_dir():
        return sorted(p.rglob("*.py"))
    raise FileNotFoundError(f"Path not found: {p}")
