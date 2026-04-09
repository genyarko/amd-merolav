"""Tests for core.audit — migration audit log."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from core.audit import append_audit_entry, format_history, read_audit_log


def _make_applied(line=1, rule="test_rule", confidence=0.95):
    return SimpleNamespace(line=line, rule=rule, confidence=confidence)


def _make_remaining(line=5, symbol="cuda_fn", reason="needs manual review", confidence=0.4):
    return SimpleNamespace(line=line, symbol=symbol, reason=reason, confidence=confidence)


def _make_optimization(priority=2, category="memory", suggestion="use HIP streams", url=""):
    return SimpleNamespace(priority=priority, category=category, suggestion=suggestion, url=url)


# --- append_audit_entry ---

def test_append_creates_file(tmp_path: Path):
    log_path = tmp_path / "log.jsonl"
    entry = append_audit_entry(
        file_path="test.py",
        backend="mistral",
        applied=[_make_applied()],
        remaining=[],
        optimizations=[],
        overall_confidence=0.95,
        log_path=log_path,
    )
    assert log_path.exists()
    assert entry["file"] == "test.py"
    assert entry["applied_count"] == 1

def test_append_is_valid_jsonl(tmp_path: Path):
    log_path = tmp_path / "log.jsonl"
    append_audit_entry("a.py", "mistral", [], [], [], 0.9, log_path=log_path)
    append_audit_entry("b.py", "deepseek", [], [], [], 0.8, log_path=log_path)

    lines = log_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    for line in lines:
        data = json.loads(line)
        assert "timestamp" in data
        assert "file" in data

def test_append_records_all_fields(tmp_path: Path):
    log_path = tmp_path / "log.jsonl"
    entry = append_audit_entry(
        file_path="model.py",
        backend="claude",
        applied=[_make_applied(), _make_applied(line=10)],
        remaining=[_make_remaining()],
        optimizations=[_make_optimization()],
        overall_confidence=0.72,
        validation_passed=True,
        agent_used=True,
        warnings=["some warning"],
        log_path=log_path,
    )
    assert entry["applied_count"] == 2
    assert entry["remaining_count"] == 1
    assert entry["optimization_count"] == 1
    assert entry["overall_confidence"] == 0.72
    assert entry["validation_passed"] is True
    assert entry["agent_used"] is True
    assert entry["warnings"] == ["some warning"]
    assert len(entry["applied"]) == 2
    assert entry["applied"][0]["confidence"] == 0.95
    assert entry["remaining"][0]["symbol"] == "cuda_fn"

def test_append_creates_parent_dirs(tmp_path: Path):
    log_path = tmp_path / "sub" / "dir" / "log.jsonl"
    append_audit_entry("x.py", "m", [], [], [], 0.5, log_path=log_path)
    assert log_path.exists()


# --- read_audit_log ---

def test_read_nonexistent_log(tmp_path: Path):
    assert read_audit_log(log_path=tmp_path / "nope.jsonl") == []

def test_read_returns_entries(tmp_path: Path):
    log_path = tmp_path / "log.jsonl"
    append_audit_entry("a.py", "m", [], [], [], 0.9, log_path=log_path)
    append_audit_entry("b.py", "m", [], [], [], 0.8, log_path=log_path)
    entries = read_audit_log(log_path=log_path)
    assert len(entries) == 2
    assert entries[0]["file"] == "a.py"
    assert entries[1]["file"] == "b.py"

def test_read_with_file_filter(tmp_path: Path):
    log_path = tmp_path / "log.jsonl"
    append_audit_entry("model.py", "m", [], [], [], 0.9, log_path=log_path)
    append_audit_entry("train.py", "m", [], [], [], 0.8, log_path=log_path)
    entries = read_audit_log(log_path=log_path, file_filter="model")
    assert len(entries) == 1
    assert entries[0]["file"] == "model.py"

def test_read_skips_malformed_lines(tmp_path: Path):
    log_path = tmp_path / "log.jsonl"
    log_path.write_text('{"file": "a.py"}\nnot json\n{"file": "b.py"}\n', encoding="utf-8")
    entries = read_audit_log(log_path=log_path)
    assert len(entries) == 2


# --- format_history ---

def test_format_history_empty():
    assert "No migration history" in format_history([])

def test_format_history_table():
    entries = [
        {
            "timestamp": "2025-01-01T00:00:00+00:00",
            "file": "model.py",
            "backend": "mistral",
            "applied_count": 5,
            "remaining_count": 2,
            "overall_confidence": 0.85,
            "validation_passed": True,
        },
        {
            "timestamp": "2025-01-02T12:00:00+00:00",
            "file": "train.py",
            "backend": "deepseek",
            "applied_count": 3,
            "remaining_count": 0,
            "overall_confidence": 0.95,
            "validation_passed": None,
        },
    ]
    output = format_history(entries)
    assert "model.py" in output
    assert "train.py" in output
    assert "mistral" in output
    assert "pass" in output  # validation_passed=True
    assert "-" in output     # validation_passed=None

def test_format_history_truncates_long_paths():
    entries = [{
        "timestamp": "2025-01-01T00:00:00",
        "file": "a" * 50 + "/very_long_path.py",
        "backend": "m",
        "applied_count": 0,
        "remaining_count": 0,
        "overall_confidence": 0.5,
        "validation_passed": False,
    }]
    output = format_history(entries)
    assert "..." in output
    assert "fail" in output
