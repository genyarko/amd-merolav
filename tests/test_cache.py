"""Tests for core.cache — MigrationCache."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.cache import MigrationCache


@pytest.fixture
def cache(tmp_path: Path) -> MigrationCache:
    """Create a MigrationCache in a temp directory."""
    return MigrationCache(cache_dir=tmp_path / ".rocm_cache")


# --- hash helpers ---

def test_hash_source_deterministic():
    h1 = MigrationCache.hash_source("import torch", "mistral")
    h2 = MigrationCache.hash_source("import torch", "mistral")
    assert h1 == h2

def test_hash_source_differs_by_backend():
    h1 = MigrationCache.hash_source("code", "mistral")
    h2 = MigrationCache.hash_source("code", "deepseek")
    assert h1 != h2

def test_hash_source_differs_by_code():
    h1 = MigrationCache.hash_source("import torch", "mistral")
    h2 = MigrationCache.hash_source("import numpy", "mistral")
    assert h1 != h2

def test_hash_planner_key_deterministic():
    issues = [{"line": 1, "symbol": "cuda", "reason": "test"}]
    h1 = MigrationCache.hash_planner_key("code", issues)
    h2 = MigrationCache.hash_planner_key("code", issues)
    assert h1 == h2

def test_hash_planner_key_differs_by_issues():
    h1 = MigrationCache.hash_planner_key("code", [{"line": 1}])
    h2 = MigrationCache.hash_planner_key("code", [{"line": 2}])
    assert h1 != h2


# --- is_unchanged ---

def test_is_unchanged_no_manifest(cache: MigrationCache):
    assert cache.is_unchanged("foo.py", "code", "mistral") is False

def test_is_unchanged_after_put(cache: MigrationCache):
    cache.put_migration("foo.py", "code", "mistral", "migrated", [], [])
    assert cache.is_unchanged("foo.py", "code", "mistral") is True

def test_is_unchanged_after_source_change(cache: MigrationCache):
    cache.put_migration("foo.py", "code_v1", "mistral", "migrated", [], [])
    assert cache.is_unchanged("foo.py", "code_v2", "mistral") is False


# --- migration cache ---

def test_get_migration_miss(cache: MigrationCache):
    assert cache.get_migration("code", "mistral") is None

def test_put_and_get_migration(cache: MigrationCache):
    applied = [{"line": 1, "rule": "test", "confidence": 1.0}]
    remaining = [{"line": 5, "symbol": "x", "reason": "manual"}]
    cache.put_migration("f.py", "src", "mistral", "migrated_src", applied, remaining)

    result = cache.get_migration("src", "mistral")
    assert result is not None
    assert result["migrated_code"] == "migrated_src"
    assert result["applied"] == applied
    assert result["remaining"] == remaining
    assert "cached_at" in result

def test_put_migration_updates_manifest(cache: MigrationCache):
    cache.put_migration("a.py", "code", "mistral", "out", [], [])
    manifest = json.loads(cache._manifest_path.read_text(encoding="utf-8"))
    assert "a.py" in manifest
    assert "source_hash" in manifest["a.py"]
    assert "timestamp" in manifest["a.py"]


# --- planner cache ---

def test_get_planner_miss(cache: MigrationCache):
    assert cache.get_planner("code", []) is None

def test_put_and_get_planner(cache: MigrationCache):
    issues = [{"line": 1, "symbol": "cuda", "reason": "needs porting"}]
    cache.put_planner("code", issues, "Step 1: do thing\nStep 2: done")

    plan = cache.get_planner("code", issues)
    assert plan == "Step 1: do thing\nStep 2: done"

def test_planner_cache_differs_by_issues(cache: MigrationCache):
    cache.put_planner("code", [{"line": 1}], "plan A")
    assert cache.get_planner("code", [{"line": 2}]) is None


# --- clear ---

def test_clear_empty_cache(cache: MigrationCache):
    assert cache.clear() == 0

def test_clear_removes_files(cache: MigrationCache):
    cache.put_migration("a.py", "code", "m", "out", [], [])
    cache.put_planner("code", [], "plan")
    removed = cache.clear()
    assert removed >= 3  # manifest + migration + planner
    assert not cache.root.exists()


# --- stats ---

def test_stats_empty(cache: MigrationCache):
    s = cache.stats()
    assert s["tracked_files"] == 0
    assert s["cached_migrations"] == 0
    assert s["cached_plans"] == 0

def test_stats_after_puts(cache: MigrationCache):
    cache.put_migration("a.py", "c1", "m", "o1", [], [])
    cache.put_migration("b.py", "c2", "m", "o2", [], [])
    cache.put_planner("c1", [], "plan1")
    s = cache.stats()
    assert s["tracked_files"] == 2
    assert s["cached_migrations"] == 2
    assert s["cached_plans"] == 1


# --- corrupt data handling ---

def test_corrupt_manifest_recovers(cache: MigrationCache):
    cache._ensure_dirs()
    cache._manifest_path.write_text("not json!", encoding="utf-8")
    # Should not raise — returns empty manifest
    assert cache.is_unchanged("foo.py", "code", "m") is False

def test_corrupt_migration_returns_none(cache: MigrationCache):
    cache._ensure_dirs()
    h = MigrationCache.hash_source("code", "m")
    (cache._migrations_dir / f"{h}.json").write_text("bad json", encoding="utf-8")
    assert cache.get_migration("code", "m") is None
