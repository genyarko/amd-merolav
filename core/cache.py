"""File-based migration cache — avoids redundant LLM calls on unchanged files.

Cache layout::

    .rocm_cache/
        manifest.json          — file path → {source_hash, result_hash, timestamp}
        migrations/
            <hash>.json        — cached MigrationResult (code, applied, remaining)
        planner/
            <hash>.txt         — cached Planner output (plain text plan)
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from core.logging import get_logger

logger = get_logger(__name__)

_DEFAULT_CACHE_DIR = ".rocm_cache"


class MigrationCache:
    """File-based cache for migration results and planner output."""

    def __init__(self, cache_dir: str | Path = _DEFAULT_CACHE_DIR) -> None:
        self.root = Path(cache_dir)
        self._migrations_dir = self.root / "migrations"
        self._planner_dir = self.root / "planner"
        self._manifest_path = self.root / "manifest.json"

    # --- Setup ---

    def _ensure_dirs(self) -> None:
        self._migrations_dir.mkdir(parents=True, exist_ok=True)
        self._planner_dir.mkdir(parents=True, exist_ok=True)

    # --- Hash helpers ---

    @staticmethod
    def hash_source(source: str, backend: str = "") -> str:
        """SHA-256 hash of source code + backend identifier."""
        payload = f"{source}\x00{backend}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @staticmethod
    def hash_planner_key(source: str, remaining_issues: list[dict]) -> str:
        """SHA-256 hash for planner cache — source + remaining issues."""
        issues_str = json.dumps(remaining_issues, sort_keys=True)
        payload = f"{source}\x00{issues_str}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    # --- Manifest (incremental tracking) ---

    def _load_manifest(self) -> dict[str, Any]:
        if self._manifest_path.exists():
            try:
                return json.loads(self._manifest_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                logger.warning("Corrupt cache manifest, starting fresh")
                return {}
        return {}

    def _save_manifest(self, manifest: dict[str, Any]) -> None:
        self._ensure_dirs()
        self._manifest_path.write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )

    def is_unchanged(self, file_path: str, source: str, backend: str) -> bool:
        """Check if a file's source is unchanged since last cached migration."""
        manifest = self._load_manifest()
        entry = manifest.get(file_path)
        if not entry:
            return False
        current_hash = self.hash_source(source, backend)
        return entry.get("source_hash") == current_hash

    # --- Migration result cache ---

    def get_migration(self, source: str, backend: str) -> dict | None:
        """Retrieve a cached migration result, or None if not cached."""
        h = self.hash_source(source, backend)
        path = self._migrations_dir / f"{h}.json"
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                logger.info("Cache hit for migration (hash=%s...)", h[:12])
                return data
            except (json.JSONDecodeError, OSError):
                logger.warning("Corrupt cache entry %s, ignoring", h[:12])
                return None
        return None

    def put_migration(
        self,
        file_path: str,
        source: str,
        backend: str,
        migrated_code: str,
        applied: list[dict],
        remaining: list[dict],
    ) -> None:
        """Store a migration result in the cache."""
        self._ensure_dirs()
        h = self.hash_source(source, backend)
        data = {
            "migrated_code": migrated_code,
            "applied": applied,
            "remaining": remaining,
            "cached_at": time.time(),
        }
        path = self._migrations_dir / f"{h}.json"
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")

        # Update manifest
        manifest = self._load_manifest()
        manifest[file_path] = {
            "source_hash": h,
            "result_hash": hashlib.sha256(migrated_code.encode()).hexdigest()[:16],
            "timestamp": time.time(),
            "backend": backend,
        }
        self._save_manifest(manifest)
        logger.info("Cached migration for %s (hash=%s...)", file_path, h[:12])

    # --- Planner cache ---

    def get_planner(self, source: str, remaining_issues: list[dict]) -> str | None:
        """Retrieve a cached planner plan, or None if not cached."""
        h = self.hash_planner_key(source, remaining_issues)
        path = self._planner_dir / f"{h}.txt"
        if path.exists():
            try:
                plan = path.read_text(encoding="utf-8")
                logger.info("Cache hit for planner (hash=%s...)", h[:12])
                return plan
            except OSError:
                return None
        return None

    def put_planner(self, source: str, remaining_issues: list[dict], plan: str) -> None:
        """Store a planner result in the cache."""
        self._ensure_dirs()
        h = self.hash_planner_key(source, remaining_issues)
        path = self._planner_dir / f"{h}.txt"
        path.write_text(plan, encoding="utf-8")
        logger.info("Cached planner output (hash=%s...)", h[:12])

    # --- Cache management ---

    def clear(self) -> int:
        """Delete all cached data. Returns number of files removed."""
        count = 0
        if self.root.exists():
            for p in self.root.rglob("*"):
                if p.is_file():
                    p.unlink()
                    count += 1
            # Remove empty dirs
            for p in sorted(self.root.rglob("*"), reverse=True):
                if p.is_dir():
                    try:
                        p.rmdir()
                    except OSError:
                        pass
            try:
                self.root.rmdir()
            except OSError:
                pass
        logger.info("Cleared cache: %d files removed", count)
        return count

    def stats(self) -> dict[str, int]:
        """Return cache statistics."""
        manifest = self._load_manifest()
        migration_count = sum(1 for _ in self._migrations_dir.glob("*.json")) if self._migrations_dir.exists() else 0
        planner_count = sum(1 for _ in self._planner_dir.glob("*.txt")) if self._planner_dir.exists() else 0
        return {
            "tracked_files": len(manifest),
            "cached_migrations": migration_count,
            "cached_plans": planner_count,
        }
