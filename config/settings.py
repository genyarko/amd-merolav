"""Application settings loaded from environment variables."""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # API keys
    mistral_api_key: str = ""
    deepseek_api_key: str = ""
    anthropic_api_key: str = ""

    # Planner — DeepSeek-R1 (reasoning)
    planner_base_url: str = "http://localhost:8000/v1"
    planner_model: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"

    # Executor — configurable (defaults to Codestral)
    executor_base_url: str = "https://api.mistral.ai/v1"
    executor_model: str = "codestral-latest"
    executor_api_key: str = ""          # generic key for any executor; falls back to mistral_api_key
    executor_context_limit: int = 0     # 0 = auto-detect from backend

    # Agent settings
    max_rounds: int = 8
    default_backend: str = "mistral"

    # Timeouts (seconds)
    planner_timeout: int = 600
    test_timeout: int = 30

    # Logging
    log_level: str = "WARNING"
    log_file: str = ""

    # ROCm target version (e.g. "6.0") — filters out mappings not available
    rocm_version: str = ""

    # Cache
    cache_dir: str = ".rocm_cache"

    # CLI / UX
    output_format: str = "diff"  # diff | json | markdown | patch
    interactive: bool = False
    watch: bool = False

    # Chunking
    chunk_size: int = 4000  # max tokens per chunk for large-file migration


def get_settings() -> Settings:
    return Settings()
