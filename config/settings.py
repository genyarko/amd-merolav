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

    # Executor — Mistral API (Codestral)
    executor_base_url: str = "https://api.mistral.ai/v1"
    executor_model: str = "codestral-latest"

    # Agent settings
    max_rounds: int = 6
    default_backend: str = "mistral"


def get_settings() -> Settings:
    return Settings()
