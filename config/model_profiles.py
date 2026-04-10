"""Model backend profiles for AutoGen LLM configuration."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from config.settings import Settings

CONFIG_LIST_PATH = Path(__file__).parent / "oai_config_list.json"


def get_planner_config(settings: Settings) -> dict[str, Any]:
    """Return AutoGen llm_config for the Planner (DeepSeek-R1)."""
    return {
        "config_list": [
            {
                "model": settings.planner_model,
                "base_url": settings.planner_base_url,
                "api_key": "EMPTY",
                "timeout": settings.planner_timeout,  # configurable via --planner-timeout
                "price": [0, 0],  # self-hosted, suppress ag2 cost warning
            }
        ],
        "temperature": 0.6,  # Higher temp for creative reasoning
    }


def get_executor_config(settings: Settings) -> dict[str, Any]:
    """Return AutoGen llm_config for the Executor (any OpenAI-compatible endpoint)."""
    api_key = settings.executor_api_key or settings.mistral_api_key or "EMPTY"
    return {
        "config_list": [
            {
                "model": settings.executor_model,
                "base_url": settings.executor_base_url,
                "api_key": api_key,
                "price": [0, 0],  # suppress ag2 cost warning
            }
        ],
        "temperature": 0.1,
    }


def _build_config_list(settings: Settings) -> list[dict[str, Any]]:
    """Build the full config list from settings, substituting env vars."""
    return [
        {
            "model": settings.executor_model,
            "base_url": settings.executor_base_url,
            "api_key": "EMPTY",
            "tags": ["self-hosted", "executor"],
        },
        {
            "model": "codestral-latest",
            "base_url": "https://api.mistral.ai/v1",
            "api_key": settings.mistral_api_key,
            "tags": ["mistral", "fallback"],
        },
        {
            "model": "deepseek-coder",
            "base_url": "https://api.deepseek.com/v1",
            "api_key": settings.deepseek_api_key,
            "tags": ["deepseek", "fallback"],
        },
        {
            "model": "claude-sonnet-4-20250514",
            "base_url": "https://api.anthropic.com/v1",
            "api_key": settings.anthropic_api_key,
            "tags": ["claude", "optional"],
        },
    ]


# Mapping from CLI --backend flag to tag filter
BACKEND_TAG_MAP: dict[str, list[str]] = {
    "self-hosted": ["self-hosted"],
    "mistral": ["mistral"],
    "deepseek": ["deepseek"],
    "claude": ["claude"],
}


def get_config_list(backend: str, settings: Settings | None = None) -> list[dict[str, Any]]:
    """Return AutoGen-compatible config_list filtered by backend profile."""
    if settings is None:
        from config.settings import get_settings
        settings = get_settings()

    all_configs = _build_config_list(settings)
    required_tags = set(BACKEND_TAG_MAP.get(backend, [backend]))

    filtered = [
        cfg for cfg in all_configs
        if required_tags & set(cfg.get("tags", []))
    ]

    if not filtered:
        raise ValueError(
            f"No model config found for backend '{backend}'. "
            f"Available backends: {list(BACKEND_TAG_MAP.keys())}"
        )

    # Strip tags before passing to AutoGen (it doesn't expect them)
    for cfg in filtered:
        cfg.pop("tags", None)

    return filtered
