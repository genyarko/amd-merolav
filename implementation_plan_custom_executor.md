# Implementation Plan: Configurable Executor Backend

## Goal

Let users choose their own code executor (e.g., Gemma 4, Qwen, any OpenAI-compatible endpoint) instead of being locked to Mistral Codestral. The planner remains DeepSeek-R1 self-hosted on MI300X.

## Current Architecture

The system has two LLM roles:

1. **Planner** (DeepSeek-R1-32B) — configured via `PLANNER_BASE_URL` / `PLANNER_MODEL` in `.env`
2. **Executor** (Codestral) — hardcoded to Mistral API via `EXECUTOR_BASE_URL` / `EXECUTOR_MODEL` and the `--backend` CLI flag

The `--backend` flag currently controls **both** the executor model and its API key/URL routing. It maps to a fixed set: `self-hosted | mistral | deepseek | claude`.

### Files That Need Changes

| File | What changes |
|------|-------------|
| `config/settings.py` | Add `executor_api_key` field, decouple executor key from backend map |
| `config/model_profiles.py` | Update `get_executor_config()` to use generic executor key; add context limits for custom executors |
| `cli/main.py` | Add `--executor-url` and `--executor-model` CLI options |
| `agents/orchestrator.py` | Use executor-specific settings instead of backend-derived config |
| `core/context_budget.py` | Add fallback context limit for unknown backends |
| `.env.example` | Document new executor variables |
| `README.md` | Update backend/executor docs |
| `tests/` | Update existing tests, add new tests for custom executor |

---

## Steps

### Step 1: Add executor-specific settings to `config/settings.py`

Add a generic `executor_api_key` field so the executor isn't tied to `mistral_api_key`.

```python
# Executor — configurable (defaults to Codestral)
executor_base_url: str = "https://api.mistral.ai/v1"
executor_model: str = "codestral-latest"
executor_api_key: str = ""          # NEW — generic key for any executor
executor_context_limit: int = 0     # NEW — 0 means auto-detect from backend
```

Keep `mistral_api_key` for backward compatibility but fall back:
- If `executor_api_key` is set, use it.
- Else fall back to `mistral_api_key` (existing behavior).

### Step 2: Add `--executor-url`, `--executor-model`, and `--executor-key` CLI options

In `cli/main.py`, add three new options:

```python
executor_url: Optional[str] = typer.Option(
    None, "--executor-url",
    help="Executor LLM server URL (overrides .env EXECUTOR_BASE_URL)",
)
executor_model: Optional[str] = typer.Option(
    None, "--executor-model",
    help="Executor model name (overrides .env EXECUTOR_MODEL)",
)
executor_key: Optional[str] = typer.Option(
    None, "--executor-key",
    help="API key for the executor backend (overrides .env EXECUTOR_API_KEY)",
)
```

Wire them into `settings` overrides in the `main()` function body (same pattern as `planner_url`):

```python
if executor_url is not None:
    settings.executor_base_url = executor_url
if executor_model is not None:
    settings.executor_model = executor_model
if executor_key is not None:
    settings.executor_api_key = executor_key
```

### Step 3: Update `get_executor_config()` in `config/model_profiles.py`

Change the executor config builder to use the new generic key with Mistral fallback:

```python
def get_executor_config(settings: Settings) -> dict[str, Any]:
    """Return AutoGen llm_config for the Executor."""
    api_key = settings.executor_api_key or settings.mistral_api_key or "EMPTY"
    return {
        "config_list": [
            {
                "model": settings.executor_model,
                "base_url": settings.executor_base_url,
                "api_key": api_key,
                "price": [0, 0],
            }
        ],
        "temperature": 0.1,
    }
```

### Step 4: Update `agents/orchestrator.py` to always use executor config

Currently the orchestrator has two code paths for loading the executor LLM config:

```python
# Current (lines 110-114):
if settings.default_backend == "self-hosted":
    llm_config = get_executor_config(settings)
else:
    config_list = get_config_list(settings.default_backend, settings)
    llm_config = {"config_list": config_list, "temperature": 0.1}
```

Simplify to always use `get_executor_config()`:

```python
llm_config = get_executor_config(settings)
```

This works because the executor URL/model/key are now fully configurable via CLI or `.env`. The `--backend` flag becomes a convenience shortcut that sets these values (see Step 5).

Apply the same change to `_migrate_single_chunk()` (lines 418-422).

### Step 5: Make `--backend` a shortcut that sets executor defaults

Keep `--backend` for convenience but make it resolve to executor settings at startup. In `cli/main.py`, after settings are built:

```python
# Resolve --backend into executor settings (if not explicitly overridden)
_BACKEND_DEFAULTS = {
    "mistral": {
        "base_url": "https://api.mistral.ai/v1",
        "model": "codestral-latest",
        "key_attr": "mistral_api_key",
    },
    "deepseek": {
        "base_url": "https://api.deepseek.com/v1",
        "model": "deepseek-coder",
        "key_attr": "deepseek_api_key",
    },
    "claude": {
        "base_url": "https://api.anthropic.com/v1",
        "model": "claude-sonnet-4-20250514",
        "key_attr": "anthropic_api_key",
    },
    "self-hosted": {
        "base_url": settings.executor_base_url,
        "model": settings.executor_model,
        "key_attr": None,
    },
}

# Only apply backend defaults if user didn't explicitly set --executor-url/--executor-model
if executor_url is None and backend in _BACKEND_DEFAULTS:
    defaults = _BACKEND_DEFAULTS[backend]
    settings.executor_base_url = defaults["base_url"]
    settings.executor_model = defaults["model"]
    if defaults["key_attr"] and not settings.executor_api_key:
        settings.executor_api_key = getattr(settings, defaults["key_attr"], "")
```

This means `--backend mistral` works exactly as before, but `--executor-url http://myserver:8002/v1 --executor-model google/gemma-4-26B-A4B-it` overrides it.

### Step 6: Update context budget for custom executors

In `core/context_budget.py`, add a fallback and support for the new `executor_context_limit` setting:

```python
_DEFAULT_CONTEXT_LIMITS: dict[str, int] = {
    "mistral": 28000,
    "deepseek": 28000,
    "claude": 180000,
    "self-hosted": 12000,
}

def _get_context_limit(backend: str, settings: Settings | None = None) -> int:
    """Get context limit, respecting user override."""
    if settings and settings.executor_context_limit > 0:
        return settings.executor_context_limit
    return _DEFAULT_CONTEXT_LIMITS.get(backend, 12000)  # safe fallback
```

### Step 7: Update `.env.example`

```env
# Executor — code generation (default: Mistral Codestral)
# Override these to use any OpenAI-compatible endpoint as executor
EXECUTOR_BASE_URL=https://api.mistral.ai/v1
EXECUTOR_MODEL=codestral-latest
EXECUTOR_API_KEY=               # generic key; falls back to MISTRAL_API_KEY if empty

# Context limit override (0 = auto-detect from backend)
EXECUTOR_CONTEXT_LIMIT=0
```

### Step 8: Update health check and key check

In `cli/main.py`, update `_check_backend_key()` and `_check_backend_health()` to work with the new executor settings:

- If `executor_api_key` is set, skip the backend-specific key check.
- Health check should use `settings.executor_base_url` directly.

### Step 9: Update tests

- `tests/test_model_profiles.py` — test `get_executor_config()` with custom URL/model/key
- `tests/test_e2e.py` — verify `--executor-url` / `--executor-model` flags override correctly
- `tests/test_context_budget.py` — verify fallback context limits for unknown backends

### Step 10: Update README.md

Add a "Custom Executor" section:

```markdown
### Custom Executor

Use any OpenAI-compatible endpoint as the code executor:

    # Self-hosted Gemma 4 on port 8002
    rocm-migrate input.py --executor-url http://10.128.0.2:8002/v1 --executor-model google/gemma-4-31B-it

    # Ollama local
    rocm-migrate input.py --executor-url http://localhost:11434/v1 --executor-model codellama

    # Any OpenAI-compatible API
    rocm-migrate input.py --executor-url https://api.together.xyz/v1 --executor-model meta-llama/Llama-3-70b --executor-key $TOGETHER_KEY

The planner (DeepSeek-R1) runs independently — only the executor is swapped.
```

---

## Example Usage After Implementation

```bash
# Default (unchanged behavior)
rocm-migrate input.py --backend mistral

# Use Gemma 4 31B as executor (self-hosted)
rocm-migrate input.py --executor-url http://10.128.0.2:8002/v1 \
  --executor-model google/gemma-4-31B-it

# Use Claude as executor with DeepSeek planner
rocm-migrate input.py --executor-url https://api.anthropic.com/v1 \
  --executor-model claude-sonnet-4-20250514 \
  --executor-key $ANTHROPIC_API_KEY

# Use Ollama locally
rocm-migrate input.py --executor-url http://localhost:11434/v1 \
  --executor-model qwen2.5-coder:32b
```

---

## Backward Compatibility

- `--backend mistral|deepseek|claude|self-hosted` continues to work exactly as before.
- Existing `.env` files with `MISTRAL_API_KEY` + `EXECUTOR_BASE_URL` + `EXECUTOR_MODEL` still work.
- `--executor-url` / `--executor-model` take precedence over `--backend` when both are specified.
- No breaking changes to the planner pipeline.
