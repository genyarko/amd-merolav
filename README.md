# CUDA→ROCm Migration Agent

Automatically migrates NVIDIA CUDA Python/PyTorch code to AMD ROCm using a two-phase pipeline: a rule-based pre-pass handles deterministic high-confidence replacements, then a multi-agent LLM loop (DeepSeek-R1 as Planner + Mistral Codestral as Executor) reasons through and implements the complex patterns that rules can't handle — custom kernels, profiling APIs, cuDNN tuning flags, and mixed-precision idioms.

Built for the [lablab.ai AMD Hackathon](https://lablab.ai) using an AMD MI300X GPU on DigitalOcean Developer Cloud.

---

## Architecture

```
User CLI invocation
      │
      ▼
┌─────────────┐
│  CLI (main) │──── reads .py file(s)
└──────┬──────┘
       │
       ▼
┌──────────────┐
│   Analyzer   │──── static scan: finds CUDA symbols, imports, device refs
└──────┬───────┘
       │ analysis_report (CUDA usages with line numbers)
       ▼
┌──────────────┐
│   Migrator   │──── rule-based pre-pass (~30 deterministic mappings)
└──────┬───────┘     cuDNN→MIOpen, CUDA_VISIBLE_DEVICES→HIP, imports, etc.
       │ partially_migrated_code + remaining_issues
       ▼
┌────────────────────────────────────────────┐
│          Two-Phase LLM Pipeline            │
│                                            │
│  Phase 1 — Planner (DeepSeek-R1)          │
│    Reasons through remaining issues        │
│    Produces a numbered migration plan      │
│                                            │
│  Phase 2 — Executor GroupChat             │
│  ┌──────────┐  ┌──────────┐  ┌────────┐  │
│  │ Executor │→ │ Reviewer │→ │ Tester │  │
│  │Codestral │  │Codestral │  │  (AST) │  │
│  └──────────┘  └──────────┘  └────────┘  │
│  Terminates on ALL_TESTS_PASSED           │
└──────────┬─────────────────────────────────┘
           │
           ▼
┌──────────────┐
│    Differ    │──── unified diff: original vs migrated
└──────┬───────┘
       ▼
  CLI output: diff + AMD optimization suggestions + validation report
```

---

## Quick Start

```bash
# Install dependencies
pip install ag2 typer rich pydantic-settings python-dotenv openai

# Configure
cp .env.example .env
# Edit .env with your Mistral API key (free tier works)

# Migrate a file
python -m cli.main your_cuda_script.py

# With verbose agent conversation output
python -m cli.main your_cuda_script.py --verbose

# Force agents even if rule-based pass handles everything (for demo)
python -m cli.main your_cuda_script.py --verbose --force-agents

# Rule-based only (no LLM, instant)
python -m cli.main your_cuda_script.py --no-agent
```

Output is written to `./rocm_output/` by default.

---

## Supported Backends

| Backend | Model | Use Case |
|---------|-------|----------|
| `mistral` (default) | Mistral Codestral | Free API, great for code |
| `self-hosted` | Any vLLM-served model | MI300X or local GPU |
| `deepseek` | DeepSeek Coder | Alternative API |
| `claude` | Claude Sonnet | Optional fallback |

The **Planner** always uses the self-hosted vLLM endpoint (`PLANNER_BASE_URL`) — set this to your MI300X or any OpenAI-compatible server running DeepSeek-R1.

---

## Self-Hosting the Planner on AMD MI300X

```bash
# On your ROCm server — pull and serve DeepSeek-R1-Distill-Qwen-32B
docker run -d --device /dev/kfd --device /dev/dri \
  --group-add video --group-add render \
  -v /models:/models \
  -p 8000:8000 \
  rocm/vllm:latest \
  vllm serve /models/DeepSeek-R1-Distill-Qwen-32B \
  --gpu-memory-utilization 0.95 \
  --max-model-len 32768

# Download the model (one time, ~62GB)
hf download deepseek-ai/DeepSeek-R1-Distill-Qwen-32B --local-dir /models/DeepSeek-R1-Distill-Qwen-32B
```

Then set in `.env`:
```
PLANNER_BASE_URL=http://your-server-ip:8000/v1
PLANNER_MODEL=/models/DeepSeek-R1-Distill-Qwen-32B
```

---

## What Gets Migrated

**Rule-based (automatic, instant):**
- `torch.backends.cudnn.*` → `torch.backends.miopen.*`
- `CUDA_VISIBLE_DEVICES` → `HIP_VISIBLE_DEVICES`
- `CUDA_LAUNCH_BLOCKING` → `HIP_LAUNCH_BLOCKING`
- `import pycuda` → `import hip`
- `from pycuda.compiler import SourceModule` → `from hip.compiler import SourceModule`
- `torch.cuda.amp.autocast` → `torch.amp.autocast`
- cuDNN benchmark/deterministic/tf32 flags → MIOpen equivalents with comments

**LLM agents (complex reasoning required):**
- Custom CUDA kernels (`<<<grid, block>>>`) → HIP porting plan
- NVTX profiling → `torch.hip.nvtx`
- pycuda kernel execution API → hip-python equivalents
- Flash attention settings → ROCm composable_kernel
- Low-confidence API calls flagged with notes

---

## Benchmark

Full pipeline on `demo_complex.py` (149 lines of CUDA code):

| Metric | Value |
|--------|-------|
| End-to-end time | **31.5 seconds** |
| Rule-based changes applied | 9 automatic |
| LLM reasoning model | DeepSeek-R1-32B on MI300X |
| MI300X VRAM used | 175GB / 192GB (91%) |
| Agent rounds to pass | 1–2 (round-robin, terminates on ALL_TESTS_PASSED) |

---

## Demo

> 📹 *[Add demo video link here after recording]*

---

## Project Structure

```
├── cli/main.py              # CLI entry point (typer)
├── core/
│   ├── analyzer.py          # AST + regex CUDA usage scanner
│   ├── migrator.py          # Rule-based pre-pass
│   └── differ.py            # Unified diff generation
├── agents/
│   ├── planner.py           # DeepSeek-R1 one-shot planner
│   ├── coder.py             # Executor agent (Codestral)
│   ├── reviewer.py          # Reviewer agent
│   ├── tester.py            # AST/import validation
│   └── orchestrator.py      # ag2 GroupChat wiring
├── knowledge/
│   ├── cuda_rocm_map.py     # CUDA→HIP runtime API mappings
│   ├── torch_cuda_map.py    # PyTorch-specific mappings
│   └── optimizations.py     # AMD optimization suggestions
├── config/
│   ├── settings.py          # Pydantic Settings (env-driven)
│   └── model_profiles.py    # Backend config builder
├── demo/
│   └── demo_complex.py      # Example: complex CUDA patterns
└── .env.example             # Config template
```

---

## License

MIT
