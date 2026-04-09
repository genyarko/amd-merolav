# CUDA→ROCm Migration Agent

[![Tests](https://github.com/genyarko/amd-merolav/actions/workflows/test.yml/badge.svg)](https://github.com/genyarko/amd-merolav/actions/workflows/test.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Automatically migrates NVIDIA CUDA Python/PyTorch code to AMD ROCm using a two-phase pipeline: a rule-based pre-pass handles deterministic high-confidence replacements, then a multi-agent LLM loop (DeepSeek-R1 as Planner + Mistral Codestral as Executor) reasons through and implements the complex patterns that rules can't handle — custom kernels, profiling APIs, cuDNN tuning flags, and mixed-precision idioms.

Built for the [lablab.ai AMD Hackathon](https://lablab.ai) using an AMD MI300X GPU on DigitalOcean Developer Cloud.

---

## Installation

### Option 1: pip install (recommended)

```bash
pip install rocm-migrate
```

Or install from source:

```bash
git clone https://github.com/genyarko/amd-merolav.git
cd amd-merolav
pip install -e ".[all]"
```

### Option 2: Docker (CPU-only, no GPU required)

```bash
docker pull ghcr.io/genyarko/amd-merolav:latest

# Migrate a file
docker run --rm \
  -v $(pwd)/my_cuda_code:/input:ro \
  -v $(pwd)/output:/output \
  ghcr.io/genyarko/amd-merolav:latest \
  /input --output /output --no-agent
```

### Option 3: Docker with AMD GPU (full validation)

```bash
docker pull ghcr.io/genyarko/amd-merolav:latest-rocm

docker run --rm \
  --device /dev/kfd --device /dev/dri \
  --group-add video --shm-size 16g \
  -v $(pwd)/my_cuda_code:/input:ro \
  -v $(pwd)/output:/output \
  -v .env:/app/.env:ro \
  ghcr.io/genyarko/amd-merolav:latest-rocm \
  /input --output /output --validate-on-gpu
```

### Option 4: docker-compose

```bash
# Place CUDA files in ./input/, then:
docker compose up rocm-migrate        # CPU-only
docker compose up rocm-migrate-gpu    # With AMD GPU validation
docker compose up vllm-planner        # Start the DeepSeek-R1 planner server
```

---

## Platform-Specific Notes

| Platform | ROCm GPU? | How to install | Notes |
|----------|-----------|----------------|-------|
| **Linux (AMD GPU)** | Yes | `pip install rocm-migrate` | Full support: rule-based + LLM + GPU validation |
| **Linux (no GPU)** | No | `pip install rocm-migrate` | Use `--no-agent` or `--backend mistral` (API) |
| **macOS** | No | `pip install rocm-migrate` | API-only mode: `--backend mistral` or `--backend claude` |
| **Windows** | No | `pip install rocm-migrate` | API-only mode: `python -m cli.main` |
| **Docker** | Optional | See above | CPU image for rules, GPU image for validation |
| **AMD Developer Cloud** | Yes | Pre-installed on MI300X instances | See [vLLM setup guide](VLLM_SETUP.md) |

---

## Quick Start

```bash
# Configure (for LLM-powered migration)
cp .env.example .env
# Edit .env with your Mistral API key (free tier works)

# Migrate a file (rule-based only, no API key needed)
rocm-migrate your_cuda_script.py --no-agent

# Migrate with LLM agents
rocm-migrate your_cuda_script.py --backend mistral

# With verbose agent conversation output
rocm-migrate your_cuda_script.py --verbose --force-agents

# Migrate an entire directory
rocm-migrate ./my_cuda_project/ --no-agent

# Show diff without writing files
rocm-migrate your_cuda_script.py --diff-only

# Interactive mode (review each change)
rocm-migrate your_cuda_script.py --interactive

# If rocm-migrate is not on PATH, use:
python -m cli.main your_cuda_script.py --no-agent
```

Output is written to `./rocm_output/` by default.

---

## Architecture

```
User CLI invocation
      |
      v
+-------------+
|  CLI (main) |---- reads .py file(s)
+------+------+
       |
       v
+--------------+
|   Analyzer   |---- static scan: finds CUDA symbols, imports, device refs
+------+-------+
       | analysis_report (CUDA usages with line numbers)
       v
+--------------+
|   Migrator   |---- rule-based pre-pass (~30 deterministic mappings)
+------+-------+     cuDNN->MIOpen, CUDA_VISIBLE_DEVICES->HIP, imports, etc.
       | partially_migrated_code + remaining_issues
       v
+--------------------------------------------+
|          Two-Phase LLM Pipeline            |
|                                            |
|  Phase 1 -- Planner (DeepSeek-R1)         |
|    Reasons through remaining issues        |
|    Produces a numbered migration plan      |
|                                            |
|  Phase 2 -- Executor GroupChat            |
|  +----------+  +----------+  +--------+  |
|  | Executor |->| Reviewer |->| Tester |  |
|  |Codestral |  |Codestral |  |  (AST) |  |
|  +----------+  +----------+  +--------+  |
|  Terminates on ALL_TESTS_PASSED           |
+----------+---------------------------------+
           |
           v
+--------------+
|    Differ    |---- unified diff: original vs migrated
+------+-------+
       v
  CLI output: diff + AMD optimization suggestions + validation report
```

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
# Using docker-compose (easiest)
docker compose up vllm-planner

# Or manually with Docker
docker run -d --name vllm \
  --device=/dev/kfd --device=/dev/dri \
  --group-add video \
  --shm-size 16g \
  -p 8001:8000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  vllm/vllm-openai-rocm:v0.17.1 \
  --model deepseek-ai/DeepSeek-R1-Distill-Qwen-32B \
  --host 0.0.0.0 --port 8000
```

Then set in `.env`:
```
PLANNER_BASE_URL=http://your-server-ip:8001/v1
PLANNER_MODEL=deepseek-ai/DeepSeek-R1-Distill-Qwen-32B
```

See [VLLM_SETUP.md](VLLM_SETUP.md) for detailed setup instructions.

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

## CLI Reference

```
rocm-migrate [OPTIONS] INPUT_PATH

Arguments:
  INPUT_PATH          Path to .py file or directory of .py files

Options:
  --backend TEXT      Model backend: self-hosted | mistral | deepseek | claude [default: mistral]
  --output TEXT       Output directory for migrated files [default: ./rocm_output/]
  --diff-only         Only show diff, don't write files
  --no-agent          Skip LLM agents, only apply rule-based migration
  --force-agents      Always run LLM agents even if rules resolve everything
  --no-test           Skip validation step
  --interactive, -i   Review each proposed change interactively
  --watch, -w         Watch input files for changes and re-migrate
  --dry-run           Show migration summary without writing files
  --format TEXT       Output format: diff | json | markdown | patch [default: diff]
  --no-cache          Skip cache, force fresh migration
  --rocm-version TEXT Target ROCm version (e.g. 6.0)
  --verbose, -v       Show detailed output
  --quiet, -q         Suppress all output except errors
  --help              Show this message and exit
```

---

## Benchmark

Full pipeline on `demo_complex.py` (149 lines of CUDA code):

| Metric | Value |
|--------|-------|
| End-to-end time | **31.5 seconds** |
| Rule-based changes applied | 9 automatic |
| LLM reasoning model | DeepSeek-R1-32B on MI300X |
| MI300X VRAM used | 175GB / 192GB (91%) |
| Agent rounds to pass | 1-2 (round-robin, terminates on ALL_TESTS_PASSED) |

---

## Development

```bash
# Clone and install dev dependencies
git clone https://github.com/genyarko/amd-merolav.git
cd amd-merolav
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Run tests with coverage
pytest tests/ -v --cov=core --cov=agents --cov=knowledge --cov=testing --cov-report=term-missing

# Run only fast tests (no LLM required)
pytest tests/ -v -m "not requires_llm"
```

---

## Project Structure

```
├── cli/main.py              # CLI entry point (typer)
├── core/
│   ├── analyzer.py          # AST + regex CUDA usage scanner
│   ├── migrator.py          # Rule-based pre-pass
│   ├── differ.py            # Unified diff generation
│   ├── audit.py             # Migration audit logging
│   ├── cache.py             # Migration result caching
│   ├── chunker.py           # Large-file chunked migration
│   ├── cuda_c_migrator.py   # CUDA C/C++ kernel migration
│   ├── quality.py           # Migration quality reports
│   └── logging.py           # Structured logging
├── agents/
│   ├── planner.py           # DeepSeek-R1 one-shot planner
│   ├── coder.py             # Executor agent (Codestral)
│   ├── reviewer.py          # Reviewer agent
│   ├── tester.py            # AST/import validation
│   └── orchestrator.py      # ag2 GroupChat wiring
├── knowledge/
│   ├── cuda_rocm_map.py     # CUDA→HIP runtime API mappings
│   ├── cuda_c_map.py        # CUDA C kernel API mappings
│   ├── torch_cuda_map.py    # PyTorch-specific mappings
│   └── optimizations.py     # AMD optimization suggestions
├── testing/
│   ├── runner.py            # Sandbox code execution
│   ├── validators.py        # Migration validation checks
│   └── equivalence.py       # Semantic equivalence testing
├── config/
│   ├── settings.py          # Pydantic Settings (env-driven)
│   └── model_profiles.py    # Backend config builder
├── tests/                   # Pytest test suite
├── demo/                    # Example CUDA files
├── Dockerfile               # Multi-stage: cpu + gpu targets
├── docker-compose.yml       # Easy local deployment
└── .env.example             # Config template
```

---

## License

MIT
