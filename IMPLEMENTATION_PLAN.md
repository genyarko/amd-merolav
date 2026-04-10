# CUDA-to-ROCm Migration Agent - Implementation Plan

## Context
Building a hackathon submission for the AMD competition: a multi-agent coding tool that takes NVIDIA CUDA Python/PyTorch code and migrates it to AMD ROCm. Uses AutoGen for coder/reviewer loops, self-hosted models on MI300X via vLLM, with free API fallbacks (Mistral, DeepSeek). Designed to win the "Product Feedback" prize by being a tool that helps devs use AMD's ROCm.

---

## Architecture Overview

```
User CLI invocation
      │
      ▼
┌─────────────┐
│  CLI (main)  │──── reads .py file(s)
└──────┬──────┘
       │
       ▼
┌──────────────┐
│   Analyzer   │──── static scan: finds cuda symbols, imports, device refs
└──────┬───────┘
       │ analysis_report (JSON: list of CUDA usages with line numbers)
       ▼
┌──────────────┐
│  Migrator    │──── rule-based pre-pass using knowledge/ mappings
└──────┬───────┘     (handles trivial renames: cuda→hip, cuDNN→MIOpen imports)
       │ partially_migrated_code + remaining_issues
       ▼
┌─────────────────────────────────────────────────┐
│         AutoGen Orchestrator (GroupChat)          │
│                                                   │
│  ┌──────────┐    ┌───────────┐    ┌──────────┐  │
│  │  Coder   │◄──►│ Reviewer  │◄──►│  Tester  │  │
│  │  Agent   │    │  Agent    │    │  Agent   │  │
│  └──────────┘    └───────────┘    └──────────┘  │
│                                                   │
│  Round 1: Coder completes migration              │
│  Round 2: Reviewer critiques, flags issues       │
│  Round 3: Coder revises                          │
│  Round 4: Tester runs validation, reports        │
│  Round 5: If fail → Coder fixes; if pass → done │
│  (max_rounds configurable, default 6)            │
└──────────┬────────────────────────────────────────┘
           │
           ▼
┌──────────────┐
│   Differ     │──── unified diff: original vs final migrated code
└──────┬───────┘
       │
       ▼
  CLI output: diff view + optimization suggestions + validation report
```

**Hybrid approach**: Rule-based pre-pass handles trivial renames (high-confidence mappings), LLM agents handle complex patterns. This reduces token usage and improves reliability.

---

## Project Structure

```
Rocm/
├── pyproject.toml / requirements.txt
├── .env.example
├── config/
│   ├── settings.py              # Pydantic Settings (env-driven)
│   ├── oai_config_list.json     # AutoGen LLM config (vLLM/Mistral/DeepSeek/Claude)
│   └── model_profiles.py        # Backend profile selector (filter by tags)
├── knowledge/
│   ├── cuda_rocm_map.py         # CUDA→HIP runtime/driver API mappings (~200 entries)
│   ├── torch_cuda_map.py        # PyTorch-specific mappings + warnings
│   ├── library_map.py           # cuDNN→MIOpen, cuBLAS→rocBLAS, NCCL→RCCL, etc.
│   └── optimizations.py         # AMD-specific optimization suggestion rules
├── agents/
│   ├── coder.py                 # Coder agent - performs migration
│   ├── reviewer.py              # Reviewer agent - validates/critiques
│   ├── tester.py                # Tester agent (no LLM) - runs validation functions
│   └── orchestrator.py          # Wires agents into AutoGen GroupChat
├── core/
│   ├── analyzer.py              # AST + regex scan for CUDA usage
│   ├── differ.py                # Unified diff generation
│   ├── migrator.py              # Rule-based pre-pass using knowledge base
│   └── file_io.py               # Safe file I/O
├── testing/
│   ├── runner.py                # Sandboxed execution with mocks
│   ├── mock_hip.py              # Mock torch.cuda/HIP for CPU-only validation
│   └── validators.py            # AST checks (no remaining CUDA refs, etc.)
├── cli/
│   └── main.py                  # Typer CLI entry point
├── tests/
│   ├── test_analyzer.py
│   ├── test_migrator.py
│   ├── test_knowledge.py
│   ├── test_agents.py
│   └── fixtures/                # Sample CUDA scripts for testing
└── demo/
    ├── demo_input.py            # Sample CUDA script for hackathon demo
    └── demo_walkthrough.md
```

---

## Implementation Phases

### Phase 1: Foundation (Day 1 morning) ✅ COMPLETE
1. ~~Initialize project: `pyproject.toml`, `requirements.txt`, `.env.example`~~
2. ~~Build `config/settings.py` with Pydantic Settings (reads `.env`)~~
3. ~~Build `config/model_profiles.py` and `oai_config_list.json` template~~
4. ~~Implement `core/file_io.py` (read user files, output results)~~
5. ~~Implement `core/differ.py` (Python `difflib.unified_diff` wrapper)~~

### Phase 2: Knowledge Base (Day 1 midday) ✅ COMPLETE
1. ~~Build `knowledge/cuda_rocm_map.py` — the core mapping dictionary~~
2. ~~Build `knowledge/torch_cuda_map.py` — PyTorch-specific mappings~~
3. ~~Build `knowledge/library_map.py` — ecosystem library mappings~~
4. ~~Build `knowledge/optimizations.py` — AMD optimization suggestion rules~~
5. ~~Write tests for knowledge base completeness (32 tests passing)~~

### Phase 3: Static Analysis + Rule-Based Migrator (Day 1 afternoon) ✅ COMPLETE
1. ~~Build `core/analyzer.py` — AST + regex scan~~
2. ~~Build `core/migrator.py` — applies knowledge base mappings~~
3. ~~Write tests with fixture files (24 tests passing)~~

### Phase 4: AutoGen Agents (Day 1 evening / Day 2 morning) ✅ COMPLETE
1. ~~Build `agents/coder.py` — system prompt engineering is critical here~~
2. ~~Build `agents/reviewer.py` — system prompt for ROCm expertise~~
3. ~~Build `agents/tester.py` — function-calling agent wrapping test runner~~
4. ~~Build `agents/orchestrator.py` — GroupChat wiring (adapted for pyautogen 0.2.0b2)~~
5. ~~Test agent modules (20 tests passing)~~

### Phase 5: Test Runner (Day 2 midday) ✅ COMPLETE
1. ~~Build `testing/mock_hip.py` — mock modules (torch.cuda, hip, miopen, migraphx)~~
2. ~~Build `testing/validators.py` — AST validation (cudnn refs, imports, device strings, env vars)~~
3. ~~Build `testing/runner.py` — sandboxed subprocess execution with mock injection~~
4. ~~Tests (27 passing: validators + runner + timeout handling)~~

### Phase 6: CLI + Polish (Day 2 afternoon) ✅ COMPLETE
1. ~~Build `cli/main.py` — Typer CLI with rich output, diff view, optimization suggestions~~
2. ~~Create demo files (`demo/demo_input.py` — ResNet fine-tuning with 12+ CUDA patterns)~~
3. ~~End-to-end test: CLI runs against demo file, all validation passes~~
4. ~~Full test suite: 103 tests passing~~

---

## Detailed Module Designs

### config/oai_config_list.json

AutoGen's standard config format. For vLLM self-hosted on MI300X:

```json
[
  {
    "model": "mistralai/Codestral-22B-v0.1",
    "base_url": "http://<MI300X_HOST>:8000/v1",
    "api_key": "EMPTY",
    "tags": ["self-hosted", "codestral"]
  },
  {
    "model": "codestral-latest",
    "base_url": "https://api.mistral.ai/v1",
    "api_key": "${MISTRAL_API_KEY}",
    "tags": ["mistral", "fallback"]
  },
  {
    "model": "deepseek-coder",
    "base_url": "https://api.deepseek.com/v1",
    "api_key": "${DEEPSEEK_API_KEY}",
    "tags": ["deepseek", "fallback"]
  },
  {
    "model": "claude-sonnet-4-20250514",
    "base_url": "https://api.anthropic.com/v1",
    "api_key": "${ANTHROPIC_API_KEY}",
    "tags": ["claude", "optional"]
  }
]
```

`config/model_profiles.py` provides a function `get_config_list(profile: str)` that filters by tags. The CLI accepts `--backend self-hosted|mistral|deepseek|claude`.

### knowledge/cuda_rocm_map.py — Knowledge Base Structure

```python
# Organized by category for maintainability.
# Each entry: cuda_symbol -> (rocm_symbol, notes, confidence)

RUNTIME_API_MAP: dict[str, tuple[str, str, float]] = {
    "cudaMalloc":          ("hipMalloc",          "", 1.0),
    "cudaFree":            ("hipFree",            "", 1.0),
    "cudaMemcpy":          ("hipMemcpy",          "", 1.0),
    "cudaMemcpyAsync":     ("hipMemcpyAsync",     "", 1.0),
    "cudaDeviceSynchronize": ("hipDeviceSynchronize", "", 1.0),
    "cudaGetDeviceCount":  ("hipGetDeviceCount",  "", 1.0),
    "cudaSetDevice":       ("hipSetDevice",       "", 1.0),
    "cudaEventCreate":     ("hipEventCreate",     "", 1.0),
    "cudaEventRecord":     ("hipEventRecord",     "", 1.0),
    "cudaEventSynchronize":("hipEventSynchronize","", 1.0),
    "cudaStreamCreate":    ("hipStreamCreate",    "", 1.0),
    # ... ~200 more entries sourced from AMD's official mapping table
}

DRIVER_API_MAP = { ... }  # cuCtx*, cuModule*, etc. → hip equivalents

ERROR_CODE_MAP = {
    "cudaSuccess":           "hipSuccess",
    "cudaErrorMemoryAllocation": "hipErrorOutOfMemory",
    # ...
}

DEFINE_MAP = {
    "cudaMemcpyHostToDevice":   "hipMemcpyHostToDevice",
    "cudaMemcpyDeviceToHost":   "hipMemcpyDeviceToHost",
    "cudaMemcpyDeviceToDevice": "hipMemcpyDeviceToDevice",
}

ENV_VAR_MAP = {
    "CUDA_VISIBLE_DEVICES": "HIP_VISIBLE_DEVICES",
    "CUDA_LAUNCH_BLOCKING": "HIP_LAUNCH_BLOCKING",
}
```

Source: [CUDA to HIP API Function Comparison](https://rocm.docs.amd.com/projects/HIP/en/latest/reference/api_syntax.html) and [HIPIFY documentation](https://rocm.docs.amd.com/projects/HIPIFY/en/latest/).

### knowledge/torch_cuda_map.py

```python
# PyTorch-specific. Most torch.cuda.* calls work as-is on ROCm,
# but some need awareness/warnings.

TORCH_PASSTHROUGH = {
    'torch.cuda.is_available()': 'Works on ROCm — returns True if HIP device present',
    'torch.device("cuda")':      'ROCm uses "cuda" device string — no change needed',
    '.cuda()':                    'No change — works on ROCm via HIP backend',
}

TORCH_WARNINGS = {
    'torch.backends.cudnn': 'Replace with torch.backends.miopen on ROCm.',
    'torch.backends.cudnn.benchmark': 'Replace with MIOPEN auto-tuning: MIOPEN_FIND_MODE=3',
    'torch.cuda.amp': 'AMP works on ROCm. Prefer torch.amp.autocast("cuda") on PyTorch >= 2.0',
    'torch.cuda.nccl': 'Replace with RCCL. Set env NCCL_SOCKET_IFNAME for multi-GPU.',
    'torch.cuda.nvtx': 'Use rocTX/roctx for profiling markers on ROCm.',
}

BACKENDS_REPLACE = {
    'torch.backends.cudnn.benchmark = True': '# MIOpen auto-tunes by default on ROCm',
    'torch.backends.cudnn.deterministic = True': 'torch.backends.miopen.deterministic = True',
    'torch.backends.cudnn.enabled = True': 'torch.backends.miopen.enabled = True',
}

IMPORT_MAP = {
    'import pycuda': ('import hip', 'Use hip-python bindings'),
    'import cupy':   ('import cupy', 'CuPy supports ROCm — install cupy-rocm'),
    'import triton': ('import triton', 'Triton supports ROCm backend'),
    'import tensorrt': ('import migraphx', 'TensorRT → MIGraphX on AMD'),
}
```

### knowledge/library_map.py

```python
LIBRARY_EQUIVALENTS = {
    "cuDNN":    {"rocm": "MIOpen",    "notes": "Drop-in for conv/RNN/norm ops"},
    "cuBLAS":   {"rocm": "rocBLAS",   "notes": "GEMM/BLAS operations"},
    "cuSPARSE": {"rocm": "rocSPARSE", "notes": "Sparse matrix operations"},
    "cuFFT":    {"rocm": "rocFFT",    "notes": "FFT operations"},
    "cuRAND":   {"rocm": "rocRAND",   "notes": "Random number generation"},
    "NCCL":     {"rocm": "RCCL",      "notes": "Multi-GPU collective comms"},
    "TensorRT": {"rocm": "MIGraphX",  "notes": "Inference optimization engine"},
    "Thrust":   {"rocm": "rocThrust", "notes": "Parallel algorithms library"},
    "CUB":      {"rocm": "hipCUB",    "notes": "Block/warp/device primitives"},
    "cuSOLVER": {"rocm": "rocSOLVER", "notes": "Dense linear algebra solvers"},
    "CUTLASS":  {"rocm": "composable_kernel", "notes": "GEMM kernel templates"},
    "NVRTC":    {"rocm": "hipRTC",    "notes": "Runtime compilation"},
    "cuDLA":    {"rocm": null,         "notes": "No direct equivalent (NVIDIA Jetson-specific)"},
}
```

### knowledge/optimizations.py

```python
OPTIMIZATION_RULES = [
    {
        "trigger": "transformers",
        "suggestion": "Use optimum-amd for ROCm-optimized transformer inference. pip install optimum[amd]",
        "url": "https://github.com/huggingface/optimum-amd",
    },
    {
        "trigger": "torch.nn.Conv",
        "suggestion": "MIOpen auto-tunes convolutions. Set MIOPEN_FIND_MODE=3 for exhaustive tuning.",
    },
    {
        "trigger": "torch.compile",
        "suggestion": "torch.compile works on ROCm with Triton backend. Install triton-rocm.",
    },
    {
        "trigger": "flash_attn",
        "suggestion": "Flash Attention 2 supported on ROCm. Use composable_kernel backend for MI300X.",
    },
    {
        "trigger": "DataParallel",
        "suggestion": "Prefer DistributedDataParallel with RCCL backend for multi-GPU on ROCm.",
    },
    {
        "trigger": "torch.cuda.amp",
        "suggestion": "AMP works. MI300X supports BF16 natively for better training stability.",
    },
    {
        "trigger": "CUDA_VISIBLE_DEVICES",
        "suggestion": "Replace with HIP_VISIBLE_DEVICES (or ROCR_VISIBLE_DEVICES).",
    },
]
```

### core/analyzer.py

Uses Python `ast` module plus regex fallback:

1. Parse the file with `ast.parse()`
2. Walk the AST to find:
   - Import statements containing cuda/pycuda/cupy keywords
   - Attribute accesses like `torch.cuda.*`, `torch.backends.cudnn.*`
   - String literals containing "cuda" (device strings)
   - Function calls matching known CUDA API names from the knowledge base
3. Regex pass for non-Python patterns (CUDA kernel launch syntax `<<<...>>>` in inline strings, environment variables like `CUDA_VISIBLE_DEVICES`)
4. Returns an `AnalysisReport` dataclass: list of `CudaUsage(line, col, symbol, category, context_snippet)`

### core/migrator.py

Deterministic pre-pass before LLM involvement:

1. Takes source code + `AnalysisReport`
2. Applies simple string/AST replacements from knowledge base (confidence == 1.0 entries only)
3. Flags items that need LLM judgment (confidence < 1.0, or complex patterns)
4. Returns `MigrationResult(code: str, applied: list, remaining: list, warnings: list)`

This hybrid approach means the LLM handles only the hard cases, reducing token usage and improving reliability.

### agents/coder.py — System Prompt

```
You are an expert CUDA-to-ROCm migration engineer.

You receive:
1. Original CUDA Python code
2. A partial migration (rule-based pass already applied)
3. A list of remaining issues that need your expertise

Your job:
- Complete the migration to make the code fully ROCm-compatible
- Handle complex patterns: custom CUDA kernels, multi-GPU logic, mixed precision
- Preserve code semantics exactly
- Add comments where behavior may differ on AMD GPUs
- Return ONLY the complete migrated Python code in a code block

Key facts:
- PyTorch ROCm uses "cuda" as device string (NOT "rocm" or "hip")
- torch.cuda.is_available() returns True on ROCm
- cuDNN → MIOpen, cuBLAS → rocBLAS, NCCL → RCCL
- CUDA_VISIBLE_DEVICES → HIP_VISIBLE_DEVICES (or ROCR_VISIBLE_DEVICES)
- pycuda → hip-python bindings
```

### agents/reviewer.py — System Prompt

```
You are a ROCm migration code reviewer.

Review the migrated code and check for:
1. Any remaining CUDA-specific references that won't work on ROCm
2. Incorrect API translations
3. Missing environment variable changes
4. Multi-GPU communication changes (NCCL→RCCL)
5. Performance pitfalls on AMD GPUs
6. Missing MIOpen tuning opportunities

If issues found: list each with line number and fix.
If code looks correct: respond with APPROVED and list any optional optimizations.

IMPORTANT: Do not rewrite the code. Only provide review feedback.
```

### agents/tester.py

The Tester is a `ConversableAgent` with `llm_config=False` and a registered function:

```python
@tester_agent.register_for_execution()
def run_validation(code: str) -> str:
    results = []
    results.append(validators.check_no_cuda_refs(code))
    results.append(validators.check_imports_valid(code))
    results.append(validators.check_device_strings(code))
    results.append(runner.execute_with_mocks(code))
    return format_validation_report(results)
```

### agents/orchestrator.py

```python
def run_migration(original_code, analysis, pre_migrated, config):
    llm_config = {"config_list": load_config_list(config.backend)}

    coder = ConversableAgent("Coder", system_message=CODER_PROMPT, llm_config=llm_config)
    reviewer = ConversableAgent("Reviewer", system_message=REVIEWER_PROMPT, llm_config=llm_config)
    tester = ConversableAgent("Tester", llm_config=False,
                              is_termination_msg=lambda m: "ALL_TESTS_PASSED" in m)

    groupchat = GroupChat(
        agents=[coder, reviewer, tester],
        messages=[],
        max_round=config.max_rounds,  # default 6
        speaker_selection_method="round_robin"  # coder→reviewer→tester→...
    )
    manager = GroupChatManager(groupchat=groupchat, llm_config=llm_config)

    seed = format_seed_message(original_code, pre_migrated, analysis)
    coder.initiate_chat(manager, message=seed)

    return extract_final_code(groupchat.messages)
```

### testing/runner.py and testing/mock_hip.py

Works without a real AMD GPU:

1. **`mock_hip.py`**: Creates mock modules that simulate `torch.cuda` on CPU. Patches `torch.cuda.is_available` to return `True`, replaces `.cuda()` tensor calls with CPU operations, mocks `torch.backends.miopen`. Injected via `sys.modules` before executing user code.

2. **`validators.py`**: Pure AST checks:
   - `check_no_cuda_refs(code)`: Parses AST, ensures no `cuda` string literals remain in device() calls that should be `hip`, no `cudnn` references remain, etc. Note: `"cuda"` in `torch.device("cuda")` is CORRECT on ROCm, so the validator must distinguish this from genuine CUDA-only APIs.
   - `check_imports_valid(code)`: Ensures no `import pycuda`, `import cupy` without ROCm note, etc.
   - `check_env_vars(code)`: Flags `CUDA_VISIBLE_DEVICES` usage.

3. **`runner.py`**: Uses `subprocess` with timeout to execute the migrated code in an isolated Python process with mocks injected. Captures stdout/stderr/return code. Reports: PASS (runs without error), FAIL (exception with traceback), or WARN (runs but with deprecation warnings).

---

## CLI Design

```
Usage: rocm-migrate [OPTIONS] INPUT_PATH

Arguments:
  INPUT_PATH    Path to .py file or directory of .py files

Options:
  --backend     Model backend: self-hosted | mistral | deepseek | claude  [default: mistral]
  --vllm-url    vLLM server URL (for self-hosted)  [default: http://localhost:8000/v1]
  --max-rounds  Max agent refinement rounds  [default: 6]
  --output      Output directory for migrated files  [default: ./rocm_output/]
  --diff-only   Only show diff, don't write files
  --no-test     Skip validation step
  --verbose     Show full agent conversation
  --help        Show this message
```

---

## Model Backend Configuration

`oai_config_list.json` supports 4 backends via tags:
1. **self-hosted**: vLLM on MI300X (`base_url: http://<host>:8000/v1`, model: `Codestral-22B-v0.1`)
2. **mistral-api**: Mistral Experiment tier (free, 1 req/s, 30 req/min)
3. **deepseek-api**: DeepSeek API (free credits)
4. **claude**: Anthropic API (optional)

All use OpenAI-compatible endpoints — no extra SDKs needed except `anthropic` for Claude.

### Configuration Flow

```
.env file (secrets):
  MISTRAL_API_KEY=...
  DEEPSEEK_API_KEY=...
  ANTHROPIC_API_KEY=...   # optional
  VLLM_BASE_URL=http://mi300x-host:8000/v1

config/settings.py reads .env via pydantic-settings

CLI --backend flag selects which entries from oai_config_list.json to use
  (filtered by "tags" field)

For hackathon demo:
  1. Start with --backend mistral (free tier, immediate, 1 req/s)
  2. Switch to --backend self-hosted when MI300X is available
  3. Show both working in the demo
```

---

## Dependencies

```
pyautogen>=0.2.35,<0.3
typer>=0.12
rich>=13.0
pydantic-settings>=2.0
python-dotenv>=1.0
# Optional: anthropic>=0.30 (for Claude backend)
# Dev: pytest>=8.0, pytest-asyncio>=0.23
```

---

## Key Design Decisions

1. **Tester agent has no LLM** — uses registered functions for deterministic validation (saves tokens)
2. **Rule-based pre-pass before LLM** — handles high-confidence mappings, LLM only tackles complex patterns
3. **`torch.device("cuda")` is CORRECT on ROCm** — validator must not flag this as an error
4. **Round-robin speaker selection** — predictable Coder→Reviewer→Tester flow
5. **pyautogen 0.2.x** — stable, well-documented, supports vLLM via OpenAI-compatible config

---

## Potential Challenges and Mitigations

| Challenge | Mitigation |
|---|---|
| vLLM function calling support | Tester uses registered functions, not OpenAI function calling. Coder/Reviewer use plain chat completion only. |
| Free Mistral tier rate limit (1 req/s) | Each agent round is 1 request. 6 rounds = 6 seconds minimum. Acceptable for demo. Add retry with backoff. |
| LLM hallucinating wrong API mappings | Rule-based migrator handles high-confidence mappings first. LLM only handles remaining items. Reviewer agent catches errors. Tester validates. |
| No real AMD GPU for testing | Mock-based validation is the whole point of the test runner design. AST checks catch most issues deterministically. |

---

## Hackathon Demo Strategy

The demo script should show:
1. A realistic CUDA training script (ResNet fine-tuning with multi-GPU, mixed precision, cuDNN benchmark) as input
2. The analyzer identifying all 12+ CUDA-specific patterns
3. The rule-based migrator handling 8 of them automatically
4. The Coder agent completing the remaining 4 complex migrations
5. The Reviewer catching a subtle issue (e.g., `cudnn.benchmark` not converted)
6. The Coder fixing it
7. The Tester validating: all checks pass
8. A clean unified diff output with rich terminal formatting
9. AMD optimization suggestions (Optimum-AMD, MIOpen tuning, RCCL)

---

## Verification Plan

1. **Unit tests**: `pytest tests/` — test analyzer, migrator, knowledge base completeness, validators
2. **Integration test**: Feed `demo/demo_input.py` through full pipeline with `--backend mistral`
3. **Diff check**: Verify output diff is clean and correct
4. **Validation**: Tester reports ALL_TESTS_PASSED on migrated code
5. **Multi-backend**: Test with at least 2 backends (Mistral API + one other)

---

## Next Steps: Demo-Ready Checklist

### Step 1: AMD Developer Cloud Deployment

Updated. Once on the droplet, run:

  bash setup_gpu.sh hf_YOUR_TOKEN

- [ ] Sign up at [AMD Developer Cloud](https://www.amd.com/en/developer/resources/developer-cloud.html) and claim $100 credits
- [ ] **Use the "vLLM" Quick Start Package** (vLLM 0.17.1, ROCm 7.2.0) — this is pre-configured for LLM inference, no manual setup needed
  - Alternative: "ROCm Software" package if you want to install everything from scratch
  - Available access methods: SSH or JupyterLabs
- [ ] Connect via SSH and verify GPU: `rocm-smi` should show MI300X
- [ ] vLLM is pre-installed. Launch a model directly:
  ```bash
  vllm serve mistralai/Codestral-22B-v0.1 --host 0.0.0.0 --port 8000
  ```
  Alternative models to try: `deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct`, `Qwen/Qwen2.5-Coder-7B-Instruct`
- [ ] Clone this repo onto the MI300X instance
- [ ] Update `.env` with `VLLM_BASE_URL=http://localhost:8000/v1`
- [ ] Run the full pipeline end-to-end with `--backend self-hosted`:
  ```bash
  python -m cli.main demo/demo_input.py --backend self-hosted --verbose
  ```
- [ ] Verify all tests pass on the cloud instance: `pytest tests/`
- [ ] **Benchmark**: Time the full migration pipeline on MI300X vs free Mistral API — capture the speedup numbers for the demo

### Step 2: End-to-End Demo Recording

Record a **2-3 minute terminal demo** (use [asciinema](https://asciinema.org/) or screen recording):

1. **Show the problem** (15s): Open `demo/demo_input.py` — highlight CUDA-specific code (cudnn, NCCL, CUDA_VISIBLE_DEVICES, mixed precision)
2. **Run the tool** (30s): `python -m cli.main demo/demo_input.py --backend self-hosted --verbose`
3. **Show agent conversation** (45s): Coder migrates → Reviewer catches issue → Coder fixes → Tester validates → ALL_TESTS_PASSED
4. **Show the output** (30s): Rich diff view with green/red highlighting, optimization suggestions panel
5. **Show it running** (30s): Run the migrated output file on the MI300X to prove it actually works on AMD hardware
6. **Closing slide** (15s): Architecture diagram, tech stack, "Built on AMD Developer Cloud"

**Tips for a strong demo:**
- Pre-warm the vLLM model so there's no cold-start delay
- Use `--verbose` to show the multi-agent conversation (judges love seeing agents collaborate)
- Have a second terminal ready showing `rocm-smi` with GPU utilization during inference

### Step 3: Build in Public (Extra Prize)

You need **at least 2 technical updates** on social media. Here's a posting plan:

**Post 1 — "Building a CUDA→ROCm Migration Agent" (publish during development)**
- Share a screenshot of the agent conversation (Coder → Reviewer → Tester loop)
- Brief explanation: "Built a multi-agent system that auto-migrates CUDA Python code to AMD ROCm. Uses rule-based pre-pass + LLM agents for complex patterns."
- Tag: `@lablab` on X / `lablab.ai` on LinkedIn, `@AIatAMD` on X / `AMD Developer` on LinkedIn
- Hashtags: `#ROCm #AMDDeveloper #AIAgents #Hackathon`

**Post 2 — "Developer Experience Feedback on AMD Developer Cloud" (publish after deployment)**
- Share honest feedback about the AMD Developer Cloud experience:
  - Setup process (how easy/hard was provisioning MI300X?)
  - ROCm compatibility (did PyTorch just work? Any gotchas?)
  - vLLM on AMD (performance, stability, any issues?)
  - What's better/worse than NVIDIA ecosystem?
- This is what wins the "Product Feedback" prize — be specific and constructive
- Tag same accounts as above

**Post 3 (bonus) — "Results: MI300X vs Free API for LLM Agent Inference"**
- Share benchmark numbers: latency per agent round on MI300X vs Mistral free tier
- Screenshot of `rocm-smi` during inference showing GPU utilization
- This shows you actually used the hardware, not just the API

### Step 4: Open-Source the Project

- [ ] Add a `LICENSE` file (MIT recommended)
- [ ] Write a `README.md` with:
  - One-paragraph description
  - Architecture diagram (copy from this plan)
  - Quick start: `pip install -r requirements.txt && python -m cli.main your_cuda_script.py`
  - Example input/output diff screenshot
  - Supported backends table
  - Link to demo video
- [ ] Clean up `.env` — ensure no real API keys are committed (`.gitignore` should already handle this)
- [ ] Push to a public GitHub repo
- [ ] Add the repo link to your hackathon submission

### Step 5: Submission Package

The final submission should include:

1. **GitHub repo** — public, with README, LICENSE, and clean commit history
2. **Demo video** — 2-3 min screen recording showing the full pipeline
3. **Devpost/submission form** — fill in:
   - Track: "AI Agents & Agentic Workflows"
   - Tech stack: AutoGen + vLLM + Codestral on MI300X + ROCm
   - What it does: one paragraph
   - How AMD hardware was used: vLLM inference on MI300X, ROCm compatibility validation
   - Link to Build in Public posts
4. **Slide deck (optional but recommended)** — 3-5 slides:
   - Problem: migrating CUDA code to ROCm is tedious and error-prone
   - Solution: automated multi-agent migration tool
   - Architecture: analyzer → rule-based migrator → LLM agents → validator
   - Demo results: before/after diff, benchmark numbers
   - Why AMD: this tool grows the ROCm ecosystem by lowering the migration barrier

---

## Timeline Estimate

| Task | Effort |
|---|---|
| AMD Cloud deployment + vLLM setup | Half day |
| End-to-end testing on MI300X | 2-3 hours |
| Demo recording | 1-2 hours |
| Build in Public posts (2-3) | 1 hour |
| README + open-source cleanup | 1-2 hours |
| Submission form + slide deck | 1 hour |
| **Total remaining** | **~1.5-2 days** |

---

## Post-Hackathon Improvement Roadmap

The following phases take the project from "impressive hackathon demo" to "production-grade daily-driver tool." They are ordered by impact and dependency — later phases build on earlier ones.

---

### Phase 7: Error Handling & Logging ✅

**Goal**: Eliminate silent failures and make the tool debuggable in production use.

**Steps**:

1. **Add structured logging throughout the codebase**
   - Create `core/logging.py` with a configured `logging.Logger` using Python's standard `logging` module
   - Use `logging.getLogger(__name__)` in every module instead of printing or silently passing
   - Map CLI `--verbose` to `DEBUG` level, normal mode to `INFO`, add `--quiet` for `WARNING`-only
   - Log to both stderr (for terminal) and an optional rotating file (`--log-file`)

2. **Fix silent exception swallowing**
   - `core/analyzer.py`: Replace `except SyntaxError: pass` with logged warning + fallback explanation
   - `agents/orchestrator.py`: Catch and log LLM API errors (timeout, rate limit, auth failure) with actionable messages
   - `testing/runner.py`: Log subprocess stderr on non-zero exit codes

3. **Add graceful degradation for LLM backends**
   - In `agents/orchestrator.py`, wrap each agent call with retry logic (exponential backoff, max 3 retries)
   - If the Planner backend is unreachable, fall back to a simpler prompt sent to the Executor model
   - If all LLM backends fail, complete the migration with rule-based pass only and warn the user
   - Add connection health check at CLI startup: `GET /v1/models` to verify backend availability before starting

4. **Make timeouts configurable**
   - Add `--planner-timeout` and `--test-timeout` CLI flags (currently hardcoded at 600s and 30s)
   - Expose in `config/settings.py` as Pydantic fields with sensible defaults

5. **Add input validation**
   - Validate input file exists and is readable before starting pipeline
   - Validate file is valid Python (or warn if not) before AST parsing
   - Validate output directory is writable
   - Validate API keys are set for the selected backend before making calls

**Files to modify**: `core/analyzer.py`, `core/migrator.py`, `agents/orchestrator.py`, `agents/planner.py`, `agents/coder.py`, `testing/runner.py`, `cli/main.py`, `config/settings.py`
**New files**: `core/logging.py`

---

### Phase 8: Multi-File & Project-Level Migration ✅

**Goal**: Handle real-world projects where CUDA usage spans multiple files with cross-module dependencies.

**Steps**:

1. **Add directory traversal and file discovery**
   - Extend `cli/main.py` to accept directories and glob patterns (e.g., `rocm-migrate src/**/*.py`)
   - Add `--recursive` flag for directory inputs
   - Add `--exclude` patterns (e.g., `--exclude "tests/*"`) to skip files
   - Use `pathlib.Path.rglob()` for cross-platform file discovery

2. **Build an import graph resolver**
   - Create `core/import_graph.py` that:
     - Parses all Python files in the project to build an import dependency graph
     - Identifies which modules define CUDA symbols and which modules consume them
     - Determines migration order (leaf dependencies first, then consumers)
   - Use Python's `ast` module to extract `import` and `from ... import` statements
   - Resolve relative imports using the project's package structure

3. **Add cross-file symbol tracking**
   - Extend `core/analyzer.py` with a `ProjectAnalyzer` class that:
     - Runs `analyze_source()` on each file
     - Tracks which CUDA symbols are exported (defined in `__all__` or at module scope)
     - Tracks which files import those symbols
     - Generates a `ProjectAnalysisReport` with per-file reports + dependency edges
   - This enables migrating a CUDA utility module AND updating all its consumers

4. **Implement project-level migration orchestration**
   - Create `core/project_migrator.py` that:
     - Takes a `ProjectAnalysisReport`
     - Migrates files in dependency order (utilities first, then consumers)
     - Passes cross-file context to the LLM agents (e.g., "this module's `init_cuda()` was renamed to `init_hip()`")
     - Generates a project-level summary report

5. **Add progress reporting for multi-file runs**
   - Show a Rich progress bar with file count, current file, and per-file status
   - Generate a summary table at the end: files processed, changes applied, warnings, errors

**Files to modify**: `cli/main.py`, `core/analyzer.py`
**New files**: `core/import_graph.py`, `core/project_migrator.py`

---

### Phase 9: Confidence Scoring & Migration Quality Metrics ✅

**Goal**: Give users clear signals about which migrations are reliable and which need manual review.

**Steps**:

1. **Add confidence scores to every migration action**
   - Extend `MigrationResult.applied` entries with a `confidence: float` field (0.0–1.0)
   - Rule-based replacements from knowledge base: use the existing confidence values (most are 1.0)
   - LLM-generated changes: assign confidence based on:
     - Whether the Reviewer approved (0.9) or flagged concerns (0.5–0.7)
     - Whether the Tester passed (boost +0.1) or failed (drop to 0.3)
     - Whether the pattern matches a known mapping (boost +0.1)

2. **Implement a migration quality report**
   - Create `core/quality.py` with a `MigrationQualityReport` dataclass:
     - `high_confidence`: list of changes with confidence >= 0.9
     - `needs_review`: list of changes with confidence 0.5–0.89
     - `low_confidence`: list of changes with confidence < 0.5
     - `overall_score`: weighted average confidence across all changes
   - Display in CLI output as a color-coded table (green/yellow/red)

3. **Detect and flag false positives**
   - In `core/analyzer.py`, add heuristics to distinguish CUDA API calls from coincidental name matches:
     - Variable names containing "cuda" (e.g., `cuda_device_count = 4`) — not a CUDA API call
     - Strings in comments or docstrings — informational, not functional
     - CUDA references inside `try/except ImportError` blocks — already guarded
   - Add a `is_false_positive: bool` field to `CudaUsage` entries

4. **Generate a human-readable review checklist**
   - After migration, output a markdown checklist of items needing manual review
   - Include file path, line number, original code, migrated code, and reason for low confidence
   - Optionally write to `rocm_output/REVIEW_CHECKLIST.md`

**Files to modify**: `core/analyzer.py`, `core/migrator.py`, `agents/orchestrator.py`, `cli/main.py`
**New files**: `core/quality.py`

---

### Phase 10: CUDA C/C++ Kernel Support ✅

**Goal**: Handle `.cu` files and inline CUDA C code in Python strings, even if full automatic migration isn't possible.

**Steps**:

1. **Add CUDA C/C++ pattern detection**
   - Extend `core/analyzer.py` to recognize `.cu` and `.cuh` file extensions
   - Add regex patterns for CUDA C constructs:
     - Kernel declarations: `__global__ void`, `__device__`, `__shared__`
     - Kernel launches: `<<<gridDim, blockDim>>>`
     - CUDA runtime calls in C: `cudaMalloc`, `cudaMemcpy`, etc.
     - CUDA-specific types: `dim3`, `cudaStream_t`, `cudaEvent_t`
   - Detect inline CUDA C in Python strings (e.g., pycuda `SourceModule("")` blocks)

2. **Build a CUDA C → HIP C mapping layer**
   - Create `knowledge/cuda_c_map.py` with C-specific mappings:
     - `__global__` → `__global__` (same in HIP)
     - `<<<grid, block>>>` → `hipLaunchKernelGGL()` macro
     - `cudaMalloc` → `hipMalloc` (same as Python mappings, but in C context)
     - CUDA headers: `cuda_runtime.h` → `hip/hip_runtime.h`
     - Texture/surface API mappings (complex, low confidence)

3. **Implement basic `.cu` file migration**
   - Create `core/cuda_c_migrator.py` that:
     - Applies high-confidence C-level replacements (headers, runtime API calls, types)
     - Flags complex patterns (texture memory, cooperative groups, warp intrinsics) for manual review
     - Suggests running AMD's `hipify-perl` or `hipify-clang` for full automated conversion
   - For inline CUDA C in Python strings: extract, migrate, and re-embed

4. **Integrate HIPIFY as an optional backend**
   - If `hipify-perl` is available on the system, offer to run it as a first pass on `.cu` files
   - Parse HIPIFY output and feed it into the LLM agents for refinement
   - Add `--use-hipify` CLI flag

5. **Add `.cu` file support to the CLI**
   - Accept `.cu` and `.cuh` files as input alongside `.py`
   - When processing a project directory, discover and process all CUDA file types
   - Show C-specific migration stats in the summary

**Files to modify**: `core/analyzer.py`, `cli/main.py`
**New files**: `knowledge/cuda_c_map.py`, `core/cuda_c_migrator.py`

---

### Phase 11: Caching & Incremental Migration ✅

**Goal**: Avoid redundant LLM calls, support iterative workflows, and maintain an audit trail.

**Steps**:

1. **Implement a migration cache**
   - Create `core/cache.py` with a file-based cache (JSON or SQLite in `.rocm_cache/`):
     - Key: SHA-256 hash of (source code + analysis report + backend model)
     - Value: migration result (migrated code, applied changes, confidence scores)
   - Check cache before invoking LLM agents; use cached result if source hasn't changed
   - Add `--no-cache` CLI flag to force fresh migration
   - Add `--clear-cache` CLI flag to wipe the cache

2. **Add incremental migration support**
   - Track previously migrated files in `.rocm_cache/manifest.json`:
     - File path, source hash, last migration timestamp, result hash
   - On re-run, only re-migrate files whose source hash has changed
   - Show "X files unchanged, Y files re-migrated" in CLI output

3. **Build a migration audit log**
   - Create `core/audit.py` that writes a structured log to `rocm_output/migration_log.json`:
     - Timestamp, input file, backend used, changes applied, confidence scores, warnings
     - Agent conversation summary (if `--verbose`)
   - Append to the log on each run (don't overwrite)
   - Add `--show-history` CLI flag to display migration history for a file

4. **Add Planner output caching specifically**
   - The Planner (DeepSeek-R1) is the slowest and most expensive call
   - Cache Planner reasoning output separately, keyed by (source code hash + remaining issues hash)
   - Reuse cached plan even if Executor/Reviewer models change

**Files to modify**: `agents/orchestrator.py`, `agents/planner.py`, `cli/main.py`
**New files**: `core/cache.py`, `core/audit.py`

---

### Phase 12: Expanded Test Coverage ✅

**Goal**: Catch regressions, validate edge cases, and enable confident refactoring.

**Steps**:

1. **Add end-to-end tests with mocked LLM responses**
   - Create `tests/test_e2e.py` that:
     - Mocks the OpenAI-compatible API endpoint (using `unittest.mock.patch` or `responses` library)
     - Feeds predefined LLM responses for each agent round
     - Verifies the full pipeline produces expected output for each fixture
   - Cover scenarios: simple migration (rule-based only), complex migration (agents needed), all-backends-down fallback

2. **Add edge case tests for the analyzer**
   - `tests/test_analyzer_edge_cases.py`:
     - CUDA references inside comments and docstrings (should not trigger migration)
     - Variable names containing "cuda" (e.g., `is_cuda_available = True`)
     - Nested template syntax that looks like kernel launches
     - `from pycuda.compiler import *` (wildcard imports)
     - Files with syntax errors (should fall back to regex gracefully)
     - Empty files and files with no CUDA patterns

3. **Add property-based tests for the migrator**
   - Use `hypothesis` library to generate random valid Python code
   - Property: migrated code should always be valid Python (parseable by `ast.parse`)
   - Property: migrating already-migrated code should be a no-op (idempotency)
   - Property: migration should never increase the number of CUDA references

4. **Add regression tests for known issues**
   - Create `tests/fixtures/` entries for each known edge case:
     - Multi-line string containing CUDA C kernel code
     - Conditional CUDA imports (`try: import pycuda except: pass`)
     - Mixed CUDA and ROCm code (partially migrated files)
     - Very large files (1000+ lines) to test chunking behavior
   - Pin expected output for each fixture

5. **Add agent interaction tests**
   - Create `tests/test_agent_interactions.py`:
     - Test that Reviewer feedback is correctly passed back to Coder
     - Test that Tester termination condition (`ALL_TESTS_PASSED`) works
     - Test max_rounds enforcement
     - Test graceful handling of malformed LLM responses

6. **Add CI configuration**
   - Create `.github/workflows/test.yml`:
     - Run `pytest` on push and PR
     - Test on Python 3.10, 3.11, 3.12
     - Skip LLM-dependent tests in CI (mark with `@pytest.mark.requires_llm`)
     - Report code coverage with `pytest-cov`

**Files to modify**: `tests/test_analyzer.py`, `tests/test_migrator.py`
**New files**: `tests/test_e2e.py`, `tests/test_analyzer_edge_cases.py`, `tests/test_agent_interactions.py`, `tests/conftest.py` (shared fixtures), `.github/workflows/test.yml`
**New dependencies**: `hypothesis`, `pytest-cov`, `responses` (or `aioresponses`)

---

### Phase 13: Validation Improvements ✅

**Goal**: Move beyond AST-only checks to give users higher assurance that migrated code actually works.

**Steps**:

1. **Add optional real-execution validation against ROCm runtime**
   - Extend `testing/runner.py` with a `execute_on_rocm()` mode:
     - Detect if a real ROCm/HIP runtime is available (`torch.cuda.is_available()` on a ROCm system)
     - If available, run migrated code in a subprocess with real GPU access
     - Capture and report runtime errors, CUDA/HIP errors, incorrect results
   - Add `--validate-on-gpu` CLI flag (disabled by default, requires AMD hardware)

2. **Add semantic equivalence checking**
   - Create `testing/equivalence.py`:
     - For tensor operations: run both original (CUDA) and migrated (ROCm) code on CPU
     - Compare output tensors with `torch.allclose()` (configurable tolerance)
     - Report semantic mismatches with the specific operation and tensor values
   - This catches cases where migration changes behavior, not just syntax

3. **Enhance AST validators with pattern-aware checks**
   - Add validators for:
     - Mixed CUDA/ROCm imports (partially migrated state)
     - Orphaned environment variables (e.g., `HIP_VISIBLE_DEVICES` set but never used)
     - Incompatible library combinations (e.g., `import tensorrt` alongside `import migraphx`)
     - Deprecated ROCm APIs (check against a version-specific deprecation list)

4. **Add a validation summary with actionable feedback**
   - Instead of just PASS/FAIL, provide:
     - What was checked and the result
     - For failures: specific line, expected vs actual, suggested fix
     - Severity levels: ERROR (will crash), WARNING (may behave differently), INFO (cosmetic)

**Files to modify**: `testing/runner.py`, `testing/validators.py`, `cli/main.py`
**New files**: `testing/equivalence.py`

---

### Phase 14: CLI / UX Enhancements ✅

**Goal**: Make the tool pleasant and efficient for daily use by developers.

**Steps**:

1. **Add interactive mode**
   - Add `--interactive` CLI flag that shows each proposed change and prompts accept/reject/edit:
     - Display the change with context (3 lines before/after)
     - Options: `[y]es / [n]o / [e]dit / [a]ll / [q]uit`
     - Rejected changes are logged for review
   - Use Rich's `Prompt` and `Confirm` for terminal UI

2. **Add dry-run mode with summary table**
   - Add `--dry-run` flag (alias for existing `--diff-only` but with enhanced output):
     - Summary table: file, # changes, confidence distribution, estimated complexity
     - Per-change table: line, original, replacement, confidence, category
     - Total migration effort estimate: "X high-confidence, Y need review, Z need manual work"

3. **Add watch mode for iterative development**
   - Add `--watch` flag that monitors input files for changes using `watchdog` library
   - On file change: re-analyze, re-migrate (using cache for unchanged portions), re-validate
   - Display incremental updates in the terminal
   - Useful when developers are manually fixing low-confidence items

4. **Support glob patterns for input**
   - Accept patterns like `rocm-migrate "src/**/*.py"` and `rocm-migrate src/ --include "*.py" --exclude "test_*"`
   - Display matched files before starting and confirm with user

5. **Add output format options**
   - `--format diff` (default): unified diff with Rich highlighting
   - `--format json`: structured JSON output for programmatic consumption
   - `--format markdown`: migration report in markdown (for PRs/docs)
   - `--format patch`: standard `.patch` file that can be applied with `git apply`

6. **Add shell completions**
   - Use Typer's built-in completion generation for bash/zsh/fish/PowerShell
   - Document installation in README

**Files to modify**: `cli/main.py`, `config/settings.py`
**New dependencies**: `watchdog` (for watch mode)

---

### Phase 15: Knowledge Base Expansion ✅

**Goal**: Cover the full breadth of the CUDA ecosystem so fewer patterns fall through to LLM agents.

**Steps**:

1. **Add Thrust → rocThrust mappings**
   - In `knowledge/library_map.py`, add detailed API mappings:
     - `thrust::device_vector` → `thrust::device_vector` (works with rocThrust)
     - `thrust::sort`, `thrust::reduce`, `thrust::transform` — same API, different backend
     - Header mappings: `<thrust/sort.h>` → same (rocThrust is API-compatible)
   - Add to `knowledge/cuda_rocm_map.py` for any divergent APIs

2. **Add CUB → hipCUB mappings**
   - `cub::DeviceReduce` → `hipcub::DeviceReduce`
   - `cub::BlockReduce` → `hipcub::BlockReduce`
   - Header: `<cub/cub.cuh>` → `<hipcub/hipcub.hpp>`
   - Namespace: `cub::` → `hipcub::`

3. **Add CUDA Graphs → HIP Graphs mappings**
   - `cudaGraphCreate` → `hipGraphCreate`
   - `cudaStreamBeginCapture` → `hipStreamBeginCapture`
   - `cudaGraphLaunch` → `hipGraphLaunch`
   - All `cudaGraph*` → `hipGraph*` (consistent naming pattern)
   - Add notes about feature parity gaps (some HIP Graph features lag behind CUDA)

4. **Add TensorRT → MIGraphX migration hints**
   - `import tensorrt as trt` → `import migraphx`
   - `trt.Builder` → `migraphx.parse_onnx` (different API paradigm)
   - `trt.Runtime` → `migraphx.load` / `migraphx.run`
   - Mark these as low-confidence (API paradigms differ significantly)
   - Add optimization note: export to ONNX first, then use MIGraphX

5. **Add cooperative groups and warp intrinsics mappings**
   - `cooperative_groups::thread_block` → HIP equivalent
   - `__shfl_sync`, `__shfl_down_sync` → `__shfl`, `__shfl_down` (HIP warp intrinsics)
   - `__ballot_sync` → `__ballot`
   - Mark warp-size differences: NVIDIA warp = 32, AMD wavefront = 64 (MI300X) — flag this as a critical behavioral difference

6. **Add version-aware mappings**
   - Some mappings depend on ROCm version (e.g., ROCm 5.x vs 6.x vs 7.x)
   - Add `min_rocm_version` field to mapping entries
   - Add `--rocm-version` CLI flag (default: latest)
   - Filter out mappings not available in the target ROCm version

**Files to modify**: `knowledge/cuda_rocm_map.py`, `knowledge/library_map.py`, `knowledge/torch_cuda_map.py`, `knowledge/optimizations.py`
**New files**: `knowledge/cuda_c_map.py` (if not created in Phase 10)

---

### Phase 16: Smart Context Handling for Large Files ✅

**Goal**: Handle files of any size without overflowing LLM context windows.

**Steps**:

1. **Implement function-level chunking**
   - Create `core/chunker.py` that:
     - Parses the file's AST to identify top-level functions, classes, and module-level code
     - Splits the file into chunks, each containing one logical unit + its imports/dependencies
     - Tracks which chunks contain CUDA patterns (skip clean chunks)
   - Respect LLM context limits: default chunk size = 4000 tokens, configurable via `--chunk-size`

2. **Add chunk-aware migration**
   - Modify `agents/orchestrator.py` to:
     - Migrate each chunk independently through the agent pipeline
     - Pass shared context (imports, class definitions, global variables) to each chunk
     - Reassemble chunks into the final migrated file
   - Preserve original file structure (comments, whitespace, ordering)

3. **Implement context prioritization**
   - When a file exceeds context limits even after chunking:
     - Prioritize chunks with CUDA patterns
     - Include only relevant knowledge base entries (not the full mapping)
     - Summarize the analysis report instead of passing it verbatim
   - Add a `core/context_budget.py` that estimates token usage and trims inputs accordingly

4. **Add streaming output for large files**
   - Show migration progress per-chunk in the CLI
   - Display partial results as chunks complete (don't wait for the full file)
   - Allow interruption and resumption (save completed chunks to cache)

**Files to modify**: `agents/orchestrator.py`, `cli/main.py`
**New files**: `core/chunker.py`, `core/context_budget.py`

---

### Phase 17: Packaging & Distribution

**Goal**: Make the tool installable via pip and runnable anywhere with minimal setup.

**Steps**:

1. **Add proper Python packaging**
   - Update `pyproject.toml` with:
     - `[project.scripts]` entry: `rocm-migrate = "cli.main:app"` so the tool installs as a CLI command
     - Proper classifiers, keywords, and project URLs
     - Version management (start with `0.2.0` post-hackathon)
   - Add `__init__.py` files to all packages if missing
   - Add `__main__.py` to root package so `python -m rocm_migrate` works
   - Test: `pip install -e .` followed by `rocm-migrate --help`

2. **Create a Docker image**
   - Create `Dockerfile` with:
     - Base: `rocm/pytorch:latest` (includes ROCm runtime + PyTorch)
     - Install the tool and dependencies
     - Default entrypoint: `rocm-migrate`
   - Create `docker-compose.yml` for easy local use:
     - Mount input directory as volume
     - Pass through GPU devices for real validation
   - Publish to Docker Hub / GitHub Container Registry

3. **Add a GitHub Actions release workflow**
   - `.github/workflows/release.yml`:
     - Trigger on version tag push (`v*.*.*`)
     - Build and publish to PyPI
     - Build and push Docker image
     - Create GitHub Release with changelog

4. **Add platform-specific installation docs**
   - Linux (native ROCm): `pip install rocm-migrate`
   - macOS/Windows (no ROCm, API-only mode): `pip install rocm-migrate` with `--backend mistral`
   - Docker (with GPU): `docker run --device /dev/kfd --device /dev/dri rocm-migrate input.py`
   - AMD Developer Cloud: pre-install script for MI300X instances

**Files to modify**: `pyproject.toml`, `README.md`
**New files**: `Dockerfile`, `docker-compose.yml`, `.github/workflows/release.yml`, `__main__.py`

---

## Improvement Roadmap Summary

| Phase | Focus Area | Priority | Complexity | Key Deliverable |
|-------|-----------|----------|------------|-----------------|
| 7 | Error Handling & Logging | **Critical** | Low | Debuggable, resilient pipeline |
| 8 | Multi-File Migration | **Critical** | High | Project-level migration support |
| 9 | Confidence Scoring | **Critical** | Medium | Actionable migration quality metrics |
| 10 | CUDA C/C++ Support | High | High | `.cu` file analysis and migration |
| 11 | Caching & Incremental | High | Medium | Fast re-runs, audit trail |
| 12 | Expanded Test Coverage | High | Medium | CI-ready test suite with 90%+ coverage |
| 13 | Validation Improvements | Medium | Medium | Real-execution and equivalence testing |
| 14 | CLI / UX Enhancements | Medium | Medium | Interactive mode, watch, output formats |
| 15 | Knowledge Base Expansion | Medium | Low | Broader CUDA ecosystem coverage |
| 16 | Large File Handling | Medium | High | Chunk-based migration for any file size |
| 17 | Packaging & Distribution | Low | Low | pip install, Docker, CI/CD |

**Phases 7–17**: Complete. All implemented and tested.

---

# Hackathon Multi-Track Expansion

The lablab.ai AMD Hackathon has multiple tracks. Track 1 (CUDA→ROCm migration agent) is submitted. Tracks 2 and 3 extend the project to showcase AMD MI300X capabilities for real ML workloads — fine-tuning and multimodal inference — using existing plant disease detection datasets.

All three tracks share the same AMD infrastructure (MI300X on DigitalOcean Developer Cloud) and demonstrate the ROCm ecosystem end-to-end: from developer tooling (Track 1) to training (Track 2) to serving (Track 3).

---

## Track 2: Fine-Tuning on AMD GPUs

### Phase 18: Dataset Preparation & Merging

**Goal**: Combine multiple Kaggle plant disease datasets into a single unified HuggingFace-format dataset on the MI300X.

**Steps**:

1. **Download datasets from Kaggle to MI300X**
   - Install Kaggle CLI on the droplet
   - Download: `merolavtechnology/dataset-for-crop-pest-and-disease-detection` (Mendeley crop pest, augmented)
   - Download: `ohagwucollinspatrick/amini-cocoa-contamination-dataset` (cocoa contamination)
   - Download: `plantvillage-dataset` (PlantVillage, optional secondary source)

2. **Merge and standardize datasets**
   - Normalize class labels across all datasets (lowercase, underscore-separated)
   - Deduplicate images across sources
   - Add cocoa contamination classes to the existing 22-class crop pest taxonomy
   - Validate all images (remove corrupt files)
   - Target: 30+ classes covering cashew, cassava, maize, tomato, cocoa

3. **Convert to HuggingFace ImageFolder format**
   - Structure: `data/{train,val,test}/{class_name}/*.jpg`
   - Stratified split: 80/10/10 (same strategy as Kaggle notebooks)
   - Generate `dataset_info.json` with class metadata
   - Push to HuggingFace Hub (optional, for reproducibility)

4. **Create dataset statistics report**
   - Per-class image counts, class imbalance analysis
   - Compute class weights for balanced training
   - Sample visualizations per class

**Files to create**: `vision/data/prepare_dataset.py`, `vision/data/merge_datasets.py`
**Infrastructure**: MI300X droplet, Kaggle API

---

### Phase 19: PyTorch Training Pipeline on ROCm

**Goal**: Fine-tune a large vision model on MI300X using PyTorch + ROCm, surpassing the Kaggle EfficientNet baseline.

**Steps**:

1. **Set up PyTorch training environment on MI300X**
   - Use ROCm PyTorch Docker image
   - Install: `timm`, `transformers`, `accelerate`, `wandb` (logging)
   - Verify GPU detection: `torch.cuda.is_available()` on ROCm

2. **Port training pipeline from TF/Keras to PyTorch**
   - Replace `ImageDataGenerator` → `torchvision.transforms` + `DataLoader`
   - Replace EfficientNet → larger models enabled by MI300X VRAM:
     - **ViT-Large/16** (304M params) — needs ~20GB, easily fits MI300X
     - **DINOv2-Large** (300M params) — strong zero-shot, excellent for fine-tuning
     - **EVA-02-Large** (300M params) — state-of-art on image classification
   - Implement 2-phase training: frozen backbone → fine-tune top layers
   - Implement Mixup augmentation in PyTorch
   - Add mixed-precision training (`torch.amp`) — leverages MI300X BF16 support

3. **Train and benchmark**
   - Train ViT-Large or DINOv2-Large on the merged dataset
   - Compare against Kaggle EfficientNetB0/B3 baseline:
     - Accuracy improvement (expect 5-15% gain from larger model + more VRAM)
     - Training speed (MI300X vs Kaggle T4/P100)
     - Batch size advantage (MI300X can do 256+ batch vs Kaggle's 32)
   - Log metrics with Weights & Biases
   - Generate classification report, confusion matrix, per-class F1

4. **Export and serve**
   - Save best model checkpoint
   - Export to ONNX for portable inference
   - Serve via vLLM or TorchServe on MI300X
   - Benchmark inference throughput (images/sec)

5. **ROCm-specific optimizations**
   - Profile with `rocprof` to identify bottlenecks
   - Test `torch.compile()` with ROCm backend
   - Tune `PYTORCH_HIP_ALLOC_CONF` for memory allocator
   - Document MI300X-specific findings (BF16 performance, memory bandwidth utilization)

**Key deliverable**: A plant disease classifier that demonstrably outperforms the Kaggle baseline, trained entirely on AMD MI300X with ROCm, with benchmark comparisons.

**Files to create**: `vision/train.py`, `vision/model.py`, `vision/evaluate.py`, `vision/config.py`
**Infrastructure**: MI300X droplet, ROCm PyTorch Docker

---

### Phase 20: Training Results & Track 2 Submission

**Goal**: Package training results into a polished Track 2 submission.

**Steps**:

1. **Generate benchmark report**
   - MI300X vs Kaggle GPU comparison table (speed, accuracy, batch size, VRAM)
   - Training curves (loss, accuracy) for all model variants
   - Per-class precision/recall/F1 on the merged dataset
   - ROCm-specific insights and optimizations applied

2. **Write submission materials**
   - Technical writeup: architecture, training strategy, results
   - Highlight AMD-specific advantages (192GB VRAM, BF16, large batch sizes)
   - Include code walkthrough and reproducibility instructions

3. **Publish artifacts**
   - Model weights on HuggingFace Hub
   - Dataset on HuggingFace Hub
   - Training scripts in the repo

**Files to create**: `vision/README.md`, `submission/track2_submission.md`

---

## Track 3: Vision & Multimodal AI

### Phase 21: Multimodal Model Selection & Data Preparation

**Goal**: Prepare a vision-language dataset for fine-tuning a multimodal model that can diagnose plant diseases from photos and provide treatment advice.

**Steps**:

1. **Select multimodal model**
   - Primary: **Llama 3.2 Vision (11B)** — fits MI300X with room for training
   - Alternative: **Qwen-VL (7B)** — smaller, faster to fine-tune
   - Both support image + text input/output

2. **Create vision-language QA dataset**
   - For each disease class, generate QA pairs:
     - Q: "What disease does this plant have?" → A: diagnosis + confidence
     - Q: "How should I treat this?" → A: treatment recommendations
     - Q: "Is this plant healthy?" → A: yes/no + explanation
     - Q: "What crop is this?" → A: crop identification
   - Sources for treatment text:
     - Agricultural extension databases
     - Existing disease descriptions from dataset metadata
     - LLM-generated treatment advice (reviewed for accuracy)
   - Format as conversation turns with image references
   - Target: 5-10 QA pairs per class × 30+ classes = 150-300+ training examples

3. **Format for fine-tuning**
   - Convert to the model's expected chat format (Llama or Qwen)
   - Structure: `[{"role": "user", "content": [image, question]}, {"role": "assistant", "content": answer}]`
   - Split into train/val sets

**Files to create**: `vision/data/create_qa_dataset.py`, `vision/data/treatment_knowledge.json`

---

### Phase 22: Multimodal Fine-Tuning & App

**Goal**: Fine-tune the vision-language model on MI300X and build a demo application.

**Steps**:

1. **Fine-tune multimodal model on MI300X**
   - Use HuggingFace `transformers` + `peft` (LoRA) for efficient fine-tuning
   - LoRA reduces VRAM needs: fine-tune Llama 3.2 Vision 11B with <40GB
   - Training config: LoRA rank 16-32, learning rate 2e-5, 3-5 epochs
   - Mixed precision (BF16) on MI300X

2. **Serve fine-tuned model via vLLM**
   - Load fine-tuned adapter on top of base model
   - Serve with vLLM on MI300X (already have the Docker setup)
   - Expose as OpenAI-compatible API: `/v1/chat/completions` with image support

3. **Build demo application**
   - Gradio or Streamlit web app
   - User uploads a photo of a plant leaf
   - App sends image to the vLLM-served model
   - Returns: disease diagnosis, severity estimate, treatment recommendations
   - Include confidence score and alternative diagnoses
   - Add example gallery with pre-loaded images

4. **Evaluate multimodal performance**
   - Test on held-out disease images
   - Compare classification accuracy vs the Track 2 pure vision model
   - Evaluate quality of generated treatment advice
   - Measure inference latency (time to diagnosis)

**Files to create**: `vision/finetune_multimodal.py`, `vision/app.py`, `vision/serve.py`
**Infrastructure**: MI300X droplet, vLLM with vision model support

---

### Phase 23: Track 3 Submission

**Goal**: Package the multimodal plant disease assistant into a polished Track 3 submission.

**Steps**:

1. **Record demo video**
   - Show: upload photo → disease diagnosis → treatment advice
   - Highlight: running on MI300X, real-time inference, multimodal understanding

2. **Write submission materials**
   - Technical writeup: model architecture, fine-tuning approach, evaluation
   - Highlight: "high-throughput agricultural inspection" use case
   - AMD-specific: MI300X memory bandwidth for vision-language inference

3. **Deploy for judges**
   - Public Gradio demo (hosted on MI300X or HuggingFace Spaces)
   - API endpoint for programmatic access

**Files to create**: `submission/track3_submission.md`

---

## Build in Public (Extra Challenge)

### Phase 24: Social Media & Documentation

**Goal**: Publish technical updates and AMD developer feedback to qualify for the Build in Public prize.

**Requirements** (from hackathon rules):
1. Share at least 2 technical updates on social media (tag @lablab on X / lablab.ai on LinkedIn, and @AIatAMD on X / AMD Developer on LinkedIn)
2. Provide meaningful feedback about building with ROCm, AMD Developer Cloud, or APIs
3. Open-source the project or publish a technical walkthrough

**Steps**:

1. **Publish social posts** (drafts already written in `social/`)
   - Post 1: Building the CUDA→ROCm migration agent (Track 1) — `post1_building_agent.md`
   - Post 2: AMD Developer Cloud feedback — `post2_amd_cloud_feedback.md`
   - Post 3: Benchmark results — `post3_benchmark_results.md`
   - Post 4: v0.2.0 update announcement — `post4_v020_update.md`
   - Post 5: (new) Training vision models on MI300X — Track 2 results
   - Post 6: (new) Multimodal plant disease assistant demo — Track 3 showcase

2. **Write technical walkthrough**
   - Blog post or GitHub wiki covering the full journey:
     - Setting up vLLM on MI300X with ROCm
     - Fine-tuning vision models with PyTorch on ROCm
     - Building a multimodal AI app on AMD hardware
     - ROCm developer experience: what worked, what was painful, what's missing

3. **Provide AMD developer feedback**
   - Document ROCm friction points (vLLM pip vs Docker, Triton issues, etc.)
   - Suggest improvements to AMD Developer Cloud onboarding
   - Share performance comparisons (MI300X vs NVIDIA equivalents)

**Files to create**: `social/post5_training_on_mi300x.md`, `social/post6_multimodal_demo.md`, `social/technical_walkthrough.md`

---

## Multi-Track Timeline

| Week | Track 2 (Fine-Tuning) | Track 3 (Multimodal) | Build in Public |
|------|----------------------|---------------------|----------------|
| **1** | Phase 18: Download datasets, merge, convert to HuggingFace format | Phase 21: Select model, create QA dataset | Post existing drafts (posts 1-4) |
| **2** | Phase 19: Train ViT-Large/DINOv2 on MI300X, benchmark vs Kaggle | Phase 22: Fine-tune Llama 3.2 Vision with LoRA | Post 5: Training benchmarks |
| **3** | Phase 20: Results, optimize, export | Phase 22 (cont): Build Gradio demo app | Post 6: Demo video |
| **4** | Submit Track 2 | Phase 23: Submit Track 3 | Technical walkthrough + feedback |

## Multi-Track Roadmap Summary

| Phase | Track | Focus Area | Priority | Complexity | Key Deliverable |
|-------|-------|-----------|----------|------------|-----------------|
| 18 | Track 2 | Dataset Preparation | **Critical** | Medium | Merged multi-crop disease dataset on MI300X |
| 19 | Track 2 | PyTorch Training on ROCm | **Critical** | High | ViT-Large fine-tuned on MI300X, beating Kaggle baseline |
| 20 | Track 2 | Results & Submission | **Critical** | Low | Track 2 submission with benchmarks |
| 21 | Track 3 | Multimodal Data Prep | **Critical** | Medium | Vision-language QA dataset for plant diseases |
| 22 | Track 3 | Multimodal Fine-Tuning & App | **Critical** | High | Fine-tuned Llama 3.2 Vision + Gradio demo |
| 23 | Track 3 | Submission | **Critical** | Low | Track 3 submission with demo |
| 24 | Public | Social Media & Docs | High | Low | 6+ posts, technical walkthrough, AMD feedback |

**Implementation order**: 18 → 19 + 21 (parallel) → 20 + 22 (parallel) → 23 → 24

Phases 18 and 21 share dataset work. Phases 19 and 22 can run in parallel since they use different models. Phase 24 runs continuously throughout.
