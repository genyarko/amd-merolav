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
