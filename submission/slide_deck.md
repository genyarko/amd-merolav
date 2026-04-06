# Slide Deck — CUDA→ROCm Migration Agent
## 5 slides, copy into Google Slides / PowerPoint

---

## SLIDE 1 — Problem

**Title:** Migrating CUDA to ROCm is Tedious and Error-Prone

**Body:**
- AMD ROCm is production-ready — but moving CUDA codebases is a manual, time-consuming process
- Developers must learn cuDNN→MIOpen differences, HIP environment variables, pycuda→hip-python, flash attention quirks...
- One missed substitution silently breaks training
- Result: developers stay on NVIDIA even when AMD hardware is available and cheaper

**Visual:** Side-by-side before/after code snippet (grab from the diff output)

---

## SLIDE 2 — Solution

**Title:** Automated Multi-Agent Migration

**Body:**
- CLI tool: `python -m cli.main your_cuda_script.py`
- Two-phase pipeline:
  - **Phase 1:** Rule-based pre-pass — 30+ deterministic substitutions, zero LLM cost
  - **Phase 2:** Multi-agent LLM loop — DeepSeek-R1 plans, Codestral implements, Reviewer validates, Tester confirms
- Produces a clean unified diff + AMD optimization suggestions
- Terminates automatically when ALL_TESTS_PASSED

**Visual:** Architecture diagram (copy from README)

---

## SLIDE 3 — Architecture

**Title:** How It Works

```
Input CUDA .py
      ↓
  Analyzer (AST scan)
      ↓
  Rule-Based Migrator (30+ mappings)
      ↓
  DeepSeek-R1 Planner (MI300X) → step-by-step plan
      ↓
  Codestral Executor → migrated code
      ↓
  Reviewer → validates correctness
      ↓
  Tester → ALL_TESTS_PASSED → output diff
```

**Key point:** DeepSeek-R1's chain-of-thought reasoning runs on the AMD MI300X — the hardware runs the tool that migrates code TO the hardware.

---

## SLIDE 4 — Results

**Title:** Demo Results

| Metric | Value |
|--------|-------|
| Input | 149-line CUDA demo (custom kernels, NVTX, cuDNN, pycuda) |
| Rule-based changes | 9 automatic, zero LLM cost |
| Full pipeline time | **31.5 seconds** |
| Planner hardware | AMD MI300X, 91% VRAM (175GB / 192GB) |
| Agent rounds | 1–2 to pass validation |

**Visual:** Screenshot of terminal output with Planner panel + ALL_TESTS_PASSED + diff

**Key changes made:**
- `pycuda` → `hip` (hip-python)
- `torch.backends.cudnn.*` → `torch.backends.miopen.*`
- `CUDA_VISIBLE_DEVICES` → `HIP_VISIBLE_DEVICES`
- `torch.cuda.nvtx` → `torch.hip.nvtx`
- Flash attention → ROCm composable_kernel comment

---

## SLIDE 5 — Why AMD

**Title:** Growing the ROCm Ecosystem

**Body:**
- The #1 blocker for AMD GPU adoption is migration friction
- This tool removes that blocker — any developer can migrate a CUDA codebase in under a minute
- Built entirely on AMD infrastructure: MI300X for inference, ROCm for validation
- DeepSeek-R1's reasoning is noticeably better than direct-answer models for migration tasks — chain-of-thought catches ROCm-specific gotchas
- Open source — the community can extend the knowledge base with new mappings

**Call to action:** github.com/[your-repo] | #ROCm #AMDDeveloper
