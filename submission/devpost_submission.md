# Devpost Submission — Copy-Paste Ready

---

## Project Name
CUDA→ROCm Migration Agent

## Tagline (one line)
Multi-agent AI tool that automatically migrates NVIDIA CUDA Python code to AMD ROCm.

## Track
AI Agents & Agentic Workflows

## Tech Stack
- ag2 (multi-agent framework, formerly AutoGen)
- DeepSeek-R1-Distill-Qwen-32B (Planner — self-hosted on AMD MI300X via vLLM)
- Mistral Codestral (Executor — code generation via API)
- vLLM on ROCm (serving layer on MI300X)
- AMD MI300X GPU on DigitalOcean Developer Cloud
- PyTorch on ROCm 7.2.0
- Python / Typer CLI / Rich terminal UI

---

## What it does (submission paragraph)

CUDA→ROCm Migration Agent is a command-line tool that automatically converts NVIDIA CUDA Python/PyTorch code to run on AMD ROCm hardware. It uses a two-phase pipeline: first, a rule-based pre-pass applies ~30 deterministic high-confidence substitutions (cuDNN→MIOpen, CUDA_VISIBLE_DEVICES→HIP_VISIBLE_DEVICES, import replacements, backend flags) without any LLM calls. Then, a multi-agent loop powered by DeepSeek-R1-Distill-Qwen-32B (running on an AMD MI300X via vLLM) reasons through the remaining complex patterns — custom CUDA kernels, NVTX profiling calls, flash attention settings, and mixed-precision idioms — and produces a step-by-step migration plan. Mistral Codestral implements the plan, a Reviewer agent validates correctness, and an automated Tester runs AST and import checks. The pipeline terminates automatically when all tests pass, producing a clean unified diff and AMD-specific optimization suggestions. Full pipeline on a 149-line CUDA demo file: 31.5 seconds end-to-end.

---

## How AMD hardware was used

- **AMD MI300X (192GB HBM3)** on DigitalOcean AMD Developer Cloud is the inference backend for the Planner agent. DeepSeek-R1-Distill-Qwen-32B (~61GB) runs via Docker (rocm/vllm:latest) with 109GB of KV cache available — enabling long chain-of-thought reasoning over large codebases.
- **ROCm 7.2.0** stack with PyTorch ROCm wheel (`torch+rocm6.2`) — the migrated output code is validated against ROCm-specific compatibility rules.
- **vLLM on ROCm** serves the model at 2103MHz SCLK with 91% VRAM utilization (model weights + pre-allocated KV cache).
- The tool itself is a direct response to the friction of migrating to ROCm — it lowers the barrier for developers adopting AMD hardware.

---

## Build in Public posts
- Post 1 (development): [Add X/LinkedIn link after posting]
- Post 2 (AMD feedback): [Add X/LinkedIn link after posting]
- Post 3 (benchmark): [Add X/LinkedIn link after posting]

---

## GitHub Repo
[Add after pushing]

## Demo Video
[Add after recording]
