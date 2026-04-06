# Post 1 — "Building a CUDA→ROCm Migration Agent"
**Platform:** X (Twitter) + LinkedIn
**Timing:** Publish now (during development / after demo recording)
**Attach:** Screenshot of the Planner → Executor → Reviewer → Tester terminal output

---

## X (Twitter) — 280 chars

Built a multi-agent CUDA→ROCm migration tool for the @lablab AMD hackathon 🔥

DeepSeek-R1 (running on MI300X) reasons through the plan. Mistral Codestral writes the code. A Reviewer + Tester loop validates it.

Rule-based pre-pass handles 80% — agents tackle the hard stuff.

@AIatAMD #ROCm #AMDDeveloper #AIAgents #Hackathon

---

## LinkedIn (longer version)

Just shipped the core of my AMD hackathon project: an automated CUDA-to-ROCm migration agent.

**How it works:**
1. Static analyzer detects CUDA API calls, env vars, imports, and backend settings
2. Rule-based pre-pass applies ~30 deterministic high-confidence replacements (cuDNN→MIOpen, CUDA_VISIBLE_DEVICES→HIP_VISIBLE_DEVICES, etc.)
3. DeepSeek-R1-Distill-Qwen-32B — running on an AMD MI300X GPU — reasons through the remaining complex cases and produces a step-by-step migration plan
4. Mistral Codestral implements the plan as working Python code
5. A Reviewer agent validates correctness; a Tester runs AST/import checks
6. The loop terminates automatically when ALL_TESTS_PASSED

The interesting part: DeepSeek-R1's chain-of-thought reasoning is genuinely useful here. It catches things like "torch.cuda.amp still works on ROCm — don't change the device string to 'hip'" that a simple regex can't reason about.

Built with ag2 (multi-agent framework), vLLM on ROCm, and a lot of debugging 😅

@lablab | @AIatAMD | AMD Developer
#ROCm #AMDDeveloper #AIAgents #Hackathon #DeepSeek #MistralAI
