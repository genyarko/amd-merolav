# Demo Video Script — 2-3 minutes

## Setup before recording
1. Terminal window sized wide, font size 16+
2. Run once beforehand so model is warm (faster response)
3. Have `demo_complex.py` open in a second window to show the input

---

## Script

**[0:00–0:20] — Show the problem**
Open `demo/demo_complex.py` in your editor.
Say: "This is a real PyTorch training script using NVIDIA CUDA — custom kernels, cuDNN settings, NVTX profiling, pycuda. We want to run this on an AMD MI300X GPU using ROCm. Manually migrating this takes 30+ minutes and is easy to get wrong."

**[0:20–0:35] — Run the tool**
Switch to terminal. Run:
```
python -m cli.main demo\demo_complex.py --verbose --force-agents
```
Say: "Let's run the migration agent."

**[0:35–1:00] — Rule-based pass**
Point to the "Applying rule-based migrations" output.
Say: "First, a static analyzer detects all CUDA usage. A rule-based pass instantly applies 9 high-confidence substitutions — environment variables, MIOpen settings, import replacements — no LLM needed."

**[1:00–1:45] — Planner reasoning**
Point to the DeepSeek-R1 Planner panel appearing.
Say: "Now DeepSeek-R1, running on the AMD MI300X, reasons through the remaining complex patterns. Watch the chain-of-thought — it's identifying the NVTX profiling issue, the custom kernel, the flash attention flag."
Let the plan render fully. Say: "It produced a 10-step migration plan in about 20 seconds."

**[1:45–2:15] — Executor + validation**
Point to Executor panel.
Say: "Mistral Codestral implements the plan. A Reviewer validates the logic. Then an automated Tester runs AST and import checks."
Point to ALL_TESTS_PASSED.
Say: "ALL_TESTS_PASSED — the pipeline terminates automatically."

**[2:15–2:45] — Show the diff**
Scroll to the diff output.
Say: "Here's the unified diff. 13 lines changed — pycuda replaced with hip-python, cuDNN replaced with MIOpen, CUDA env vars replaced, NVTX migrated to torch.hip.nvtx. The code is now ready to run on AMD ROCm."

**[2:45–3:00] — Close**
Say: "Full pipeline: 31.5 seconds. Open source at [your GitHub]. Built for the lablab AMD hackathon using a real MI300X on DigitalOcean Developer Cloud."

---

## Recording checklist
- [ ] `$env:PYTHONIOENCODING = "utf-8"` set before running (or use benchmark.ps1)
- [ ] Run once to warm the model before hitting record
- [ ] rocm-smi screenshot ready in second window
- [ ] Resize terminal to show panels cleanly
