# Post 3 (Bonus) — "MI300X vs API: LLM Agent Inference Benchmark"
**Platform:** X (Twitter) + LinkedIn
**Timing:** Publish last — after you have real timing numbers
**Attach:** Screenshot of rocm-smi during inference + terminal showing timing

---

## HOW TO GET THE NUMBERS

Run this before recording to capture real latency:

```powershell
# Time the full pipeline
Measure-Command {
    python -m cli.main demo\demo_complex.py --verbose --force-agents
} | Select-Object TotalSeconds
```

On the droplet during inference, grab rocm-smi:
```bash
watch -n 1 rocm-smi
```
Screenshot when GPU utilization spikes during DeepSeek-R1 prefill/decode.

---

## X (Twitter)

Just benchmarked MI300X self-hosted inference vs Mistral free API for my CUDA→ROCm agent:

Full pipeline (149-line CUDA file): **31.5 seconds**
- DeepSeek-R1-32B on MI300X: handles planning + reasoning
- Mistral Codestral API: code generation + review
- VRAM: 91% used (175GB / 192GB — model + KV cache pre-allocated by vLLM)

For a 32B reasoning model doing chain-of-thought migration planning, 31s end-to-end is fast.

@AIatAMD @lablab #ROCm #MI300X #LLMBenchmark #AMDDeveloper #Hackathon

---

## LinkedIn (long form)

**Benchmark: AMD MI300X Self-Hosted vs Mistral API for Multi-Agent LLM Inference**

For my CUDA→ROCm migration agent I split the work across two backends:
- **Planner**: DeepSeek-R1-Distill-Qwen-32B, self-hosted on AMD MI300X via vLLM
- **Executor**: Mistral Codestral, cloud API

Numbers on a real migration task (demo_complex.py, 149 lines of CUDA code):

| Stage | Backend | Result |
|-------|---------|--------|
| Full pipeline (planner + executor + review + test) | MI300X + Mistral API | **31.5 seconds** |
| VRAM used | MI300X | 175GB / 192GB (91%) |
| KV cache pre-allocated | vLLM | ~109GB headroom for long context |
| GPU clock | MI300X | 2103 MHz SCLK / 900 MHz MCLK |

**What the 91% VRAM means:**
vLLM pre-allocates the KV cache at startup with `--gpu-memory-utilization 0.95`. The model weights are ~61GB; the rest (~114GB) is reserved as KV cache. This is why the MI300X's 192GB HBM3 matters — you can run a 32B model AND keep an enormous context window without fragmentation.

**DeepSeek-R1's chain-of-thought on MI300X:**
R1 outputs 500-1500 tokens of `<think>` reasoning before the final plan. On the MI300X, this generates fast enough that the reasoning step doesn't dominate the pipeline. The quality improvement over a direct-answer model is noticeable — it catches ROCm-specific gotchas that simpler models miss.

**For production:**
Both models (DeepSeek-R1-32B + Codestral-22B) fit together on the MI300X (~122GB combined), which would eliminate the Mistral API latency entirely. The demo splits them to show the hybrid self-hosted + API architecture.

AMD Developer | @lablab
#ROCm #MI300X #LLMInference #AIBenchmark #AMDDeveloper #Hackathon
