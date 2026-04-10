# Post 5 — "Pluggable AI Executors: Gemma 4 on My CUDA Migration Agent"
**Platform:** LinkedIn + X
**Timing:** Publish after confirming Gemma 4 test results
**Attach:** Screenshot of terminal showing Gemma 4 executor label + ALL_TESTS_PASSED + 97% quality

---

## LinkedIn

Just shipped a feature I've been wanting to build: pluggable executors for my CUDA-to-ROCm migration agent.

The problem: I originally hardcoded Mistral Codestral as the code generation model. That worked, but it locked users into one provider. What if you want to use a self-hosted model? Or a different API? Or Google's new Gemma 4?

Now you can:

```
rocm-migrate file.py --executor-url https://generativelanguage.googleapis.com/v1beta/openai/ --executor-model gemma-4-31b-it --executor-key $GOOGLE_API_KEY
```

Any OpenAI-compatible endpoint works — Google AI Studio, Ollama, vLLM, Together AI, whatever you want.

But first — I tried the obvious thing: run both DeepSeek-R1-32B and Gemma 4 31B on the same MI300X.

192GB of VRAM should be enough for two 32B models, right? No.

vLLM pre-allocates 90% of VRAM for the first model's KV cache. DeepSeek-R1-32B was already running, consuming ~174GB (weights + KV cache). That left 17.8GB free. Gemma 4 31B needs ~62GB minimum. Even the MoE variant (Gemma 4 26B-A4B, only 3.8B active parameters) still loads full 26B weights into VRAM. Same OOM error for both.

Lesson: on a single MI300X, you can run ONE large model with a big KV cache, or TWO smaller models with reduced context. You can't have both without quantization or a multi-GPU setup.

So I pivoted: keep DeepSeek-R1 self-hosted for planning, use Gemma 4 via Google AI Studio API for code execution. Best of both worlds.

I tested Gemma 4 31B as the executor. Here's what I found:

1/ It works. Gemma 4 successfully migrated complex CUDA code (pycuda, NVTX profiling, custom kernels, mixed precision) on its first LLM turn. The Reviewer approved it, the Tester passed all checks.

2/ It's smart about reviewing. When acting as Reviewer, Gemma 4 correctly flagged that torch.backends.cuda.enable_flash_sdp isn't a standard PyTorch API. Good catch.

3/ It doesn't always hit 100%. On one run, Gemma 4 kept torch.cuda.nvtx in a try-except instead of replacing it with hip.nvtx — a valid defensive choice, but the quality scorer correctly flags it at 75% confidence since the CUDA symbol is still present.

Technical challenges I solved along the way:

- AutoGen's round-robin GroupChat was consuming the seed message as the Executor's "turn," so the LLM was never called. Fixed with a TaskDispatcher proxy agent.
- The code extractor was finding python blocks embedded in the prompt (input code), not executor output. Fixed by skipping the seed message.
- Google AI Studio has a 16k tokens/min rate limit on Gemma 4. Added exponential backoff retry (15s/30s/60s) so the pipeline waits instead of failing.

The quality verification now checks whether the agent actually resolved each issue — if the CUDA symbol is gone from the output AND validation passed, confidence goes to 100%. No more flat heuristic boosts.

Architecture: DeepSeek-R1-32B (planner, self-hosted on MI300X) + any executor model (configurable). The planner reasons through the migration, the executor writes the code, the reviewer checks it, the tester validates.

Built on AMD MI300X (192GB HBM3) via DigitalOcean Developer Cloud.

#ROCm #AMDDeveloper #Gemma4 #CUDA #PyTorch #AIAgents #OpenSource #BuildInPublic @lablab @AIatAMD

---

## X (Twitter)

Tried running DeepSeek-R1-32B + Gemma 4 31B on the same MI300X (192GB). Nope — vLLM's KV cache eats 174GB, leaving 17GB. OOM.

Pivoted: DeepSeek self-hosted for planning, Gemma 4 via API for code gen. Built a pluggable executor system so any OpenAI-compatible model works.

Gemma 4 migrated complex CUDA code on the first try.

@AIatAMD @lablab #ROCm #Gemma4 #BuildInPublic
