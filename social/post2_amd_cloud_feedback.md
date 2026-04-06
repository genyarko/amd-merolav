# Post 2 — "Developer Experience Feedback on AMD Developer Cloud"
**Platform:** X (Twitter) + LinkedIn
**Timing:** Publish after demo recording / submission
**Attach:** Screenshot of rocm-smi showing MI300X + model loaded

---

## X (Twitter) — thread format

🧵 Honest developer feedback on @AIatAMD MI300X cloud after running a 32B LLM inference workload on it for a hackathon:

1/ Provisioning: DigitalOcean AMD Developer Cloud requires polling the API for availability — MI300X droplets aren't always in stock. Took ~10 min once available. Clean API, doctl works fine.

2/ ROCm + PyTorch: Don't pip install torch — you'll get a CUDA build. Use the ROCm wheel: `pip install torch --index-url https://download.pytorch.org/whl/rocm6.2`. After that, torch.cuda.is_available() = True on ROCm. Same device strings as CUDA. This is genuinely good.

3/ vLLM: The pip package is CUDA-compiled and looks for libcudart.so.12. Use Docker instead: `docker run rocm/vllm:latest`. This worked first try, no custom builds needed.

4/ Memory: MI300X has 192GB HBM3. DeepSeek-R1-Distill-Qwen-32B uses ~61GB loaded. With --gpu-memory-utilization 0.95 you get ~109GB KV cache. That's enormous for long context.

5/ vs NVIDIA: PyTorch compatibility is excellent — almost no code changes needed for inference. Main gotcha is the tooling layer (vLLM install, pip wheel). Once it's running, it runs.

Overall: Hardware is impressive. Tooling is 80% there. Would be great to have an official ROCm-ready vLLM pip package.

@lablab #ROCm #AMDDeveloper #MI300X #Hackathon

---

## LinkedIn (long form)

**AMD MI300X Developer Cloud — Honest Technical Feedback**

Just spent several days running a multi-agent LLM inference workload on AMD's MI300X via DigitalOcean Developer Cloud. Here's the unfiltered developer experience:

**✅ What worked great:**

- **PyTorch on ROCm is seamless.** `torch.cuda.is_available()` returns True, device strings stay `"cuda"`, `.cuda()` just works. If you're migrating existing PyTorch code, 90% of it runs unchanged. This is the right design decision.
- **192GB HBM3 is a beast.** Loaded DeepSeek-R1-Distill-Qwen-32B (61GB) with 109GB free for KV cache. Running long chain-of-thought reasoning with massive context windows is where this hardware shines vs consumer NVIDIA GPUs.
- **Docker + rocm/vllm image is solid.** Pulled, ran, served — no custom compilation. The container just works.
- **rocm-smi** gives you everything nvidia-smi does. Familiar interface, good GPU utilization visibility.

**⚠️ Friction points:**

- **pip install torch gets the CUDA build.** You have to know to use `--index-url https://download.pytorch.org/whl/rocm6.2`. Not discoverable for first-timers. A ROCm-specific pip package name would help.
- **vLLM pip package is CUDA-only.** The official vLLM pip release looks for `libcudart.so.12` and fails silently on ROCm. Docker is the workaround, but it adds complexity. An official ROCm vLLM wheel would be a big quality-of-life improvement.
- **Droplet availability requires polling.** MI300X instances aren't always available — need to retry the DigitalOcean API until one comes up. A waitlist or notification system would help for hackathon time pressure.
- **No persistent storage between droplets.** Model downloads (62GB each) happen fresh every time unless you snapshot the droplet. Snapshotting a 462GB disk takes time and costs storage.

**The bottom line:**

AMD's ROCm ecosystem has come a long way. For pure PyTorch workloads, it's production-ready. The rough edges are in the install tooling layer, not the runtime. The MI300X hardware itself is genuinely impressive for large model inference — the memory bandwidth and capacity are class-leading.

Recommended for: Large model serving, long-context inference, multi-GPU workloads.
Not yet seamless for: First-time setup without prior ROCm experience.

AMD Developer | @lablab
#ROCm #AMDDeveloper #MI300X #LLMInference #DeveloperExperience #Hackathon
