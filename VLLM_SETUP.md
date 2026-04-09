# vLLM Setup on MI300X Droplet

## SSH in
```bash
ssh root@10.128.0.2
```

## Check for existing containers
```bash
docker ps
```

## Remove old vLLM container if exists
```bash
docker rm -f vllm
```

## Start vLLM
```bash
~/start_vllm.sh
```

### Script contents (`~/start_vllm.sh`)
If the script is missing, recreate with `nano ~/start_vllm.sh`:
```bash
#!/bin/bash
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
Then `chmod +x ~/start_vllm.sh`.

## Watch logs (wait for "Uvicorn running on...")
```bash
docker logs -f vllm
```

## Verify model name
```bash
curl http://localhost:8001/v1/models
```
The model ID served by vLLM is `deepseek-ai/DeepSeek-R1-Distill-Qwen-32B`.

## Test from Windows
```bash
python -m cli.main tests/fixtures/sample_cuda_simple.py --backend self-hosted --no-cache --force-agents
```
The `.env` file should have:
```
PLANNER_BASE_URL=http://129.212.182.227:8001/v1
PLANNER_MODEL=deepseek-ai/DeepSeek-R1-Distill-Qwen-32B
```
- **Public IP**: `129.212.182.227`
- **Private IP**: `10.128.0.2` (only works from within DigitalOcean network)

## Notes
- **Do NOT** `pip install vllm` — the pip version is CUDA-only. Must use Docker.
- Docker image `vllm/vllm-openai-rocm:v0.17.1` is already pulled on the droplet.
- Port 8000 is used by the `rocm` container, so vLLM uses **port 8001**.
- Model name must be `deepseek-ai/DeepSeek-R1-Distill-Qwen-32B` (NOT `/models/DeepSeek-R1-Distill-Qwen-32B`).
- Firewall (UFW) may need a rule for port 8001:
  ```bash
  ufw allow from <YOUR_IP> to any port 8001
  ufw deny 8001
  ```
