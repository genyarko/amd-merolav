#!/bin/bash
set -e

HF_TOKEN="${1:-}"

if [ -z "$HF_TOKEN" ]; then
  echo "Usage: bash setup_gpu.sh <HF_TOKEN>"
  echo "Example: bash setup_gpu.sh hf_xxx"
  exit 1
fi

MODELS=(
  "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
  "Qwen/Qwen2.5-Coder-32B-Instruct"
)

echo "=== Installing dependencies ==="
pip install -q huggingface_hub[cli] vllm

echo "=== Logging into HuggingFace ==="
huggingface-cli login --token "$HF_TOKEN"

echo "=== Verifying ROCm GPU access ==="
rocm-smi
python3 -c "import torch; print('GPUs found:', torch.cuda.device_count()); print('GPU:', torch.cuda.get_device_name(0))"

echo "=== Downloading models in parallel ==="
mkdir -p /models/logs

for MODEL in "${MODELS[@]}"; do
  NAME=$(basename "$MODEL")
  echo "--- Starting download: $MODEL ---"
  huggingface-cli download "$MODEL" \
    --local-dir "/models/$NAME" \
    --local-dir-use-symlinks False \
    > "/models/logs/${NAME}.log" 2>&1 &
done

echo "Both downloads running in background. Waiting for completion..."
wait
echo ""
echo "=== All models downloaded ==="
