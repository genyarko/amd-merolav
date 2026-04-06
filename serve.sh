#!/bin/bash
set -e

echo "=== Starting DeepSeek-R1 (reasoning/planner) on port 8000 ==="
python3 -m vllm.entrypoints.openai.api_server \
  --model /models/DeepSeek-R1-Distill-Qwen-32B \
  --host 0.0.0.0 \
  --port 8000 \
  --dtype float16 \
  --max-model-len 32768 &

PLANNER_PID=$!
echo "Planner PID: $PLANNER_PID"

echo "=== Starting Qwen2.5-Coder (executor) on port 8001 ==="
python3 -m vllm.entrypoints.openai.api_server \
  --model /models/Qwen2.5-Coder-32B-Instruct \
  --host 0.0.0.0 \
  --port 8001 \
  --dtype float16 \
  --max-model-len 32768 &

EXECUTOR_PID=$!
echo "Executor PID: $EXECUTOR_PID"

echo ""
echo "=== Both models running ==="
echo "  Planner  (DeepSeek-R1):     http://localhost:8000/v1"
echo "  Executor (Qwen2.5-Coder):   http://localhost:8001/v1"
echo ""
echo "Press Ctrl+C to stop both."

# Stop both on exit
trap "kill $PLANNER_PID $EXECUTOR_PID" EXIT
wait
