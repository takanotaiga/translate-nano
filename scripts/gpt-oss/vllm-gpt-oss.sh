#!/usr/bin/env bash
set -euo pipefail
mkdir -p logs   # ← これを追加
GPUS=${GPUS:-"0,1,2,3"}
BASE_PORT=${BASE_PORT:-8000}
IFS=',' read -ra GPU_ARR <<< "$GPUS"
i=0
for gpu in "${GPU_ARR[@]}"; do
    port=$((BASE_PORT + i))
    CUDA_VISIBLE_DEVICES="$gpu" uv run vllm serve openai/gpt-oss-20b \
        --gpu-memory-utilization 0.8 --port "$port" \
        >"logs/vllm-${gpu}.log" 2>&1 &
    i=$((i+1))
done
wait
