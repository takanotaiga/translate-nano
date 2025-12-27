CUDA_VISIBLE_DEVICES=1 uv run vllm serve openai/gpt-oss-20b \
  --gpu-memory-utilization 0.8 \
  --port 8001
