CUDA_VISIBLE_DEVICES=0 uv run vllm serve openai/gpt-oss-20b \
  --gpu-memory-utilization 0.8 \
  --port 8000
