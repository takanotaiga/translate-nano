CUDA_VISIBLE_DEVICES=1 uv run vllm serve openai/gpt-oss-20b \
  --dtype bfloat16 \
  --kv-cache-dtype fp8 \
  --enable-prefix-caching \
  --gpu-memory-utilization 0.8 \
  --block-size 16 \
  --max-num-seqs 512 \
  --max-num-batched-tokens 256k \
  --disable-log-requests \
  --host 0.0.0.0 --port 8001