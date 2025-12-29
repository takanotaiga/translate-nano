# Translate Nano

## Run translation (parquet -> parquet)

```bash
OPENAI_BASE_URLS="http://localhost:8000/v1,http://localhost:8001/v1,http://localhost:8002/v1,http://localhost:8003/v1" \
uv run dataset/translate_parquet_to_parquet.py --data-files "./c4-en-filtered-parquet-64/*.parquet" --output-dir c4-en-ja-64
```

### OpenAI API backend

```bash
export OPENAI_API_KEY="..."
export OPENAI_API_BASE_URL="https://api.openai.com/v1"
uv run dataset/translate_parquet_to_parquet.py \
  --backend openai-api \
  --data-files "./c4-en-filtered-parquet-64/*.parquet" \
  --output-dir c4-en-ja-64
```

### Hybrid (gpt-oss + OpenAI API)

```bash
export OPENAI_API_KEY="..."
export OPENAI_API_BASE_URL="https://api.openai.com/v1"
uv run dataset/translate_parquet_to_parquet.py \
  --backend hybrid \
  --data-files "./c4-en-filtered-parquet-64/*.parquet" \
  --output-dir c4-en-ja-64
```
