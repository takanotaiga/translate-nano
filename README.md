# Translate Nano

## Start SDG

```bash
OPENAI_BASE_URLS="http://localhost:8000/v1,http://localhost:8001/v1,http://localhost:8002/v1,http://localhost:8003/v1" \
  uv run dataset/translate_c4_to_parquet.py --backend openai
```

```bash
OPENAI_BASE_URLS="http://localhost:8000/v1,http://localhost:8001/v1,http://localhost:8002/v1,http://localhost:8003/v1" \
  uv run dataset/translate_c4_to_parquet.py --backend plamo
```