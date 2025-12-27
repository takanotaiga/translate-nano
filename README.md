# Translate Nano

## Start SDG

```bash
OPENAI_BASE_URLS="http://localhost:8000/v1,http://localhost:8001/v1,http://localhost:8002/v1,http://localhost:8003/v1" \
uv run dataset/translate_parquet_to_parquet.py --data-files "./c4-en-filtered-parquet-64/*.parquet" --output-dir c4-en-ja-64
```
