import os
import time

from datasets import load_dataset
from openai import OpenAI
import pyarrow as pa
import pyarrow.parquet as pq


DATA_FILES = "/home/taiga/ml_lake/translate-nano-dataset/en/c4-train.*.json"
OUTPUT_DIR = os.path.join("./", "c4-ja-parquet")
MAX_SHARD_BYTES = 300 * 1024 * 1024
MAX_SAMPLES = 100

MODEL = "openai/gpt-oss-20b"
API_KEY = os.environ.get("OPENAI_API_KEY", "dummy")
BASE_URL = os.environ.get("OPENAI_BASE_URL", "http://localhost:8000/v1")
BASE_URLS = [
    url.strip()
    for url in os.environ.get("OPENAI_BASE_URLS", "").split(",")
    if url.strip()
]
if not BASE_URLS:
    BASE_URLS = [BASE_URL]
PROGRESS_PATH = os.path.join(OUTPUT_DIR, "progress.parquet")


class LLMClientPool:
    def __init__(self, base_urls: list[str], api_key: str) -> None:
        self._clients = [
            (base_url, OpenAI(api_key=api_key, base_url=base_url))
            for base_url in base_urls
        ]
        self._next_index = 0

    def _next_client(self) -> tuple[str, OpenAI]:
        base_url, client = self._clients[self._next_index]
        self._next_index = (self._next_index + 1) % len(self._clients)
        return base_url, client

    def call(self, messages: list[dict[str, str]]) -> str:
        last_error: Exception | None = None
        for _ in range(len(self._clients)):
            base_url, client = self._next_client()
            try:
                response = client.chat.completions.create(
                    model=MODEL,
                    messages=messages,
                    temperature=0.3,
                    max_tokens=4096,
                )
                content = response.choices[0].message.content
                return content if content is not None else ""
            except Exception as exc:
                last_error = exc
                print(f"request failed for {base_url}: {exc}")
                time.sleep(0.5)
        if last_error:
            raise last_error
        raise RuntimeError("no available base_url")


LLM_POOL = LLMClientPool(BASE_URLS, API_KEY)


def call_llm(user_input: str) -> str:
    messages = [
        {
            "role": "system",
            "content": (
                "You are a professional translator who translates English into Japanese. "
                "Please translate the received text directly and accurately into Japanese. "
                "Please keep the tone of the original text. Output only the translated text. "
                "Do not use Markdown formatting. Be sure to output the translated Japanese text."
            ),
        },
        {
            "role": "user",
            "content": "The quiet cafe buzzed with ideas as sunlight spilled across half-finished notebooks.",
        },
        {
            "role": "assistant",
            "content": "静かなカフェにはアイデアが満ち、差し込む日差しが書きかけのノートを照らしていた。",
        },
        {
            "role": "user",
            "content": "She took a deep breath and stepped forward, trusting that curiosity would guide her way.",
        },
        {
            "role": "assistant",
            "content": "彼女は深く息を吸い、好奇心が自分を導いてくれると信じて一歩踏み出した。",
        },
        {
            "role": "user",
            "content": user_input,
        },
    ]

    return LLM_POOL.call(messages)


def _normalize_text(value: str | None) -> str:
    return value if value is not None else ""


def _estimate_bytes(text_en: str | None, text_ja: str | None) -> int:
    text_en = _normalize_text(text_en)
    text_ja = _normalize_text(text_ja)
    return len(text_en.encode("utf-8")) + len(text_ja.encode("utf-8"))


class ParquetShardWriter:
    def __init__(self, output_dir: str, max_bytes: int, start_shard_id: int) -> None:
        self.output_dir = output_dir
        self.max_bytes = max_bytes
        self.shard_id = start_shard_id
        self.buffer_en = []
        self.buffer_ja = []
        self.current_bytes = 0
        os.makedirs(self.output_dir, exist_ok=True)

    def add(self, text_en: str | None, text_ja: str | None) -> bool:
        text_en = _normalize_text(text_en)
        text_ja = _normalize_text(text_ja)
        self.buffer_en.append(text_en)
        self.buffer_ja.append(text_ja)
        self.current_bytes += _estimate_bytes(text_en, text_ja)
        if self.current_bytes >= self.max_bytes:
            self.flush()
            return True
        return False

    def flush(self) -> None:
        if not self.buffer_en:
            return
        table = pa.table({"text_en": self.buffer_en, "text_ja": self.buffer_ja})
        output_path = os.path.join(
            self.output_dir,
            f"c4-train-ja-{self.shard_id:05d}.parquet",
        )
        pq.write_table(table, output_path, compression="zstd")
        self.shard_id += 1
        self.buffer_en = []
        self.buffer_ja = []
        self.current_bytes = 0


def _find_next_shard_id(output_dir: str) -> int:
    if not os.path.isdir(output_dir):
        return 0
    max_id = -1
    prefix = "c4-train-ja-"
    suffix = ".parquet"
    for name in os.listdir(output_dir):
        if not (name.startswith(prefix) and name.endswith(suffix)):
            continue
        shard_str = name[len(prefix) : -len(suffix)]
        try:
            shard_id = int(shard_str)
        except ValueError:
            continue
        max_id = max(max_id, shard_id)
    return max_id + 1


def _load_progress(progress_path: str) -> dict[str, int]:
    if not os.path.exists(progress_path):
        return {"processed": 0, "shard_id": 0}
    table = pq.read_table(progress_path)
    if table.num_rows < 1:
        return {"processed": 0, "shard_id": 0}
    row = table.to_pylist()[0]
    return {
        "processed": int(row.get("processed", 0)),
        "shard_id": int(row.get("shard_id", 0)),
    }


def _write_progress(progress_path: str, processed: int, shard_id: int) -> None:
    table = pa.table(
        {
            "processed": [processed],
            "shard_id": [shard_id],
            "updated_at": [int(time.time())],
        }
    )
    pq.write_table(table, progress_path, compression="zstd")


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    progress = _load_progress(PROGRESS_PATH)
    resume_from = progress["processed"]
    next_shard_id = _find_next_shard_id(OUTPUT_DIR)
    start_shard_id = max(progress["shard_id"], next_shard_id)
    if resume_from >= MAX_SAMPLES:
        print(f"done: {resume_from} samples")
        return

    dataset = load_dataset(
        "json",
        data_files=DATA_FILES,
        streaming=True,
    )

    writer = ParquetShardWriter(OUTPUT_DIR, MAX_SHARD_BYTES, start_shard_id)
    processed = resume_from

    for idx, sample in enumerate(dataset["train"]):
        if idx < resume_from:
            continue
        if processed >= MAX_SAMPLES:
            break
        text = sample.get("text", "")
        if not text:
            flushed = writer.add("", "")
            processed += 1
            if flushed:
                _write_progress(PROGRESS_PATH, processed, writer.shard_id)
            continue
        translated = call_llm(text)
        flushed = writer.add(text, translated)
        processed += 1
        if flushed:
            _write_progress(PROGRESS_PATH, processed, writer.shard_id)
        if processed % 10 == 0:
            print(f"processed {processed}/{MAX_SAMPLES}")

    writer.flush()
    _write_progress(PROGRESS_PATH, processed, writer.shard_id)
    print(f"done: {processed} samples")


if __name__ == "__main__":
    main()
