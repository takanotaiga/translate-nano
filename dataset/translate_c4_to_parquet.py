import os

from datasets import load_dataset
from openai import OpenAI
import pyarrow as pa
import pyarrow.parquet as pq


DATA_FILES = "/home/taiga/ml_lake/translate-nano-dataset/en/c4-train.*.json"
OUTPUT_DIR = os.path.join("./", "c4-ja-parquet")
MAX_SHARD_BYTES = 300 * 1024 * 1024
MAX_SAMPLES = 10

MODEL = "openai/gpt-oss-20b"
API_KEY = os.environ.get("OPENAI_API_KEY", "dummy")
BASE_URL = os.environ.get("OPENAI_BASE_URL", "http://localhost:8000/v1")


client = OpenAI(
    api_key=API_KEY,
    base_url=BASE_URL,
)


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

    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=0.3,
        max_tokens=4096,
    )

    content = response.choices[0].message.content
    return content if content is not None else ""


def _normalize_text(value: str | None) -> str:
    return value if value is not None else ""


def _estimate_bytes(text_en: str | None, text_ja: str | None) -> int:
    text_en = _normalize_text(text_en)
    text_ja = _normalize_text(text_ja)
    return len(text_en.encode("utf-8")) + len(text_ja.encode("utf-8"))


class ParquetShardWriter:
    def __init__(self, output_dir: str, max_bytes: int) -> None:
        self.output_dir = output_dir
        self.max_bytes = max_bytes
        self.shard_id = 0
        self.buffer_en = []
        self.buffer_ja = []
        self.current_bytes = 0
        os.makedirs(self.output_dir, exist_ok=True)

    def add(self, text_en: str | None, text_ja: str | None) -> None:
        text_en = _normalize_text(text_en)
        text_ja = _normalize_text(text_ja)
        self.buffer_en.append(text_en)
        self.buffer_ja.append(text_ja)
        self.current_bytes += _estimate_bytes(text_en, text_ja)
        if self.current_bytes >= self.max_bytes:
            self.flush()

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


def main() -> None:
    dataset = load_dataset(
        "json",
        data_files=DATA_FILES,
        streaming=True,
    )

    writer = ParquetShardWriter(OUTPUT_DIR, MAX_SHARD_BYTES)
    processed = 0

    for sample in dataset["train"]:
        if processed >= MAX_SAMPLES:
            break
        text = sample.get("text", "")
        if not text:
            writer.add("", "")
            processed += 1
            continue
        translated = call_llm(text)
        writer.add(text, translated)
        processed += 1
        if processed % 10 == 0:
            print(f"processed {processed}/{MAX_SAMPLES}")

    writer.flush()
    print(f"done: {processed} samples")


if __name__ == "__main__":
    main()
