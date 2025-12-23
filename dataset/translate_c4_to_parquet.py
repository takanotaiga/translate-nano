import os
import queue
import threading
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


def _build_messages(user_input: str) -> list[dict[str, str]]:
    return [
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


class LLMWorkerPool:
    def __init__(self, base_urls: list[str], api_key: str, max_retries: int = 2) -> None:
        self._tasks: queue.Queue[tuple[int, str, int] | None] = queue.Queue()
        self._results: queue.Queue[tuple[int, str, str, Exception | None]] = queue.Queue()
        self._threads: list[threading.Thread] = []
        self._max_retries = max_retries
        for base_url in base_urls:
            thread = threading.Thread(
                target=self._worker,
                args=(base_url, api_key),
                daemon=True,
            )
            thread.start()
            self._threads.append(thread)

    @property
    def results(self) -> queue.Queue[tuple[int, str, str, Exception | None]]:
        return self._results

    def submit(self, index: int, text: str) -> None:
        self._tasks.put((index, text, 0))

    def close(self) -> None:
        for _ in self._threads:
            self._tasks.put(None)
        for thread in self._threads:
            thread.join()

    def _worker(self, base_url: str, api_key: str) -> None:
        client = OpenAI(api_key=api_key, base_url=base_url)
        while True:
            task = self._tasks.get()
            if task is None:
                self._tasks.task_done()
                break
            index, text, attempt = task
            try:
                response = client.chat.completions.create(
                    model=MODEL,
                    messages=_build_messages(text),
                    temperature=0.3,
                    max_tokens=4096,
                )
                content = response.choices[0].message.content
                translated = content if content is not None else ""
                self._results.put((index, text, translated, None))
            except Exception as exc:
                if attempt < self._max_retries:
                    time.sleep(0.5 * (attempt + 1))
                    self._tasks.put((index, text, attempt + 1))
                else:
                    self._results.put((index, text, "", exc))
            finally:
                self._tasks.task_done()


def _normalize_text(value: str | None) -> str:
    return value if value is not None else ""


def _estimate_bytes(text_en: str | None, text_ja: str | None) -> int:
    text_en = _normalize_text(text_en)
    text_ja = _normalize_text(text_ja)
    return len(text_en.encode("utf-8")) + len(text_ja.encode("utf-8"))


class ParquetShardWriter:
    def __init__(
        self,
        output_dir: str,
        max_bytes: int,
        start_shard_id: int,
        start_written: int,
    ) -> None:
        self.output_dir = output_dir
        self.max_bytes = max_bytes
        self.shard_id = start_shard_id
        self.buffer_en = []
        self.buffer_ja = []
        self.current_bytes = 0
        self.written_rows = start_written
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
        self.written_rows += len(self.buffer_en)
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


def _count_existing_rows(output_dir: str) -> int:
    if not os.path.isdir(output_dir):
        return 0
    total_rows = 0
    prefix = "c4-train-ja-"
    suffix = ".parquet"
    for name in os.listdir(output_dir):
        if not (name.startswith(prefix) and name.endswith(suffix)):
            continue
        path = os.path.join(output_dir, name)
        try:
            parquet_file = pq.ParquetFile(path)
            total_rows += parquet_file.metadata.num_rows
        except Exception as exc:
            print(f"failed to read {path}: {exc}")
    return total_rows


def _load_progress(progress_path: str) -> dict[str, int]:
    if not os.path.exists(progress_path):
        return {"processed": 0, "written": 0, "shard_id": 0}
    table = pq.read_table(progress_path)
    if table.num_rows < 1:
        return {"processed": 0, "written": 0, "shard_id": 0}
    row = table.to_pylist()[0]
    return {
        "processed": int(row.get("processed", 0)),
        "written": int(row.get("written", row.get("processed", 0))),
        "shard_id": int(row.get("shard_id", 0)),
    }


def _write_progress(
    progress_path: str,
    processed: int,
    written: int,
    shard_id: int,
) -> None:
    table = pa.table(
        {
            "processed": [processed],
            "written": [written],
            "shard_id": [shard_id],
            "updated_at": [int(time.time())],
        }
    )
    pq.write_table(table, progress_path, compression="zstd")


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    progress = _load_progress(PROGRESS_PATH)
    existing_rows = _count_existing_rows(OUTPUT_DIR)
    if progress["written"] != existing_rows:
        print(
            "progress mismatch: "
            f"progress_written={progress['written']} existing_rows={existing_rows}. "
            "using existing_rows for resume."
        )
    resume_from = existing_rows
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

    writer = ParquetShardWriter(
        OUTPUT_DIR,
        MAX_SHARD_BYTES,
        start_shard_id,
        existing_rows,
    )
    processed = resume_from
    pool = LLMWorkerPool(BASE_URLS, API_KEY)
    max_in_flight = max(1, len(BASE_URLS) * 2)
    pending: dict[int, tuple[str, str]] = {}
    in_flight = 0
    submitted = resume_from
    expected_index = resume_from
    dataset_iter = iter(dataset["train"])

    try:
        skipped = 0
        while skipped < resume_from:
            try:
                next(dataset_iter)
            except StopIteration:
                print(f"done: {processed} samples")
                return
            skipped += 1

        while expected_index < MAX_SAMPLES:
            while submitted < MAX_SAMPLES and in_flight < max_in_flight:
                try:
                    sample = next(dataset_iter)
                except StopIteration:
                    submitted = MAX_SAMPLES
                    break
                text = sample.get("text", "")
                if not text:
                    pending[submitted] = ("", "")
                else:
                    pool.submit(submitted, text)
                    in_flight += 1
                submitted += 1

            wrote_any = False
            while expected_index in pending:
                text_en, text_ja = pending.pop(expected_index)
                writer.add(text_en, text_ja)
                processed = expected_index + 1
                _write_progress(
                    PROGRESS_PATH,
                    processed,
                    writer.written_rows,
                    writer.shard_id,
                )
                expected_index += 1
                wrote_any = True
                if processed % 10 == 0:
                    print(f"processed {processed}/{MAX_SAMPLES}")

            if expected_index >= MAX_SAMPLES:
                break
            if wrote_any:
                continue
            if in_flight == 0 and submitted >= MAX_SAMPLES:
                break

            index, text_en, text_ja, error = pool.results.get()
            in_flight -= 1
            if error is not None:
                raise error
            pending[index] = (text_en, text_ja)
    finally:
        pool.close()

    writer.flush()
    _write_progress(
        PROGRESS_PATH,
        processed,
        writer.written_rows,
        writer.shard_id,
    )
    print(f"done: {processed} samples")


if __name__ == "__main__":
    main()
