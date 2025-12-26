import argparse
import glob
import json
import os
import queue
import subprocess
import threading
import time
from typing import Sequence

from datasets import load_dataset
from openai import OpenAI
import pyarrow as pa
import pyarrow.parquet as pq
try:
    from tqdm import tqdm
except Exception:  # pragma: no cover - optional dependency
    tqdm = None


DATA_FILES = "/home/taiga/ml_lake/translate-nano-dataset/en/c4-train.*.json"
# OUTPUT_DIR = os.path.join("/home/taiga/ml_lake/translate-nano-dataset/", "c4-ja-parquet")
OUTPUT_DIR = os.path.join("./", "c4-ja-parquet")
MAX_SAMPLES = -1
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "10"))
WRITE_EVERY_ROWS = int(os.environ.get("WRITE_EVERY_ROWS", "1000"))
COUNT_WORKERS = int(os.environ.get("COUNT_WORKERS", "48"))

OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "openai/gpt-oss-20b")
OPENAI_TEMPERATURE = float(os.environ.get("OPENAI_TEMPERATURE", "0.3"))
OPENAI_MAX_TOKENS = int(os.environ.get("OPENAI_MAX_TOKENS", "4096"))
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
SAMPLE_COUNT_PATH = os.path.join(OUTPUT_DIR, "sample_count.json")

SYSTEM_PROMPT = (
    "You are a professional translator who translates English into Japanese. "
    "Please translate the received text directly and accurately into Japanese. "
    "Please keep the tone of the original text. Output only the translated text. "
    "Do not use Markdown formatting. Be sure to output the translated Japanese text."
)
PROMPT_EXAMPLES = [
    (
        "The quiet cafe buzzed with ideas as sunlight spilled across half-finished notebooks.",
        "静かなカフェにはアイデアが満ち、差し込む日差しが書きかけのノートを照らしていた。",
    ),
    (
        "She took a deep breath and stepped forward, trusting that curiosity would guide her way.",
        "彼女は深く息を吸い、好奇心が自分を導いてくれると信じて一歩踏み出した。",
    ),
]


def _build_prompt(user_input: str) -> str:
    parts: list[str] = [SYSTEM_PROMPT, ""]
    for text_en, text_ja in PROMPT_EXAMPLES:
        parts.append(f"English: {text_en}")
        parts.append(f"Japanese: {text_ja}")
        parts.append("")
    parts.append(f"English: {user_input}")
    parts.append("Japanese:")
    return "\n".join(parts)


class LLMBatchWorkerPool:
    def __init__(
        self,
        base_urls: list[str],
        api_key: str,
        model: str,
        max_retries: int = 2,
        worker_count: int | None = None,
    ) -> None:
        if not base_urls:
            raise ValueError("base_urls must not be empty")
        self._tasks: queue.Queue[
            tuple[int, list[tuple[int, str]], int] | None
        ] = queue.Queue()
        self._results: queue.Queue[
            tuple[int, list[tuple[int, str]], list[str], Exception | None]
        ] = queue.Queue()
        self._threads: list[threading.Thread] = []
        self._max_retries = max_retries
        self._model = model
        total_workers = worker_count or len(base_urls)
        for i in range(total_workers):
            base_url = base_urls[i % len(base_urls)]
            thread = threading.Thread(
                target=self._worker,
                args=(base_url, api_key),
                daemon=True,
            )
            thread.start()
            self._threads.append(thread)

    @property
    def results(
        self,
    ) -> queue.Queue[tuple[int, list[tuple[int, str]], list[str], Exception | None]]:
        return self._results

    def submit(self, batch_id: int, batch: list[tuple[int, str]]) -> None:
        self._tasks.put((batch_id, batch, 0))

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
            batch_id, batch, attempt = task
            try:
                prompts = [_build_prompt(text) for _, text in batch]
                response = client.completions.create(
                    model=self._model,
                    prompt=prompts,
                    temperature=OPENAI_TEMPERATURE,
                    max_tokens=OPENAI_MAX_TOKENS,
                    stop=["\nEnglish:"],
                )
                translations = ["" for _ in prompts]
                for choice in response.choices:
                    index = getattr(choice, "index", None)
                    if index is None or index < 0 or index >= len(translations):
                        continue
                    text = choice.text or ""
                    translations[index] = text.lstrip("\n ")
                self._results.put((batch_id, batch, translations, None))
            except Exception as exc:
                if attempt < self._max_retries:
                    time.sleep(0.5 * (attempt + 1))
                    self._tasks.put((batch_id, batch, attempt + 1))
                else:
                    self._results.put((batch_id, batch, ["" for _ in batch], exc))
            finally:
                self._tasks.task_done()


def _normalize_text(value: str | None) -> str:
    return value if value is not None else ""


class ParquetShardWriter:
    def __init__(
        self,
        output_dir: str,
        flush_every_rows: int,
        start_shard_id: int,
        start_written: int,
    ) -> None:
        self.output_dir = output_dir
        self.flush_every_rows = max(1, flush_every_rows)
        self.shard_id = start_shard_id
        self._buffer_en: list[str] = []
        self._buffer_ja: list[str] = []
        self._schema = pa.schema(
            [
                ("text_en", pa.string()),
                ("text_ja", pa.string()),
            ]
        )
        self.written_rows = start_written
        os.makedirs(self.output_dir, exist_ok=True)

    @property
    def buffered_rows(self) -> int:
        return len(self._buffer_en)

    def _flush_buffer(self) -> int:
        if not self._buffer_en:
            return 0
        output_path = os.path.join(
            self.output_dir,
            f"c4-train-ja-{self.shard_id:05d}.parquet",
        )
        table = pa.table(
            {"text_en": self._buffer_en, "text_ja": self._buffer_ja},
            schema=self._schema,
        )
        pq.write_table(table, output_path, compression="zstd")
        flushed_rows = len(self._buffer_en)
        self.written_rows += flushed_rows
        self.shard_id += 1
        self._buffer_en = []
        self._buffer_ja = []
        return flushed_rows

    def add(self, text_en: str | None, text_ja: str | None) -> int:
        text_en = _normalize_text(text_en)
        text_ja = _normalize_text(text_ja)
        self._buffer_en.append(text_en)
        self._buffer_ja.append(text_ja)
        if len(self._buffer_en) >= self.flush_every_rows:
            return self._flush_buffer()
        return 0

    def flush(self) -> int:
        return self._flush_buffer()


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


def _count_jsonl_rows_slow(paths: list[str]) -> int | None:
    total = 0
    for path in paths:
        try:
            with open(path, "rb") as handle:
                for _ in handle:
                    total += 1
        except Exception as exc:
            print(f"failed to count rows in {path}: {exc}")
            return None
    return total


def _count_jsonl_rows_fast(paths: list[str]) -> int | None:
    try:
        result = subprocess.run(
            ["xargs", "-P", str(COUNT_WORKERS), "-n", "1", "wc", "-l"],
            input="\n".join(paths),
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception as exc:
        print(f"parallel wc -l failed, falling back to slow count: {exc}")
        return _count_jsonl_rows_slow(paths)
    total = 0
    for line in result.stdout.splitlines():
        parts = line.strip().split()
        if not parts:
            continue
        try:
            total += int(parts[0])
        except ValueError:
            continue
    return total


def _load_cached_sample_count(path: str, data_files_pattern: str) -> int | None:
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception:
        return None
    if payload.get("pattern") != data_files_pattern:
        return None
    try:
        return int(payload["count"])
    except Exception:
        return None


def _save_cached_sample_count(path: str, data_files_pattern: str, count: int) -> None:
    payload = {"pattern": data_files_pattern, "count": int(count)}
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=True)
    os.replace(tmp_path, path)


def _count_jsonl_rows(data_files_pattern: str) -> int | None:
    paths = sorted(glob.glob(data_files_pattern))
    if not paths:
        return None
    return _count_jsonl_rows_fast(paths)


def _resolve_max_samples(
    dataset: object,
    max_samples: int,
    data_files_pattern: str,
    sample_count_path: str,
) -> int | None:
    if max_samples >= 0:
        return max_samples
    cached = _load_cached_sample_count(sample_count_path, data_files_pattern)
    if cached is not None:
        return cached
    try:
        split_info = dataset["train"].info.splits.get("train")
        if split_info is not None and split_info.num_examples is not None:
            return int(split_info.num_examples)
    except Exception:
        pass
    return None


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Translate C4 data to parquet.")
    parser.add_argument(
        "--model",
        default=None,
        help="Override model name for translation.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    if args.model:
        model = args.model
    else:
        model = OPENAI_MODEL
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
    dataset = load_dataset(
        "json",
        data_files=DATA_FILES,
        streaming=True,
    )
    max_samples = _resolve_max_samples(
        dataset,
        MAX_SAMPLES,
        DATA_FILES,
        SAMPLE_COUNT_PATH,
    )
    if max_samples is not None and resume_from >= max_samples:
        print(f"done: {resume_from} samples")
        return

    writer = ParquetShardWriter(
        OUTPUT_DIR,
        WRITE_EVERY_ROWS,
        start_shard_id,
        existing_rows,
    )
    processed = resume_from
    batch_size = max(1, BATCH_SIZE)
    max_in_flight = max(1, len(BASE_URLS))
    pool = LLMBatchWorkerPool(
        BASE_URLS,
        API_KEY,
        model=model,
        worker_count=max_in_flight,
    )
    pending: dict[int, tuple[str, str]] = {}
    in_flight = 0
    submitted = resume_from
    expected_index = resume_from
    dataset_iter = iter(dataset["train"])
    progress_bar = None
    last_written = existing_rows
    skipped_errors = 0
    batch: list[tuple[int, str]] = []
    batch_id = 0
    if tqdm is not None:
        progress_bar = tqdm(
            total=max_samples,
            initial=resume_from,
            desc="Translating",
            unit="samples",
        )
    count_queue: queue.Queue[int] | None = None
    if max_samples is None and MAX_SAMPLES < 0:
        count_queue = queue.Queue(maxsize=1)

        def _count_worker() -> None:
            count = _count_jsonl_rows(DATA_FILES)
            if count is None:
                return
            _save_cached_sample_count(SAMPLE_COUNT_PATH, DATA_FILES, count)
            try:
                count_queue.put(count, block=False)
            except Exception:
                return

        threading.Thread(target=_count_worker, daemon=True).start()

    try:
        skipped = 0
        while skipped < resume_from:
            try:
                next(dataset_iter)
            except StopIteration:
                print(f"done: {processed} samples")
                return
            skipped += 1

        dataset_exhausted = False
        while True:
            if max_samples is None and count_queue is not None:
                try:
                    max_samples = count_queue.get_nowait()
                except queue.Empty:
                    max_samples = None
                if max_samples is not None and progress_bar is not None:
                    progress_bar.total = max_samples
                    progress_bar.refresh()

            while (
                in_flight < max_in_flight
                and (max_samples is None or submitted < max_samples)
                and not dataset_exhausted
            ):
                while (
                    not dataset_exhausted
                    and (max_samples is None or submitted < max_samples)
                    and len(batch) < batch_size
                ):
                    try:
                        sample = next(dataset_iter)
                    except StopIteration:
                        dataset_exhausted = True
                        if max_samples is not None:
                            submitted = max_samples
                        break
                    text = sample.get("text", "")
                    if not text:
                        pending[submitted] = ("", "")
                    else:
                        batch.append((submitted, text))
                    submitted += 1

                if batch:
                    pool.submit(batch_id, batch)
                    in_flight += 1
                    batch_id += 1
                    batch = []
                if dataset_exhausted:
                    break

            if (
                batch
                and in_flight < max_in_flight
                and (dataset_exhausted or (max_samples is not None and submitted >= max_samples))
            ):
                pool.submit(batch_id, batch)
                in_flight += 1
                batch_id += 1
                batch = []

            wrote_any = False
            while expected_index in pending:
                text_en, text_ja = pending.pop(expected_index)
                flushed_rows = writer.add(text_en, text_ja)
                processed = expected_index + 1
                if flushed_rows > 0:
                    _write_progress(
                        PROGRESS_PATH,
                        writer.written_rows,
                        writer.written_rows,
                        writer.shard_id,
                    )
                    last_written = writer.written_rows
                if progress_bar is not None:
                    progress_bar.update(1)
                    progress_bar.set_postfix(
                        in_flight=in_flight,
                        pending=len(pending),
                        buffered=writer.buffered_rows,
                    )
                expected_index += 1
                wrote_any = True
                if progress_bar is None:
                    if max_samples is None:
                        print(f"processed {processed}")
                    else:
                        print(f"processed {processed}/{max_samples}")

            if max_samples is not None and expected_index >= max_samples:
                break
            if in_flight == 0 and (dataset_exhausted or (max_samples is not None and submitted >= max_samples)):
                if not pending and not batch:
                    break
            if wrote_any:
                continue
            if in_flight == 0:
                continue

            _batch_id, result_batch, translations, error = pool.results.get()
            in_flight -= 1
            if error is not None:
                skipped_errors += len(result_batch)
                print(f"skip batch start_index={result_batch[0][0]} error={error}")
                translations = ["" for _ in result_batch]
            for (index, text_en), text_ja in zip(result_batch, translations):
                pending[index] = (text_en, text_ja)
    finally:
        pool.close()

    final_flushed = writer.flush()
    if final_flushed > 0 or last_written != writer.written_rows:
        _write_progress(
            PROGRESS_PATH,
            writer.written_rows,
            writer.written_rows,
            writer.shard_id,
        )
    if progress_bar is not None:
        progress_bar.close()
    print(f"done: {writer.written_rows} samples")


if __name__ == "__main__":
    main()
