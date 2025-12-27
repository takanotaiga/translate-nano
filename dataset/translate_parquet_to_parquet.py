import argparse
import glob
import json
import os
import queue
import threading
import time
from typing import Iterable, Sequence

from openai import OpenAI
import pyarrow as pa
import pyarrow.parquet as pq
try:
    from tqdm import tqdm
except Exception:  # pragma: no cover - optional dependency
    tqdm = None


DEFAULT_DATA_FILES = os.path.join(".", "c4-en-filtered-parquet", "*.parquet")
DEFAULT_OUTPUT_DIR = os.path.join(".", "c4-ja-parquet-filtered")
DEFAULT_INPUT_COLUMN = "en_text"
DEFAULT_OUTPUT_EN_COLUMN = "text_en"
DEFAULT_OUTPUT_JA_COLUMN = "text_ja"
DEFAULT_OUTPUT_PREFIX = "c4-train-ja-filtered"
DEFAULT_MAX_SAMPLES = -1
DEFAULT_READ_BATCH_SIZE = 1024

TARGET_IN_FLIGHT = 80
WRITE_EVERY_ROWS = 1000

GPT_OSS_MODEL = os.environ.get(
    "GPT_OSS_MODEL",
    os.environ.get("OPENAI_MODEL", "openai/gpt-oss-20b"),
)
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
    def __init__(
        self,
        base_urls: list[str],
        api_key: str,
        model: str,
        max_retries: int = 2,
        worker_count: int | None = None,
        results_queue: queue.Queue[tuple[int, str, str, Exception | None]] | None = None,
    ) -> None:
        self._tasks: queue.Queue[tuple[int, str, int] | None] = queue.Queue()
        self._results = results_queue or queue.Queue()
        self._threads: list[threading.Thread] = []
        self._max_retries = max_retries
        self._model = model
        if not base_urls:
            raise ValueError("base_urls must not be empty")
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
        if base_url:
            client = OpenAI(api_key=api_key, base_url=base_url)
        else:
            client = OpenAI(api_key=api_key)
        while True:
            task = self._tasks.get()
            if task is None:
                self._tasks.task_done()
                break
            index, text, attempt = task
            try:
                response = client.chat.completions.create(
                    model=self._model,
                    messages=_build_messages(text),
                    temperature=OPENAI_TEMPERATURE,
                    max_tokens=OPENAI_MAX_TOKENS,
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


class _PoolState:
    def __init__(self, name: str, pool: LLMWorkerPool, max_in_flight: int) -> None:
        self.name = name
        self.pool = pool
        self.max_in_flight = max(1, max_in_flight)
        self.in_flight = 0


def _pick_pool(
    pools: list[_PoolState],
    start_index: int,
) -> tuple[_PoolState | None, int]:
    if not pools:
        return None, start_index
    for offset in range(len(pools)):
        pool = pools[(start_index + offset) % len(pools)]
        if pool.in_flight < pool.max_in_flight:
            return pool, start_index + offset + 1
    return None, start_index


def _normalize_text(value: str | None) -> str:
    return value if value is not None else ""


class ParquetShardWriter:
    def __init__(
        self,
        output_dir: str,
        flush_every_rows: int,
        start_shard_id: int,
        start_written: int,
        file_prefix: str,
        output_en_column: str,
        output_ja_column: str,
    ) -> None:
        self.output_dir = output_dir
        self.flush_every_rows = max(1, flush_every_rows)
        self.shard_id = start_shard_id
        self.file_prefix = file_prefix
        self.output_en_column = output_en_column
        self.output_ja_column = output_ja_column
        self._buffer_en: list[str] = []
        self._buffer_ja: list[str] = []
        self._schema = pa.schema(
            [
                (self.output_en_column, pa.string()),
                (self.output_ja_column, pa.string()),
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
            f"{self.file_prefix}-{self.shard_id:05d}.parquet",
        )
        table = pa.table(
            {self.output_en_column: self._buffer_en, self.output_ja_column: self._buffer_ja},
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


def _find_next_shard_id(output_dir: str, file_prefix: str) -> int:
    if not os.path.isdir(output_dir):
        return 0
    max_id = -1
    prefix = f"{file_prefix}-"
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


def _count_existing_rows(output_dir: str, file_prefix: str) -> int:
    if not os.path.isdir(output_dir):
        return 0
    total_rows = 0
    prefix = f"{file_prefix}-"
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


def _expand_data_files(data_files: str) -> list[str]:
    paths: list[str] = []
    for token in data_files.split(","):
        token = token.strip()
        if not token:
            continue
        if os.path.isdir(token):
            token = os.path.join(token, "*.parquet")
        if any(char in token for char in "*?["):
            paths.extend(glob.glob(token))
        elif os.path.exists(token):
            paths.append(token)
    return sorted(set(paths))


def _count_parquet_rows(paths: list[str], text_column: str) -> int | None:
    total = 0
    for path in paths:
        try:
            parquet_file = pq.ParquetFile(path)
        except Exception as exc:
            print(f"failed to read {path}: {exc}")
            return None
        if text_column not in parquet_file.schema.names:
            print(f"missing column {text_column} in {path}")
            return None
        total += parquet_file.metadata.num_rows
    return total


def _resolve_max_samples(
    max_samples: int,
    data_files_pattern: str,
    sample_count_path: str,
    paths: list[str],
    text_column: str,
) -> int | None:
    if max_samples >= 0:
        return max_samples
    cached = _load_cached_sample_count(sample_count_path, data_files_pattern)
    if cached is not None:
        return cached
    count = _count_parquet_rows(paths, text_column)
    if count is None:
        return None
    _save_cached_sample_count(sample_count_path, data_files_pattern, count)
    return count


def _iter_parquet_texts(
    paths: Iterable[str],
    text_column: str,
    start_row: int,
    batch_size: int,
) -> Iterable[str | None]:
    remaining = max(0, start_row)
    for path in paths:
        parquet_file = pq.ParquetFile(path)
        num_rows = parquet_file.metadata.num_rows
        if remaining >= num_rows:
            remaining -= num_rows
            continue
        for batch in parquet_file.iter_batches(
            batch_size=batch_size,
            columns=[text_column],
        ):
            values = batch.column(0).to_pylist()
            if remaining:
                if remaining >= len(values):
                    remaining -= len(values)
                    continue
                values = values[remaining:]
                remaining = 0
            for value in values:
                yield value


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Translate parquet data to parquet.")
    parser.add_argument(
        "--data-files",
        default=DEFAULT_DATA_FILES,
        help="Glob pattern, file, or directory for input parquet files.",
    )
    parser.add_argument(
        "--input-column",
        default=DEFAULT_INPUT_COLUMN,
        help="Column name that contains the English text.",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to write parquet shards.",
    )
    parser.add_argument(
        "--output-en-column",
        default=DEFAULT_OUTPUT_EN_COLUMN,
        help="Output column name for English text.",
    )
    parser.add_argument(
        "--output-ja-column",
        default=DEFAULT_OUTPUT_JA_COLUMN,
        help="Output column name for Japanese text.",
    )
    parser.add_argument(
        "--file-prefix",
        default=DEFAULT_OUTPUT_PREFIX,
        help="Filename prefix for output parquet shards.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=DEFAULT_MAX_SAMPLES,
        help="Optional cap on samples to translate. -1 means no limit.",
    )
    parser.add_argument(
        "--read-batch-size",
        type=int,
        default=DEFAULT_READ_BATCH_SIZE,
        help="Row batch size for parquet reader.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Override model name for the selected backend.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    gpt_model = GPT_OSS_MODEL
    if args.model:
        gpt_model = args.model

    data_files_pattern = args.data_files
    paths = _expand_data_files(data_files_pattern)
    if not paths:
        print(f"no parquet files matched: {data_files_pattern}")
        return

    os.makedirs(args.output_dir, exist_ok=True)
    progress_path = os.path.join(args.output_dir, "progress.parquet")
    sample_count_path = os.path.join(args.output_dir, "sample_count.json")

    progress = _load_progress(progress_path)
    existing_rows = _count_existing_rows(args.output_dir, args.file_prefix)
    if progress["written"] != existing_rows:
        print(
            "progress mismatch: "
            f"progress_written={progress['written']} existing_rows={existing_rows}. "
            "using existing_rows for resume."
        )
    resume_from = existing_rows
    next_shard_id = _find_next_shard_id(args.output_dir, args.file_prefix)
    start_shard_id = max(progress["shard_id"], next_shard_id)

    max_samples = _resolve_max_samples(
        args.max_samples,
        data_files_pattern,
        sample_count_path,
        paths,
        args.input_column,
    )
    if max_samples is not None and resume_from >= max_samples:
        print(f"done: {resume_from} samples")
        return

    writer = ParquetShardWriter(
        args.output_dir,
        WRITE_EVERY_ROWS,
        start_shard_id,
        existing_rows,
        args.file_prefix,
        args.output_en_column,
        args.output_ja_column,
    )
    processed = resume_from
    max_in_flight = max(1, TARGET_IN_FLIGHT)
    results_queue: queue.Queue[tuple[int, str, str, Exception | None]] = queue.Queue()
    pools: list[_PoolState] = []
    pool = LLMWorkerPool(
        BASE_URLS,
        API_KEY,
        model=gpt_model,
        worker_count=max_in_flight,
        results_queue=results_queue,
    )
    pools.append(_PoolState("gpt-oss", pool, max_in_flight))

    pending: dict[int, tuple[str, str]] = {}
    in_flight_by_index: dict[int, _PoolState] = {}
    submitted = resume_from
    expected_index = resume_from
    dataset_iter = iter(
        _iter_parquet_texts(
            paths,
            args.input_column,
            start_row=resume_from,
            batch_size=max(1, args.read_batch_size),
        )
    )
    progress_bar = None
    last_written = existing_rows
    skipped_errors = 0
    next_pool_index = 0
    if tqdm is not None:
        progress_bar = tqdm(
            total=max_samples,
            initial=resume_from,
            desc="Translating",
            unit="samples",
        )

    try:
        dataset_exhausted = False
        while True:
            while max_samples is None or submitted < max_samples:
                pool_state, next_pool_index = _pick_pool(pools, next_pool_index)
                if pool_state is None:
                    break
                try:
                    text = next(dataset_iter)
                except StopIteration:
                    dataset_exhausted = True
                    if max_samples is not None:
                        submitted = max_samples
                    break
                if not isinstance(text, str) or not text:
                    pending[submitted] = ("", "")
                else:
                    pool_state.pool.submit(submitted, text)
                    pool_state.in_flight += 1
                    in_flight_by_index[submitted] = pool_state
                submitted += 1

            wrote_any = False
            while expected_index in pending:
                text_en, text_ja = pending.pop(expected_index)
                flushed_rows = writer.add(text_en, text_ja)
                processed = expected_index + 1
                if flushed_rows > 0:
                    _write_progress(
                        progress_path,
                        writer.written_rows,
                        writer.written_rows,
                        writer.shard_id,
                    )
                    last_written = writer.written_rows
                if progress_bar is not None:
                    total_in_flight = sum(pool.in_flight for pool in pools)
                    progress_bar.update(1)
                    progress_bar.set_postfix(
                        in_flight=total_in_flight,
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
            if wrote_any:
                continue
            total_in_flight = sum(pool.in_flight for pool in pools)
            if total_in_flight == 0 and (
                dataset_exhausted or (max_samples is not None and submitted >= max_samples)
            ):
                break

            index, text_en, text_ja, error = results_queue.get()
            pool_state = in_flight_by_index.pop(index, None)
            if pool_state is not None:
                pool_state.in_flight = max(0, pool_state.in_flight - 1)
            if error is not None:
                skipped_errors += 1
                print(f"skip index={index} error={error}")
                pending[index] = ("", "")
            else:
                pending[index] = (text_en, text_ja)
    finally:
        for pool_state in pools:
            pool_state.pool.close()

    final_flushed = writer.flush()
    if final_flushed > 0 or last_written != writer.written_rows:
        _write_progress(
            progress_path,
            writer.written_rows,
            writer.written_rows,
            writer.shard_id,
        )
    if progress_bar is not None:
        progress_bar.close()
    print(f"done: {writer.written_rows} samples")


if __name__ == "__main__":
    main()
