import argparse
import multiprocessing as mp
import os
from typing import Sequence

from datasets import load_dataset
import pyarrow as pa
import pyarrow.parquet as pq
import tiktoken


DEFAULT_DATA_FILES = "/home/taiga/ml_lake/translate-nano-dataset/en/c4-train.*.json"
DEFAULT_OUTPUT_DIR = os.path.join(".", "c4-en-filtered-parquet")
DEFAULT_TEXT_FIELD = "text"
DEFAULT_SPLIT = "train"
DEFAULT_MAX_TOKENS = 512
DEFAULT_BATCH_SIZE = 2048
DEFAULT_WORKERS = 96
DEFAULT_MAX_FILE_SIZE_MB = 300

_ENCODING = None


def _init_worker() -> None:
    global _ENCODING
    _ENCODING = tiktoken.get_encoding("o200k_harmony")


def _count_tokens(text: str) -> int:
    if not text:
        return 0
    try:
        return len(_ENCODING.encode(text))
    except Exception:
        return -1


class ParquetSizeWriter:
    def __init__(
        self,
        output_dir: str,
        max_file_size_bytes: int,
        flush_every_rows: int,
        file_prefix: str,
    ) -> None:
        self.output_dir = output_dir
        self.max_file_size_bytes = max(1, max_file_size_bytes)
        self.flush_every_rows = max(1, flush_every_rows)
        self.file_prefix = file_prefix
        self._schema = pa.schema([("en_text", pa.string())])
        self._buffer: list[str] = []
        self._writer: pq.ParquetWriter | None = None
        self._current_path = ""
        self._shard_id = 0
        os.makedirs(self.output_dir, exist_ok=True)
        self._open_writer()

    def _open_writer(self) -> None:
        if self._writer is not None:
            self._writer.close()
        filename = f"{self.file_prefix}-{self._shard_id:05d}.parquet"
        self._current_path = os.path.join(self.output_dir, filename)
        self._writer = pq.ParquetWriter(
            self._current_path,
            self._schema,
            compression="zstd",
        )
        self._shard_id += 1

    def _rotate_if_needed(self) -> None:
        try:
            size = os.path.getsize(self._current_path)
        except OSError:
            return
        if size >= self.max_file_size_bytes:
            self._open_writer()

    def _flush_buffer(self) -> int:
        if not self._buffer:
            return 0
        if self._writer is None:
            self._open_writer()
        table = pa.table({"en_text": self._buffer}, schema=self._schema)
        self._writer.write_table(table)
        flushed = len(self._buffer)
        self._buffer = []
        self._rotate_if_needed()
        return flushed

    def add(self, text: str) -> int:
        self._buffer.append(text)
        if len(self._buffer) >= self.flush_every_rows:
            return self._flush_buffer()
        return 0

    def flush(self) -> int:
        return self._flush_buffer()

    def close(self) -> None:
        self._flush_buffer()
        if self._writer is not None:
            self._writer.close()
            self._writer = None


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Filter C4 data by token count and write to parquet.",
    )
    parser.add_argument(
        "--data-files",
        default=DEFAULT_DATA_FILES,
        help="Glob pattern for input JSON files.",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to write parquet shards.",
    )
    parser.add_argument(
        "--split",
        default=DEFAULT_SPLIT,
        help="Dataset split to read.",
    )
    parser.add_argument(
        "--text-field",
        default=DEFAULT_TEXT_FIELD,
        help="Field name that contains the text.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help="Keep samples with token count below this threshold.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Batch size for tokenization workers.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help="Number of parallel tokenization workers.",
    )
    parser.add_argument(
        "--max-file-size-mb",
        type=int,
        default=DEFAULT_MAX_FILE_SIZE_MB,
        help="Rotate output parquet files at this size in MB.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=-1,
        help="Optional cap on samples to scan. -1 means no limit.",
    )
    parser.add_argument(
        "--file-prefix",
        default="c4-train-en-filtered",
        help="Filename prefix for parquet shards.",
    )
    return parser.parse_args(argv)


def _process_batch(
    pool: mp.pool.Pool,
    texts: list[str],
    max_tokens: int,
    writer: ParquetSizeWriter,
    workers: int,
) -> tuple[int, int]:
    if not texts:
        return 0, 0
    chunk_size = max(1, len(texts) // max(1, workers * 4))
    counts = pool.map(_count_tokens, texts, chunksize=chunk_size)
    kept = 0
    skipped_special = 0
    for text, count in zip(texts, counts):
        if count < 0:
            skipped_special += 1
            continue
        if count < max_tokens:
            writer.add(text)
            kept += 1
    return kept, skipped_special


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    dataset = load_dataset(
        "json",
        data_files=args.data_files,
        streaming=True,
    )
    dataset_iter = iter(dataset[args.split])

    max_samples = args.max_samples
    if max_samples is not None and max_samples < 0:
        max_samples = None

    ctx = mp.get_context("fork")
    workers = max(1, args.workers)
    batch_size = max(1, args.batch_size)
    writer = ParquetSizeWriter(
        args.output_dir,
        max(1, args.max_file_size_mb) * 1024 * 1024,
        flush_every_rows=batch_size,
        file_prefix=args.file_prefix,
    )

    processed = 0
    kept = 0
    skipped = 0
    skipped_special = 0
    buffer: list[str] = []

    with ctx.Pool(processes=workers, initializer=_init_worker) as pool:
        while True:
            if max_samples is not None and processed >= max_samples:
                break
            try:
                sample = next(dataset_iter)
            except StopIteration:
                break
            processed += 1
            text = sample.get(args.text_field)
            if text is None or not isinstance(text, str):
                skipped += 1
                continue
            buffer.append(text)
            if len(buffer) >= batch_size:
                batch_kept, batch_skipped_special = _process_batch(
                    pool,
                    buffer,
                    args.max_tokens,
                    writer,
                    workers,
                )
                kept += batch_kept
                skipped_special += batch_skipped_special
                buffer = []

        if buffer:
            batch_kept, batch_skipped_special = _process_batch(
                pool,
                buffer,
                args.max_tokens,
                writer,
                workers,
            )
            kept += batch_kept
            skipped_special += batch_skipped_special

    writer.close()
    print(f"processed: {processed}")
    print(f"skipped: {skipped}")
    print(f"skipped_special: {skipped_special}")
    print(f"kept: {kept}")


if __name__ == "__main__":
    main()
