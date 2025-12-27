import argparse
import math
from typing import Sequence

from datasets import load_dataset
import tiktoken


DEFAULT_DATA_FILES = "/home/taiga/ml_lake/translate-nano-dataset/en/c4-train.*.json"
DEFAULT_TEXT_FIELD = "text"
DEFAULT_MAX_SAMPLES = 1000
DEFAULT_BUCKET_SIZE = 200
DEFAULT_SPLIT = "train"
DEFAULT_FORMAT = "auto"


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Show token count distribution for the first N samples.",
    )
    parser.add_argument(
        "--data-files",
        default=DEFAULT_DATA_FILES,
        help="Glob pattern for input JSON files.",
    )
    parser.add_argument(
        "--split",
        default=DEFAULT_SPLIT,
        help="Dataset split to read.",
    )
    parser.add_argument(
        "--format",
        default=DEFAULT_FORMAT,
        choices=("auto", "json", "parquet"),
        help="Dataset format to load.",
    )
    parser.add_argument(
        "--text-field",
        default=DEFAULT_TEXT_FIELD,
        help="Field name that contains the text.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=DEFAULT_MAX_SAMPLES,
        help="Number of samples to read from the start.",
    )
    parser.add_argument(
        "--bucket-size",
        type=int,
        default=DEFAULT_BUCKET_SIZE,
        help="Bucket size for the histogram.",
    )
    return parser.parse_args(argv)


def _infer_format(data_files: str, requested: str) -> str:
    if requested != "auto":
        return requested
    lower = data_files.lower()
    if ".parquet" in lower or ".pq" in lower:
        return "parquet"
    return "json"


def _resolve_text_field(sample: dict[str, object], requested: str) -> str:
    if requested in sample:
        return requested
    if requested == DEFAULT_TEXT_FIELD and "en_text" in sample:
        return "en_text"
    return requested


def _safe_token_count(encoding: tiktoken.Encoding, text: str) -> int | None:
    try:
        return len(encoding.encode(text))
    except Exception:
        return None


def _append_count(
    sample: dict[str, object],
    text_field: str,
    encoding: tiktoken.Encoding,
    counts: list[int],
) -> tuple[int, int, int]:
    text = sample.get(text_field)
    if text is None or not isinstance(text, str):
        counts.append(0)
        return 1, 0, 0
    if not text:
        counts.append(0)
        return 0, 1, 0
    count = _safe_token_count(encoding, text)
    if count is None:
        counts.append(0)
        return 0, 0, 1
    counts.append(count)
    return 0, 0, 0


def _percentile(sorted_counts: list[int], pct: float) -> float:
    if not sorted_counts:
        return 0.0
    k = (len(sorted_counts) - 1) * (pct / 100.0)
    lower = math.floor(k)
    upper = math.ceil(k)
    if lower == upper:
        return float(sorted_counts[int(k)])
    weight = k - lower
    return sorted_counts[lower] + (sorted_counts[upper] - sorted_counts[lower]) * weight


def _build_histogram(counts: list[int], bucket_size: int) -> list[tuple[int, int, int]]:
    if bucket_size <= 0:
        raise ValueError("bucket_size must be positive")
    if not counts:
        return [(0, bucket_size - 1, 0)]
    max_count = max(counts)
    bucket_count = (max_count // bucket_size) + 1
    buckets = [0 for _ in range(bucket_count)]
    for count in counts:
        buckets[count // bucket_size] += 1
    result: list[tuple[int, int, int]] = []
    for idx, count in enumerate(buckets):
        start = idx * bucket_size
        end = start + bucket_size - 1
        result.append((start, end, count))
    return result


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    max_samples = max(0, args.max_samples)
    if max_samples == 0:
        print("samples: 0")
        print("missing_text: 0")
        print("empty_text: 0")
        print("skipped_special: 0")
        print("min: 0 max: 0 mean: 0.00")
        print("p50: 0.0 p90: 0.0 p95: 0.0 p99: 0.0")
        print("histogram:")
        for start, end, count in _build_histogram([], args.bucket_size):
            print(f"{start:>6}-{end:<6} {count:>6} (  0.0%)")
        return

    dataset_format = _infer_format(args.data_files, args.format)
    dataset = load_dataset(
        dataset_format,
        data_files=args.data_files,
        streaming=True,
    )
    dataset_iter = iter(dataset[args.split])
    encoding = tiktoken.get_encoding("o200k_harmony")
    counts: list[int] = []
    missing_text = 0
    empty_text = 0
    skipped_special = 0
    try:
        first_sample = next(dataset_iter)
    except StopIteration:
        first_sample = None
    if first_sample is not None:
        text_field = _resolve_text_field(first_sample, args.text_field)
        missing, empty, special = _append_count(
            first_sample,
            text_field,
            encoding,
            counts,
        )
        missing_text += missing
        empty_text += empty
        skipped_special += special
        read_samples = 1
    else:
        text_field = args.text_field
        read_samples = 0

    while read_samples < max_samples:
        try:
            sample = next(dataset_iter)
        except StopIteration:
            break
        missing, empty, special = _append_count(
            sample,
            text_field,
            encoding,
            counts,
        )
        missing_text += missing
        empty_text += empty
        skipped_special += special
        read_samples += 1

    total = len(counts)
    counts_sorted = sorted(counts)
    mean = (sum(counts) / total) if total else 0.0
    min_count = counts_sorted[0] if counts_sorted else 0
    max_count = counts_sorted[-1] if counts_sorted else 0
    p50 = _percentile(counts_sorted, 50)
    p90 = _percentile(counts_sorted, 90)
    p95 = _percentile(counts_sorted, 95)
    p99 = _percentile(counts_sorted, 99)

    print(f"samples: {total}")
    print(f"missing_text: {missing_text}")
    print(f"empty_text: {empty_text}")
    print(f"skipped_special: {skipped_special}")
    print(f"min: {min_count} max: {max_count} mean: {mean:.2f}")
    print(f"p50: {p50:.1f} p90: {p90:.1f} p95: {p95:.1f} p99: {p99:.1f}")
    print("histogram:")
    for start, end, count in _build_histogram(counts, args.bucket_size):
        pct = (count / total * 100.0) if total else 0.0
        print(f"{start:>6}-{end:<6} {count:>6} ({pct:>5.1f}%)")


if __name__ == "__main__":
    main()
