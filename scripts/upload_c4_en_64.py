from __future__ import annotations

import argparse
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import login


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Upload c4-en-filtered-parquet-64 to a Hugging Face dataset repo."
    )
    parser.add_argument(
        "--data-dir",
        default="c4-en-filtered-parquet-64",
        help="Directory containing parquet shards.",
    )
    parser.add_argument(
        "--repo-id",
        default="taigatakano/c4-en-64token",
        help="Destination dataset repo on Hugging Face.",
    )
    parser.add_argument(
        "--split-name",
        default="train",
        help="Split name to use on the Hub.",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="Hugging Face token (optional).",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create a private dataset repo.",
    )
    parser.add_argument(
        "--max-shard-size",
        default="1GB",
        help="Max shard size for Hub upload.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.token:
        login(token=args.token)

    data_dir = Path(args.data_dir)
    parquet_files = sorted(str(path) for path in data_dir.glob("*.parquet"))
    if not parquet_files:
        raise SystemExit(f"No parquet files found in {data_dir}")

    dataset = load_dataset("parquet", data_files={args.split_name: parquet_files})
    dataset.push_to_hub(
        args.repo_id,
        private=args.private,
        max_shard_size=args.max_shard_size,
    )


if __name__ == "__main__":
    main()
