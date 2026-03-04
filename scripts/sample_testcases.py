#!/usr/bin/env python3
"""
Sample test cases from each dataset.

Reads test_cases.jsonl from output/test_cases/{dataset}/ and writes
a random sample of up to 100 records to output_sample/test_cases/{dataset}/.

Image paths in the sampled JSONL remain pointing to the original absolute paths.

Usage:
  python3 scripts/sample_testcases.py
  python3 scripts/sample_testcases.py --n 50 --seed 123
  python3 scripts/sample_testcases.py --datasets advbenchm holisafe
"""

import argparse
import json
import os
import random
from pathlib import Path
from typing import List, Dict, Optional

DATASETS = [
    "advbenchm",
    "holisafe",
    "jailbreakv28k",
    "mmsafetybench",
    "mossbench",
    "mssbench",
    "spa_vl",
]


def load_jsonl(path: str) -> List[Dict]:
    records = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def save_jsonl(records: List[Dict], output_file: str):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')


def sample_dataset(
    input_file: str,
    output_file: str,
    n: int,
    seed: int,
) -> Optional[int]:
    if not os.path.exists(input_file):
        print(f"  Skipping: {input_file} not found")
        return None

    records = load_jsonl(input_file)
    total = len(records)

    if total == 0:
        print(f"  Skipping: {input_file} is empty")
        return None

    rng = random.Random(seed)
    if total <= n:
        sampled = records
        print(f"  Kept all {total} records (< {n})")
    else:
        sampled = rng.sample(records, n)
        print(f"  Sampled {n} from {total} records")

    save_jsonl(sampled, output_file)
    return len(sampled)


def main():
    parser = argparse.ArgumentParser(
        description="Sample test cases from each dataset for quick evaluation"
    )
    parser.add_argument(
        "--input-base",
        type=str,
        default="output/test_cases",
        help="Base directory containing per-dataset test_cases.jsonl files"
    )
    parser.add_argument(
        "--output-base",
        type=str,
        default="output_sample/test_cases",
        help="Base directory to write sampled test_cases.jsonl files"
    )
    parser.add_argument(
        "--n",
        type=int,
        default=100,
        help="Number of samples per dataset (default: 100)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs='+',
        default=None,
        help=f"Datasets to sample (default: all). Available: {', '.join(DATASETS)}"
    )

    args = parser.parse_args()

    datasets = args.datasets if args.datasets else DATASETS

    print("=" * 60)
    print("Test Case Sampling")
    print("=" * 60)
    print(f"Input base:   {args.input_base}")
    print(f"Output base:  {args.output_base}")
    print(f"Sample size:  {args.n}")
    print(f"Random seed:  {args.seed}")
    print(f"Datasets:     {', '.join(datasets)}")
    print()

    results = {}

    for dataset in datasets:
        input_file = os.path.join(args.input_base, dataset, "test_cases.jsonl")
        output_file = os.path.join(args.output_base, dataset, "test_cases.jsonl")

        print(f"Dataset: {dataset}")
        count = sample_dataset(input_file, output_file, n=args.n, seed=args.seed)
        if count is not None:
            print(f"  Saved to: {output_file}")
            results[dataset] = count
        print()

    print("=" * 60)
    print("Sampling Summary")
    print("=" * 60)
    total_sampled = 0
    for ds, count in results.items():
        print(f"  {ds}: {count} test cases")
        total_sampled += count
    print(f"\nTotal sampled: {total_sampled} test cases")
    print("Sampling completed successfully!")


if __name__ == "__main__":
    main()
