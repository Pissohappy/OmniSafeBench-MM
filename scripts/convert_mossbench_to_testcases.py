#!/usr/bin/env python3
"""
Convert MOSSBench dataset to OmniSafeBench-MM TestCase format.

MOSSBench has two subsets:
  - oversensitivity: images/ + images/metadata.csv (~300 records)
  - contrast: contrast_images/ + contrast_images/metadata.csv (~300 records)

Images are already on disk. The metadata.csv 'file_name' field contains the
relative path (e.g. 'images/1.png') relative to the dataset root.
"""

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Dict, List


def load_metadata_csv(csv_path: str, subset: str, base_dir: str) -> List[Dict]:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Metadata CSV not found: {csv_path}")

    records = []
    missing = []

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            file_name = row.get('file_name', '').strip()
            if not file_name:
                continue

            abs_path = os.path.join(base_dir, file_name)
            abs_path = os.path.abspath(abs_path)
            if not os.path.exists(abs_path):
                missing.append(file_name)
                continue

            records.append({
                'pid': row.get('pid', '').strip(),
                'question': row.get('question', '').strip(),
                'file_name': file_name,
                'abs_image_path': abs_path,
                'short_description': row.get('short description', '').strip(),
                'description': row.get('description', '').strip(),
                'harm_label': row.get('metadata_harm', '').strip(),
                'subset': subset,
            })

    if missing:
        print(f"  Warning: {len(missing)} images not found for {subset}")
        for p in missing[:3]:
            print(f"    {p}")

    print(f"  Loaded {len(records)} records from {subset}")
    return records


def convert_to_testcases(records: List[Dict], attack_name: str = "mossbench") -> List[Dict]:
    testcases = []
    for i, rec in enumerate(records):
        testcase = {
            "test_case_id": f"{rec['subset']}_{rec['pid']}",
            "prompt": rec['question'],
            "image_path": rec['abs_image_path'],
            "metadata": {
                "attack_method": attack_name,
                "original_prompt": rec['question'],
                "subset": rec['subset'],
                "pid": rec['pid'],
                "harm_label": rec['harm_label'],
                "short_description": rec['short_description'],
            }
        }
        testcases.append(testcase)
    return testcases


def save_testcases_jsonl(testcases: List[Dict], output_file: str):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        for tc in testcases:
            f.write(json.dumps(tc, ensure_ascii=False) + '\n')
    print(f"Saved {len(testcases)} test cases to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert MOSSBench dataset to OmniSafeBench-MM TestCase format"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="/mnt/disk1/data/vlm_attack/MOSSBench",
        help="Path to MOSSBench root directory"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output/test_cases/mossbench",
        help="Output directory for converted test cases"
    )
    parser.add_argument(
        "--attack-name",
        type=str,
        default="mossbench",
        help="Attack method name for metadata"
    )
    parser.add_argument(
        "--subsets",
        type=str,
        default="oversensitivity,contrast",
        help="Comma-separated subsets to include (oversensitivity, contrast)"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("MOSSBench to OmniSafeBench-MM Conversion")
    print("=" * 60)
    print(f"Data directory: {args.data_dir}")
    print(f"Output dir:     {args.output_dir}")
    print(f"Attack name:    {args.attack_name}")
    print(f"Subsets:        {args.subsets}")
    print()

    data_dir = args.data_dir
    all_records = []

    subset_map = {
        "oversensitivity": "images/metadata.csv",
        "contrast": "contrast_images/metadata.csv",
    }

    for subset in [s.strip() for s in args.subsets.split(',')]:
        if subset not in subset_map:
            print(f"Unknown subset '{subset}', skipping")
            continue
        csv_path = os.path.join(data_dir, subset_map[subset])
        print(f"Loading subset: {subset}")
        records = load_metadata_csv(csv_path, subset=subset, base_dir=data_dir)
        all_records.extend(records)

    print()
    print(f"Total records: {len(all_records)}")
    testcases = convert_to_testcases(all_records, attack_name=args.attack_name)

    output_file = os.path.join(args.output_dir, "test_cases.jsonl")
    save_testcases_jsonl(testcases, output_file)

    print()
    print("=" * 60)
    print("Conversion Summary")
    print("=" * 60)
    print(f"Total test cases: {len(testcases)}")

    subsets = {}
    for tc in testcases:
        s = tc['metadata']['subset']
        subsets[s] = subsets.get(s, 0) + 1
    print()
    print("Breakdown by subset:")
    for s, count in sorted(subsets.items()):
        print(f"  {s}: {count}")

    print()
    print("Conversion completed successfully!")


if __name__ == "__main__":
    main()
