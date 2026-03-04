#!/usr/bin/env python3
"""
Convert SPA-VL dataset to OmniSafeBench-MM TestCase format.

SPA-VL stores images as bytes inside Parquet files (no separate image directory).
This script extracts images from the test split parquet files and saves them to disk,
then generates the test_cases.jsonl.

Test split parquet files:
  - test/harm-00000-of-00001.parquet  (265 harmful samples)
  - test/help-00000-of-00001.parquet  (265 helpful samples)

Columns: image (dict with 'bytes'), question, class1, class2, class3
"""

import argparse
import io
import json
import os
from pathlib import Path
from typing import Dict, List

import pandas as pd
from PIL import Image
from tqdm import tqdm


def extract_image_bytes(image_data) -> bytes:
    if isinstance(image_data, dict):
        return image_data['bytes']
    return bytes(image_data)


def load_and_extract_parquet(
    parquet_path: str,
    images_dir: str,
    subset: str,
    attack_name: str = "spa_vl",
    id_offset: int = 0,
) -> List[Dict]:
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(f"Parquet file not found: {parquet_path}")

    df = pd.read_parquet(parquet_path)
    print(f"  Loaded {len(df)} records from {os.path.basename(parquet_path)}")

    os.makedirs(images_dir, exist_ok=True)
    testcases = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Extracting {subset} images"):
        tc_id = id_offset + idx
        image_id = f"{subset}_{tc_id}"
        image_file = os.path.join(images_dir, f"{image_id}.png")

        # Extract and save image if not already done
        if not os.path.exists(image_file):
            try:
                img_bytes = extract_image_bytes(row['image'])
                image = Image.open(io.BytesIO(img_bytes))
                image.save(image_file)
            except Exception as e:
                print(f"  Warning: failed to extract image {image_id}: {e}")
                continue

        abs_image_path = os.path.abspath(image_file)

        question = str(row.get('question', '')).strip()
        if not question:
            continue

        testcase = {
            "test_case_id": str(tc_id),
            "prompt": question,
            "image_path": abs_image_path,
            "metadata": {
                "attack_method": attack_name,
                "original_prompt": question,
                "subset": subset,
                "class1": str(row.get('class1', '')),
                "class2": str(row.get('class2', '')),
                "class3": str(row.get('class3', '')),
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
        description="Convert SPA-VL dataset to OmniSafeBench-MM TestCase format"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="/mnt/disk1/data/vlm_attack/SPA-VL",
        help="Path to SPA-VL root directory"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output/test_cases/spa_vl",
        help="Output directory for converted test cases"
    )
    parser.add_argument(
        "--attack-name",
        type=str,
        default="spa_vl",
        help="Attack method name for metadata"
    )
    parser.add_argument(
        "--subsets",
        type=str,
        default="harm,help",
        help="Comma-separated subsets to include (harm, help)"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("SPA-VL to OmniSafeBench-MM Conversion")
    print("=" * 60)
    print(f"Data directory: {args.data_dir}")
    print(f"Output dir:     {args.output_dir}")
    print(f"Attack name:    {args.attack_name}")
    print(f"Subsets:        {args.subsets}")
    print()

    images_dir = os.path.join(args.output_dir, "images")
    all_testcases = []
    id_offset = 0

    subsets = [s.strip() for s in args.subsets.split(',')]

    for subset in subsets:
        parquet_path = os.path.join(
            args.data_dir, "test", f"{subset}-00000-of-00001.parquet"
        )
        print(f"Processing subset: {subset}")
        try:
            cases = load_and_extract_parquet(
                parquet_path=parquet_path,
                images_dir=images_dir,
                subset=subset,
                attack_name=args.attack_name,
                id_offset=id_offset,
            )
            print(f"  Generated {len(cases)} test cases")
            all_testcases.extend(cases)
            id_offset += len(cases)
        except FileNotFoundError as e:
            print(f"  Warning: {e}, skipping subset '{subset}'")

    output_file = os.path.join(args.output_dir, "test_cases.jsonl")
    save_testcases_jsonl(all_testcases, output_file)

    print()
    print("=" * 60)
    print("Conversion Summary")
    print("=" * 60)
    print(f"Total test cases: {len(all_testcases)}")
    print(f"Images saved to:  {images_dir}")

    subset_counts = {}
    for tc in all_testcases:
        s = tc['metadata']['subset']
        subset_counts[s] = subset_counts.get(s, 0) + 1
    print()
    print("Breakdown by subset:")
    for s, count in sorted(subset_counts.items()):
        print(f"  {s}: {count}")

    print()
    print("Conversion completed successfully!")


if __name__ == "__main__":
    main()
