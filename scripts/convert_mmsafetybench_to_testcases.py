#!/usr/bin/env python3
"""
Convert MM-SafetyBench dataset from Parquet format to OmniSafeBench-MM TestCase format.

This script reads MM-SafetyBench Parquet data and converts it to the JSONL format
expected by the OmniSafeBench-MM framework, allowing direct evaluation without
regenerating test cases.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List
import pandas as pd
from PIL import Image
import io
from tqdm import tqdm


def load_mmsafetybench_parquet(data_dir: str, splits: List[str] = None) -> pd.DataFrame:
    """
    Load MM-SafetyBench Parquet files from all categories.

    Args:
        data_dir: Path to MM-SafetyBench data directory
        splits: List of split types to include (e.g., ['SD', 'SD_TYPO'])

    Returns:
        DataFrame with all records, including category and split_type columns
    """
    if splits is None:
        splits = ['SD', 'SD_TYPO']

    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    # Get all category directories
    category_dirs = [d for d in data_path.iterdir() if d.is_dir() and not d.name.startswith('.')]

    if not category_dirs:
        raise FileNotFoundError(f"No category directories found in {data_dir}")

    print(f"Found {len(category_dirs)} categories")

    dfs = []
    for category_dir in sorted(category_dirs):
        category_name = category_dir.name

        for split in splits:
            parquet_file = category_dir / f"{split}.parquet"

            if not parquet_file.exists():
                print(f"Warning: {parquet_file} not found, skipping")
                continue

            df = pd.read_parquet(parquet_file)
            df['category'] = category_name
            df['split_type'] = split
            dfs.append(df)
            print(f"Loaded {category_name}/{split}.parquet: {len(df)} records")

    if not dfs:
        raise ValueError(f"No parquet files found for splits: {splits}")

    combined_df = pd.concat(dfs, ignore_index=True)
    print(f"Total records: {len(combined_df)}")

    return combined_df


def extract_and_save_image(image_data, output_dir: str, image_id: str) -> str:
    """
    Extract binary image from Parquet and save to disk.

    Args:
        image_data: Binary image data or dictionary containing 'bytes' key
        output_dir: Directory to save images
        image_id: Unique identifier for the image (used as filename)

    Returns:
        Absolute path to saved image
    """
    os.makedirs(output_dir, exist_ok=True)

    # Extract binary data - handle both dict and direct bytes
    if isinstance(image_data, dict):
        image_bytes = image_data['bytes']
    else:
        image_bytes = image_data

    # Open image and save as PNG
    image = Image.open(io.BytesIO(image_bytes))
    image_path = os.path.join(output_dir, f"{image_id}.png")
    image.save(image_path)

    return os.path.abspath(image_path)


def convert_record_to_testcase(record: Dict, image_path: str, attack_name: str) -> Dict:
    """
    Convert MM-SafetyBench record to OmniSafeBench-MM TestCase format.

    Args:
        record: MM-SafetyBench record dictionary
        image_path: Absolute path to saved image
        attack_name: Name of attack method (e.g., 'mmsafetybench')

    Returns:
        TestCase-compatible dictionary
    """
    testcase = {
        "test_case_id": record['id'],
        "prompt": record['question'],
        "image_path": image_path,
        "metadata": {
            "attack_method": attack_name,
            "original_prompt": record['question'],
            "category": record['category'],
            "split_type": record['split_type']
        }
    }
    return testcase


def save_testcases_jsonl(testcases: List[Dict], output_file: str):
    """
    Save test cases to JSONL format.

    Args:
        testcases: List of TestCase dictionaries
        output_file: Path to output JSONL file
    """
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        for tc in testcases:
            f.write(json.dumps(tc, ensure_ascii=False) + '\n')

    print(f"Saved {len(testcases)} test cases to {output_file}")


def main():
    """Main conversion function."""
    parser = argparse.ArgumentParser(
        description="Convert MM-SafetyBench dataset to OmniSafeBench-MM TestCase format"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="/mnt/disk1/data/vlm_attack/MM-SafetyBench/data",
        help="Path to MM-SafetyBench data directory"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output/test_cases/mmsafetybench",
        help="Output directory for converted test cases"
    )
    parser.add_argument(
        "--attack-name",
        type=str,
        default="mmsafetybench",
        help="Attack method name"
    )
    parser.add_argument(
        "--splits",
        type=str,
        default="SD,SD_TYPO",
        help="Comma-separated list of splits to include (default: SD,SD_TYPO)"
    )

    args = parser.parse_args()

    # Parse splits
    splits = [s.strip() for s in args.splits.split(',')]

    print("=" * 60)
    print("MM-SafetyBench to OmniSafeBench-MM Conversion")
    print("=" * 60)
    print(f"Data directory: {args.data_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Attack name: {args.attack_name}")
    print(f"Splits to convert: {', '.join(splits)}")
    print()

    # Load MM-SafetyBench data
    print("Loading MM-SafetyBench dataset...")
    df = load_mmsafetybench_parquet(args.data_dir, splits=splits)
    print()

    # Create output directories
    images_dir = os.path.join(args.output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    # Convert records
    print("Converting records to TestCase format...")
    testcases = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing records"):
        # Extract and save image
        image_id = f"{row['category']}_{row['split_type']}_{idx}"
        image_path = extract_and_save_image(row['image'], images_dir, image_id)

        # Convert to TestCase format
        testcase = convert_record_to_testcase(row.to_dict(), image_path, args.attack_name)
        testcases.append(testcase)

    print()

    # Save to JSONL
    output_file = os.path.join(args.output_dir, "test_cases.jsonl")
    save_testcases_jsonl(testcases, output_file)

    # Print statistics
    print()
    print("=" * 60)
    print("Conversion Summary")
    print("=" * 60)
    print(f"Total test cases: {len(testcases)}")
    print(f"Images saved to: {images_dir}")
    print(f"Test cases saved to: {output_file}")

    # Print breakdown by category and split
    categories = df['category'].value_counts()
    split_types = df['split_type'].value_counts()

    print()
    print("Breakdown by category:")
    for category, count in sorted(categories.items()):
        print(f"  {category}: {count}")

    print()
    print("Breakdown by split type:")
    for split, count in sorted(split_types.items()):
        print(f"  {split}: {count}")

    print()
    print("Conversion completed successfully!")


if __name__ == "__main__":
    main()
