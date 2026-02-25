#!/usr/bin/env python3
"""
Convert HADES dataset from Parquet format to OmniSafeBench-MM TestCase format.

This script reads HADES adversarial attack data and converts it to the JSONL format
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


def load_hades_parquet(parquet_dir: str) -> pd.DataFrame:
    """
    Load HADES Parquet files and concatenate into single DataFrame.

    Args:
        parquet_dir: Path to directory containing HADES parquet files

    Returns:
        DataFrame with all HADES records
    """
    parquet_path = Path(parquet_dir) / "data"
    parquet_files = sorted(parquet_path.glob("*.parquet"))

    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {parquet_path}")

    print(f"Found {len(parquet_files)} parquet files")
    dfs = []
    for pf in parquet_files:
        df = pd.read_parquet(pf)
        dfs.append(df)
        print(f"Loaded {pf.name}: {len(df)} records")

    combined_df = pd.concat(dfs, ignore_index=True)
    print(f"Total records: {len(combined_df)}")

    return combined_df


def extract_and_save_image(image_data: dict, output_dir: str, image_id: str) -> str:
    """
    Extract binary image from Parquet and save to disk.

    Args:
        image_data: Dictionary containing 'bytes' key with image binary data
        output_dir: Directory to save images
        image_id: Unique identifier for the image (used as filename)

    Returns:
        Absolute path to saved image
    """
    os.makedirs(output_dir, exist_ok=True)

    # Extract binary data
    image_bytes = image_data['bytes']

    # Open image and save as PNG
    image = Image.open(io.BytesIO(image_bytes))
    image_path = os.path.join(output_dir, f"{image_id}.png")
    image.save(image_path)

    return os.path.abspath(image_path)


def convert_record_to_testcase(record: Dict, image_path: str, attack_name: str) -> Dict:
    """
    Convert HADES record to OmniSafeBench-MM TestCase format.

    Args:
        record: HADES record dictionary
        image_path: Absolute path to saved image
        attack_name: Name of attack method (e.g., 'hades')

    Returns:
        TestCase-compatible dictionary
    """
    testcase = {
        "test_case_id": record['id'],
        "prompt": record['instruction'],
        "image_path": image_path,
        "metadata": {
            "attack_method": attack_name,
            "original_prompt": record['instruction'],
            "scenario": record['scenario'],
            "category": record['category'],
            "step": int(record['step']),
            "keywords": record['keywords']
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
        description="Convert HADES dataset to OmniSafeBench-MM TestCase format"
    )
    parser.add_argument(
        "--hades-dir",
        type=str,
        default="/mnt/disk1/data/vlm_attack/HADES",
        help="Path to HADES dataset directory"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output/test_cases/hades",
        help="Output directory for converted test cases"
    )
    parser.add_argument(
        "--attack-name",
        type=str,
        default="hades",
        help="Attack method name"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("HADES to OmniSafeBench-MM Conversion")
    print("=" * 60)
    print(f"HADES directory: {args.hades_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Attack name: {args.attack_name}")
    print()

    # Load HADES data
    print("Loading HADES dataset...")
    df = load_hades_parquet(args.hades_dir)
    print()

    # Create output directories
    images_dir = os.path.join(args.output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    # Convert records
    print("Converting records to TestCase format...")
    testcases = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing records"):
        # Extract and save image
        image_id = row['id']
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

    # Print breakdown by scenario and step
    scenarios = df['scenario'].value_counts()
    steps = df['step'].value_counts().sort_index()

    print()
    print("Breakdown by scenario:")
    for scenario, count in scenarios.items():
        print(f"  {scenario}: {count}")

    print()
    print("Breakdown by optimization step:")
    for step, count in steps.items():
        print(f"  Step {step}: {count}")

    print()
    print("Conversion completed successfully!")


if __name__ == "__main__":
    main()
