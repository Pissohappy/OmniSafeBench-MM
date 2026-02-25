#!/usr/bin/env python3
"""
Convert JailBreakV-28k dataset from CSV format to OmniSafeBench-MM TestCase format.

This script reads JailBreakV-28k CSV data and converts it to the JSONL format
expected by the OmniSafeBench-MM framework, allowing direct evaluation without
regenerating test cases.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List
import pandas as pd
from tqdm import tqdm


def load_jailbreakv_csv(csv_path: str) -> pd.DataFrame:
    """
    Load JailBreakV-28k CSV file.

    Args:
        csv_path: Path to CSV file

    Returns:
        DataFrame with all records
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    print(f"Loading CSV file: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} records")

    return df


def resolve_image_path(relative_path: str, base_dir: str) -> str:
    """
    Convert relative image path to absolute path and verify it exists.

    Args:
        relative_path: Relative path from CSV
        base_dir: Base directory for resolving paths

    Returns:
        Absolute path to image file

    Raises:
        FileNotFoundError: If image file doesn't exist
    """
    # Handle NaN or empty paths
    if pd.isna(relative_path) or not relative_path:
        raise ValueError("Image path is empty or NaN")

    # Construct absolute path
    abs_path = os.path.join(base_dir, relative_path)
    abs_path = os.path.abspath(abs_path)

    # Verify file exists
    if not os.path.exists(abs_path):
        raise FileNotFoundError(f"Image file not found: {abs_path}")

    return abs_path


def convert_record_to_testcase(record: Dict, base_dir: str, attack_name: str) -> Dict:
    """
    Convert JailBreakV record to OmniSafeBench-MM TestCase format.

    Args:
        record: JailBreakV record dictionary
        base_dir: Base directory for resolving image paths
        attack_name: Name of attack method (e.g., 'jailbreakv')

    Returns:
        TestCase-compatible dictionary
    """
    # Resolve image path
    image_path = resolve_image_path(record['image_path'], base_dir)

    # Create TestCase
    testcase = {
        "test_case_id": str(record['id']),
        "prompt": record['jailbreak_query'],
        "image_path": image_path,
        "metadata": {
            "attack_method": record['format'],
            "policy": record['policy'],
            "original_prompt": record['redteam_query'],
            "source": record['from'],
            "transfer_from_llm": bool(record['transfer_from_llm']),
            "selected_mini": bool(record['selected_mini'])
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
        description="Convert JailBreakV-28k dataset to OmniSafeBench-MM TestCase format"
    )
    parser.add_argument(
        "--csv-file",
        type=str,
        default="/mnt/disk1/data/vlm_attack/JailBreakV-28k/JailBreakV_28K/JailBreakV_28K.csv",
        help="Path to JailBreakV CSV file"
    )
    parser.add_argument(
        "--base-dir",
        type=str,
        default="/mnt/disk1/data/vlm_attack/JailBreakV-28k/JailBreakV_28K",
        help="Base directory for resolving image paths"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output/test_cases/jailbreakv",
        help="Output directory for converted test cases"
    )
    parser.add_argument(
        "--attack-name",
        type=str,
        default="jailbreakv",
        help="Attack method name"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("JailBreakV-28k to OmniSafeBench-MM Conversion")
    print("=" * 60)
    print(f"CSV file: {args.csv_file}")
    print(f"Base directory: {args.base_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Attack name: {args.attack_name}")
    print()

    # Load CSV data
    print("Loading JailBreakV-28k dataset...")
    df = load_jailbreakv_csv(args.csv_file)
    print()

    # Convert records
    print("Converting records to TestCase format...")
    testcases = []
    missing_images = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing records"):
        try:
            # Convert to TestCase format
            testcase = convert_record_to_testcase(row.to_dict(), args.base_dir, args.attack_name)
            testcases.append(testcase)
        except (FileNotFoundError, ValueError) as e:
            missing_images.append((row['id'], str(e)))
            continue

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
    print(f"Missing images: {len(missing_images)}")
    print(f"Test cases saved to: {output_file}")

    # Print breakdown by attack method
    if testcases:
        formats = {}
        policies = {}
        sources = {}

        for tc in testcases:
            fmt = tc['metadata']['attack_method']
            policy = tc['metadata']['policy']
            source = tc['metadata']['source']

            formats[fmt] = formats.get(fmt, 0) + 1
            policies[policy] = policies.get(policy, 0) + 1
            sources[source] = sources.get(source, 0) + 1

        print()
        print("Breakdown by attack method:")
        for fmt, count in sorted(formats.items()):
            print(f"  {fmt}: {count}")

        print()
        print("Breakdown by safety policy:")
        for policy, count in sorted(policies.items()):
            print(f"  {policy}: {count}")

        print()
        print("Breakdown by data source:")
        for source, count in sorted(sources.items()):
            print(f"  {source}: {count}")

    # Print missing images if any
    if missing_images:
        print()
        print(f"Warning: {len(missing_images)} images were not found:")
        for img_id, error in missing_images[:10]:  # Show first 10
            print(f"  ID {img_id}: {error}")
        if len(missing_images) > 10:
            print(f"  ... and {len(missing_images) - 10} more")

    print()
    print("Conversion completed successfully!")


if __name__ == "__main__":
    main()
