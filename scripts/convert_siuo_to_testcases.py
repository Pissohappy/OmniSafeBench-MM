#!/usr/bin/env python3
"""
Convert SIUO dataset from JSON format to OmniSafeBench-MM TestCase format.

This script reads SIUO (Safe Inputs but Unsafe Output) JSON data and converts it
to the JSONL format expected by the OmniSafeBench-MM framework, allowing direct
evaluation without regenerating test cases.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List
from tqdm import tqdm


def load_siuo_json(json_path: str) -> List[Dict]:
    """
    Load SIUO JSON file.

    Args:
        json_path: Path to SIUO JSON file

    Returns:
        List of record dictionaries
    """
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"JSON file not found: {json_path}")

    print(f"Loading JSON file: {json_path}")
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"Loaded {len(data)} records")
    return data


def resolve_image_path(image_filename: str, images_dir: str) -> str:
    """
    Convert image filename to absolute path and verify it exists.

    Args:
        image_filename: Image filename (e.g., "S-01.png")
        images_dir: Directory containing images

    Returns:
        Absolute path to image file

    Raises:
        FileNotFoundError: If image file doesn't exist
    """
    # Construct absolute path
    abs_path = os.path.join(images_dir, image_filename)
    abs_path = os.path.abspath(abs_path)

    # Verify file exists
    if not os.path.exists(abs_path):
        raise FileNotFoundError(f"Image file not found: {abs_path}")

    return abs_path


def convert_record_to_testcase(record: Dict, images_dir: str, attack_name: str) -> Dict:
    """
    Convert SIUO record to OmniSafeBench-MM TestCase format.

    Args:
        record: SIUO record dictionary
        images_dir: Directory containing images
        attack_name: Name of attack method (e.g., 'siuo')

    Returns:
        TestCase-compatible dictionary
    """
    # Resolve image path
    image_path = resolve_image_path(record['image'], images_dir)

    # Create TestCase
    testcase = {
        "test_case_id": str(record['question_id']),
        "prompt": record['question'],
        "image_path": image_path,
        "metadata": {
            "attack_method": attack_name,
            "original_prompt": record['question'],
            "category": record['category'],
            "safety_warning": record.get('safety_warning', ''),
            "reference_answer": record.get('reference_answer', '')
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
        description="Convert SIUO dataset to OmniSafeBench-MM TestCase format"
    )
    parser.add_argument(
        "--json-file",
        type=str,
        default="/mnt/disk1/data/vlm_attack/SIUO/siuo_gen.json",
        help="Path to SIUO JSON file"
    )
    parser.add_argument(
        "--images-dir",
        type=str,
        default="/mnt/disk1/data/vlm_attack/SIUO/images",
        help="Directory containing images"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output/test_cases/siuo",
        help="Output directory for converted test cases"
    )
    parser.add_argument(
        "--attack-name",
        type=str,
        default="siuo",
        help="Attack method name"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("SIUO to OmniSafeBench-MM Conversion")
    print("=" * 60)
    print(f"JSON file: {args.json_file}")
    print(f"Images directory: {args.images_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Attack name: {args.attack_name}")
    print()

    # Load JSON data
    print("Loading SIUO dataset...")
    data = load_siuo_json(args.json_file)
    print()

    # Convert records
    print("Converting records to TestCase format...")
    testcases = []
    missing_images = []

    for record in tqdm(data, desc="Processing records"):
        try:
            # Convert to TestCase format
            testcase = convert_record_to_testcase(record, args.images_dir, args.attack_name)
            testcases.append(testcase)
        except (FileNotFoundError, ValueError) as e:
            missing_images.append((record['question_id'], str(e)))
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

    # Print breakdown by category
    if testcases:
        categories = {}
        for tc in testcases:
            cat = tc['metadata']['category']
            categories[cat] = categories.get(cat, 0) + 1

        print()
        print("Breakdown by safety category:")
        for category, count in sorted(categories.items()):
            print(f"  {category}: {count}")

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
