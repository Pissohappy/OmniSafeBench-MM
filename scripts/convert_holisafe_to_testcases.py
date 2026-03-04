#!/usr/bin/env python3
"""
Convert holisafe-bench dataset to OmniSafeBench-MM TestCase format.

holisafe-bench provides a JSON file with 4031 entries; images are stored
locally in images/{category}/{subcategory}/ following the relative path
in the 'image' field of each record.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List


def load_holisafe_json(json_path: str) -> List[Dict]:
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"JSON file not found: {json_path}")
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"Loaded {len(data)} records from {json_path}")
    return data


def resolve_image_path(relative_path: str, images_base: str) -> str:
    abs_path = os.path.join(images_base, relative_path)
    abs_path = os.path.abspath(abs_path)
    if not os.path.exists(abs_path):
        raise FileNotFoundError(f"Image not found: {abs_path}")
    return abs_path


def convert_to_testcases(
    records: List[Dict],
    images_base: str,
    attack_name: str = "holisafe"
) -> List[Dict]:
    testcases = []
    missing = []

    for i, record in enumerate(records):
        try:
            image_path = resolve_image_path(record['image'], images_base)
        except FileNotFoundError as e:
            missing.append((record.get('id', i), str(e)))
            continue

        testcase = {
            "test_case_id": str(record['id']),
            "prompt": record['query'],
            "image_path": image_path,
            "metadata": {
                "attack_method": attack_name,
                "original_prompt": record['query'],
                "category": record.get('category', ''),
                "subcategory": record.get('subcategory', ''),
                "type": record.get('type', ''),
                "image_safe": record.get('image_safe', None),
                "image_safety_label": record.get('image_safety_label', None),
            }
        }
        testcases.append(testcase)

    if missing:
        print(f"Warning: {len(missing)} images not found")
        for id_, msg in missing[:5]:
            print(f"  ID {id_}: {msg}")
        if len(missing) > 5:
            print(f"  ... and {len(missing) - 5} more")

    return testcases


def save_testcases_jsonl(testcases: List[Dict], output_file: str):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        for tc in testcases:
            f.write(json.dumps(tc, ensure_ascii=False) + '\n')
    print(f"Saved {len(testcases)} test cases to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert holisafe-bench dataset to OmniSafeBench-MM TestCase format"
    )
    parser.add_argument(
        "--json-file",
        type=str,
        default="/mnt/disk1/data/vlm_attack/holisafe-bench/holisafe_bench.json",
        help="Path to holisafe_bench.json"
    )
    parser.add_argument(
        "--images-base",
        type=str,
        default="/mnt/disk1/data/vlm_attack/holisafe-bench/images",
        help="Base directory for resolving image relative paths"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output/test_cases/holisafe",
        help="Output directory for converted test cases"
    )
    parser.add_argument(
        "--attack-name",
        type=str,
        default="holisafe",
        help="Attack method name for metadata"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("holisafe-bench to OmniSafeBench-MM Conversion")
    print("=" * 60)
    print(f"JSON file:    {args.json_file}")
    print(f"Images base:  {args.images_base}")
    print(f"Output dir:   {args.output_dir}")
    print(f"Attack name:  {args.attack_name}")
    print()

    records = load_holisafe_json(args.json_file)
    testcases = convert_to_testcases(records, args.images_base, args.attack_name)

    output_file = os.path.join(args.output_dir, "test_cases.jsonl")
    save_testcases_jsonl(testcases, output_file)

    print()
    print("=" * 60)
    print("Conversion Summary")
    print("=" * 60)
    print(f"Total test cases: {len(testcases)}")

    categories = {}
    for tc in testcases:
        cat = tc['metadata']['category']
        categories[cat] = categories.get(cat, 0) + 1
    print()
    print("Breakdown by category:")
    for cat, count in sorted(categories.items()):
        print(f"  {cat}: {count}")

    print()
    print("Conversion completed successfully!")


if __name__ == "__main__":
    main()
