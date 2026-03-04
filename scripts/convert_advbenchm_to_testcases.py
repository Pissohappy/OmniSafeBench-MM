#!/usr/bin/env python3
"""
Convert AdvBenchM dataset to OmniSafeBench-MM TestCase format.

AdvBenchM has 7 harmful categories, each with 1-2 images and a JSON file
containing multiple instructions. This script generates test cases as the
Cartesian product of images × instructions for each category.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List


def load_category_instructions(json_path: str) -> Dict:
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def convert_to_testcases(
    prompts_dir: str,
    images_dir: str,
    output_dir: str,
    attack_name: str = "advbenchm"
) -> List[Dict]:
    prompts_path = Path(prompts_dir)
    images_path = Path(images_dir)

    if not prompts_path.exists():
        raise FileNotFoundError(f"Prompts directory not found: {prompts_dir}")
    if not images_path.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")

    testcases = []
    tc_id = 0

    category_files = sorted(prompts_path.glob("*.json"))
    print(f"Found {len(category_files)} categories")

    for json_file in category_files:
        category = json_file.stem
        data = load_category_instructions(str(json_file))
        instructions = data.get("instructions", [])

        # Get images for this category
        cat_img_dir = images_path / category
        if not cat_img_dir.exists():
            print(f"Warning: image dir not found for {category}, skipping")
            continue

        image_files = sorted(
            [f for f in cat_img_dir.iterdir() if f.suffix.lower() in ('.jpg', '.jpeg', '.png')]
        )
        if not image_files:
            print(f"Warning: no images found for {category}, skipping")
            continue

        print(f"Category '{category}': {len(image_files)} image(s) x {len(instructions)} instructions")

        # Cartesian product: each image × each instruction
        for img_idx, img_file in enumerate(image_files):
            abs_img_path = str(img_file.resolve())
            for instr_idx, instruction in enumerate(instructions):
                testcase = {
                    "test_case_id": str(tc_id),
                    "prompt": instruction,
                    "image_path": abs_img_path,
                    "metadata": {
                        "attack_method": attack_name,
                        "original_prompt": instruction,
                        "category": category,
                        "image_idx": img_idx,
                        "instruction_idx": instr_idx,
                    }
                }
                testcases.append(testcase)
                tc_id += 1

    return testcases


def save_testcases_jsonl(testcases: List[Dict], output_file: str):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        for tc in testcases:
            f.write(json.dumps(tc, ensure_ascii=False) + '\n')
    print(f"Saved {len(testcases)} test cases to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert AdvBenchM dataset to OmniSafeBench-MM TestCase format"
    )
    parser.add_argument(
        "--prompts-dir",
        type=str,
        default="/mnt/disk1/data/vlm_attack/AdvBenchM/prompts/all_instructions",
        help="Path to AdvBenchM prompts directory (all_instructions)"
    )
    parser.add_argument(
        "--images-dir",
        type=str,
        default="/mnt/disk1/data/vlm_attack/AdvBenchM/images/harmful",
        help="Path to AdvBenchM harmful images directory"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output/test_cases/advbenchm",
        help="Output directory for converted test cases"
    )
    parser.add_argument(
        "--attack-name",
        type=str,
        default="advbenchm",
        help="Attack method name for metadata"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("AdvBenchM to OmniSafeBench-MM Conversion")
    print("=" * 60)
    print(f"Prompts directory: {args.prompts_dir}")
    print(f"Images directory:  {args.images_dir}")
    print(f"Output directory:  {args.output_dir}")
    print(f"Attack name:       {args.attack_name}")
    print()

    testcases = convert_to_testcases(
        prompts_dir=args.prompts_dir,
        images_dir=args.images_dir,
        output_dir=args.output_dir,
        attack_name=args.attack_name,
    )

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
