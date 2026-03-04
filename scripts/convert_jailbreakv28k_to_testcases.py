#!/usr/bin/env python3
"""
Convert JailBreakV-28k dataset to OmniSafeBench-MM TestCase format.

Uses the full JailBreakV_28K.csv (or mini_JailBreakV_28K.csv), filtering
out rows that lack a valid image_path. Images are stored locally relative
to the JailBreakV_28K directory.
"""

import argparse
import csv
import json
import os
from typing import Dict, List


def load_csv(csv_path: str) -> List[Dict]:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    print(f"Loaded {len(rows)} rows from {csv_path}")
    return rows


def resolve_image_path(relative_path: str, base_dir: str) -> str:
    if not relative_path or relative_path.strip() == '':
        raise ValueError("Empty image path")
    abs_path = os.path.join(base_dir, relative_path.strip())
    abs_path = os.path.abspath(abs_path)
    if not os.path.exists(abs_path):
        raise FileNotFoundError(f"Image not found: {abs_path}")
    return abs_path


def convert_to_testcases(
    rows: List[Dict],
    base_dir: str,
    attack_name: str = "jailbreakv28k"
) -> List[Dict]:
    testcases = []
    missing = []

    for row in rows:
        img_rel = row.get('image_path', '')
        if not img_rel or img_rel.strip() in ('', 'None', 'nan'):
            missing.append((row.get('id', '?'), "No image path"))
            continue

        try:
            image_path = resolve_image_path(img_rel, base_dir)
        except (FileNotFoundError, ValueError) as e:
            missing.append((row.get('id', '?'), str(e)))
            continue

        jailbreak_query = row.get('jailbreak_query', '') or ''
        redteam_query = row.get('redteam_query', '') or ''

        # Use jailbreak_query as prompt if non-empty, else fall back to redteam_query
        prompt = jailbreak_query.strip() if jailbreak_query.strip() else redteam_query.strip()
        if not prompt:
            missing.append((row.get('id', '?'), "Empty prompt"))
            continue

        def parse_bool(val):
            if val is None:
                return None
            if isinstance(val, bool):
                return val
            s = str(val).strip().lower()
            if s in ('true', '1', 'yes'):
                return True
            if s in ('false', '0', 'no'):
                return False
            return None

        testcase = {
            "test_case_id": str(row.get('id', '')),
            "prompt": prompt,
            "image_path": image_path,
            "metadata": {
                "attack_method": attack_name,
                "original_prompt": redteam_query.strip() if redteam_query.strip() else prompt,
                "policy": row.get('policy') or '',
                "format": row.get('format') or '',
                "source": row.get('from') or '',
                "transfer_from_llm": parse_bool(row.get('transfer_from_llm')),
                "selected_mini": parse_bool(row.get('selected_mini')),
            }
        }
        testcases.append(testcase)

    if missing:
        print(f"Skipped {len(missing)} records (missing image or empty prompt)")
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
        description="Convert JailBreakV-28k dataset to OmniSafeBench-MM TestCase format"
    )
    parser.add_argument(
        "--csv-file",
        type=str,
        default="/mnt/disk1/data/vlm_attack/JailBreakV-28k/JailBreakV_28K/JailBreakV_28K.csv",
        help="Path to JailBreakV CSV file (full or mini)"
    )
    parser.add_argument(
        "--base-dir",
        type=str,
        default="/mnt/disk1/data/vlm_attack/JailBreakV-28k/JailBreakV_28K",
        help="Base directory for resolving relative image paths"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output/test_cases/jailbreakv28k",
        help="Output directory for converted test cases"
    )
    parser.add_argument(
        "--attack-name",
        type=str,
        default="jailbreakv28k",
        help="Attack method name for metadata"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("JailBreakV-28k to OmniSafeBench-MM Conversion")
    print("=" * 60)
    print(f"CSV file:    {args.csv_file}")
    print(f"Base dir:    {args.base_dir}")
    print(f"Output dir:  {args.output_dir}")
    print(f"Attack name: {args.attack_name}")
    print()

    rows = load_csv(args.csv_file)
    testcases = convert_to_testcases(rows, args.base_dir, args.attack_name)

    output_file = os.path.join(args.output_dir, "test_cases.jsonl")
    save_testcases_jsonl(testcases, output_file)

    print()
    print("=" * 60)
    print("Conversion Summary")
    print("=" * 60)
    print(f"Total test cases: {len(testcases)}")

    policies = {}
    for tc in testcases:
        p = tc['metadata']['policy'] or 'N/A'
        policies[p] = policies.get(p, 0) + 1
    print()
    print("Breakdown by policy:")
    for p, count in sorted(policies.items()):
        print(f"  {p}: {count}")

    print()
    print("Conversion completed successfully!")


if __name__ == "__main__":
    main()
