#!/usr/bin/env python3
"""
Convert mssbench dataset to OmniSafeBench-MM TestCase format.

mssbench has two subsets in combined.json:
  - chat: 300 records, images in chat/
  - embodied: 76 records, images in embodied/

For each record: use unsafe_image_path + queries[0] (chat) or
unsafe + unsafe_instructions[0] (embodied).
"""

import argparse
import json
import os
from typing import Dict, List


def load_combined_json(json_path: str) -> Dict:
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"JSON file not found: {json_path}")
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"Loaded combined.json: chat={len(data.get('chat', []))}, embodied={len(data.get('embodied', []))}")
    return data


def convert_chat_records(
    records: List[Dict],
    images_dir: str,
    attack_name: str = "mssbench"
) -> List[Dict]:
    testcases = []
    missing = []

    for i, rec in enumerate(records):
        img_name = rec.get('unsafe_image_path', '')
        if not img_name:
            missing.append((i, "No unsafe_image_path"))
            continue

        abs_path = os.path.join(images_dir, img_name)
        abs_path = os.path.abspath(abs_path)
        if not os.path.exists(abs_path):
            missing.append((i, f"Image not found: {abs_path}"))
            continue

        queries = rec.get('queries', [])
        if not queries:
            missing.append((i, "No queries"))
            continue

        prompt = queries[0].strip()
        if not prompt:
            missing.append((i, "Empty first query"))
            continue

        testcase = {
            "test_case_id": f"chat_{i}",
            "prompt": prompt,
            "image_path": abs_path,
            "metadata": {
                "attack_method": attack_name,
                "original_prompt": prompt,
                "subset": "chat",
                "intent": rec.get('intent', ''),
                "type": rec.get('Type', ''),
                "safe_image_path": os.path.abspath(
                    os.path.join(images_dir, rec.get('safe_image_path', ''))
                ) if rec.get('safe_image_path') else '',
            }
        }
        testcases.append(testcase)

    if missing:
        print(f"  Skipped {len(missing)} chat records")
        for id_, msg in missing[:3]:
            print(f"    {id_}: {msg}")

    return testcases


def convert_embodied_records(
    records: List[Dict],
    images_dir: str,
    attack_name: str = "mssbench"
) -> List[Dict]:
    testcases = []
    missing = []

    for i, rec in enumerate(records):
        img_name = rec.get('unsafe', '')
        if not img_name:
            missing.append((i, "No 'unsafe' image field"))
            continue

        abs_path = os.path.join(images_dir, img_name)
        abs_path = os.path.abspath(abs_path)
        if not os.path.exists(abs_path):
            missing.append((i, f"Image not found: {abs_path}"))
            continue

        unsafe_instructions = rec.get('unsafe_instructions', [])
        if not unsafe_instructions:
            # Fallback to unsafe_instruction (singular)
            instr = rec.get('unsafe_instruction', '').strip()
            if not instr:
                missing.append((i, "No unsafe instructions"))
                continue
        else:
            instr = unsafe_instructions[0].strip()

        if not instr:
            missing.append((i, "Empty instruction"))
            continue

        testcase = {
            "test_case_id": f"embodied_{i}",
            "prompt": instr,
            "image_path": abs_path,
            "metadata": {
                "attack_method": attack_name,
                "original_prompt": instr,
                "subset": "embodied",
                "task": rec.get('task', ''),
                "category": rec.get('category', ''),
                "observation_unsafe": rec.get('observation_unsafe', ''),
            }
        }
        testcases.append(testcase)

    if missing:
        print(f"  Skipped {len(missing)} embodied records")
        for id_, msg in missing[:3]:
            print(f"    {id_}: {msg}")

    return testcases


def save_testcases_jsonl(testcases: List[Dict], output_file: str):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        for tc in testcases:
            f.write(json.dumps(tc, ensure_ascii=False) + '\n')
    print(f"Saved {len(testcases)} test cases to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert mssbench dataset to OmniSafeBench-MM TestCase format"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="/mnt/disk1/data/vlm_attack/mssbench",
        help="Path to mssbench root directory"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output/test_cases/mssbench",
        help="Output directory for converted test cases"
    )
    parser.add_argument(
        "--attack-name",
        type=str,
        default="mssbench",
        help="Attack method name for metadata"
    )
    parser.add_argument(
        "--subsets",
        type=str,
        default="chat,embodied",
        help="Comma-separated subsets to include (chat, embodied)"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("mssbench to OmniSafeBench-MM Conversion")
    print("=" * 60)
    print(f"Data directory: {args.data_dir}")
    print(f"Output dir:     {args.output_dir}")
    print(f"Attack name:    {args.attack_name}")
    print(f"Subsets:        {args.subsets}")
    print()

    json_path = os.path.join(args.data_dir, "combined.json")
    data = load_combined_json(json_path)

    subsets = [s.strip() for s in args.subsets.split(',')]
    all_testcases = []

    if 'chat' in subsets:
        chat_dir = os.path.join(args.data_dir, "chat")
        print(f"\nProcessing chat subset ({len(data.get('chat', []))} records)...")
        chat_cases = convert_chat_records(data.get('chat', []), chat_dir, args.attack_name)
        print(f"  Generated {len(chat_cases)} chat test cases")
        all_testcases.extend(chat_cases)

    if 'embodied' in subsets:
        embodied_dir = os.path.join(args.data_dir, "embodied")
        print(f"\nProcessing embodied subset ({len(data.get('embodied', []))} records)...")
        emb_cases = convert_embodied_records(data.get('embodied', []), embodied_dir, args.attack_name)
        print(f"  Generated {len(emb_cases)} embodied test cases")
        all_testcases.extend(emb_cases)

    output_file = os.path.join(args.output_dir, "test_cases.jsonl")
    save_testcases_jsonl(all_testcases, output_file)

    print()
    print("=" * 60)
    print("Conversion Summary")
    print("=" * 60)
    print(f"Total test cases: {len(all_testcases)}")

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
