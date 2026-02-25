#!/usr/bin/env python3
"""Compare ASR and fine-grained scores across attack methods from evaluation jsonl files."""

import argparse
import json
from collections import defaultdict
from pathlib import Path


def load_records(paths):
    records = []
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                records.append(json.loads(line))
    return records


def safe_mean(values):
    return round(sum(values) / len(values), 4) if values else None


def summarize_by_attack(records):
    grouped = defaultdict(lambda: {
        "total": 0,
        "success": 0,
        "judge_score": [],
        "harmfulness_score": [],
        "alignment_score": [],
        "detail_score": [],
    })

    for item in records:
        meta = item.get("metadata", {}) or {}
        attack = item.get("attack_method") or meta.get("attack_method") or "unknown"
        bucket = grouped[attack]
        bucket["total"] += 1
        if item.get("success"):
            bucket["success"] += 1

        if item.get("judge_score") is not None:
            bucket["judge_score"].append(item["judge_score"])

        for key in ["harmfulness_score", "alignment_score", "detail_score"]:
            if meta.get(key) is not None:
                bucket[key].append(meta[key])

    summary = []
    for attack, stats in grouped.items():
        total = stats["total"]
        success = stats["success"]
        summary.append({
            "attack_method": attack,
            "total_cases": total,
            "successful_cases": success,
            "asr": round(success / total, 4) if total else None,
            "avg_judge_score": safe_mean(stats["judge_score"]),
            "avg_harmfulness": safe_mean(stats["harmfulness_score"]),
            "avg_alignment": safe_mean(stats["alignment_score"]),
            "avg_detail": safe_mean(stats["detail_score"]),
        })

    return sorted(summary, key=lambda x: x["attack_method"])


def main():
    parser = argparse.ArgumentParser(description="按 attack_method 聚合评测结果并对比 ASR/细分评分")
    parser.add_argument("--inputs", nargs="+", required=True, help="一个或多个 evaluation_results.jsonl 文件")
    parser.add_argument("--output", default="output/compare_attack_summary.json", help="聚合输出文件路径")
    args = parser.parse_args()

    for p in args.inputs:
        if not Path(p).exists():
            raise FileNotFoundError(f"Input file not found: {p}")

    records = load_records(args.inputs)
    summary = summarize_by_attack(records)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\n✅ Saved attack comparison summary to: {output_path}")


if __name__ == "__main__":
    main()
