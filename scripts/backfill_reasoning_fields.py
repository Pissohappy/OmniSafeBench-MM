#!/usr/bin/env python3
"""Backfill reasoning-related fields for historical response JSONL files.

This script is intentionally non-destructive:
- It never overwrites existing `reasoning_trace`, `final_answer`, `response_parse_status`.
- It only fills missing/null fields.
"""

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable


THINK_TAG_PATTERN = re.compile(r"<think>(.*?)</think>", flags=re.IGNORECASE | re.DOTALL)


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at line {idx} in {path}: {exc}") from exc


def split_response(model_response: str, strategy: str) -> tuple[Any, str, str]:
    if strategy == "off":
        return None, model_response, "strategy_off"

    text = model_response or ""
    match = THINK_TAG_PATTERN.search(text)
    if match:
        reasoning_trace = (match.group(1) or "").strip() or None
        final_answer = THINK_TAG_PATTERN.sub("", text, count=1).strip() or text
        return reasoning_trace, final_answer, "split_by_tag"

    return None, text, "fallback_no_tag"


def main() -> int:
    parser = argparse.ArgumentParser(description="Backfill reasoning fields for response JSONL")
    parser.add_argument("input", type=Path, help="Input responses JSONL path")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path (default: <input>.backfilled.jsonl)",
    )
    parser.add_argument(
        "--strategy",
        choices=["auto", "tag_only", "off"],
        default="auto",
        help="Reasoning split strategy used for backfill",
    )
    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Input file does not exist: {args.input}")

    output = args.output or args.input.with_suffix(".backfilled.jsonl")

    updated = 0
    total = 0
    with output.open("w", encoding="utf-8") as out_f:
        for row in iter_jsonl(args.input):
            total += 1

            model_response = row.get("model_response", "")
            reasoning_trace, final_answer, parse_status = split_response(
                model_response, args.strategy
            )

            changed = False
            if row.get("reasoning_trace") is None:
                row["reasoning_trace"] = reasoning_trace
                changed = True
            if row.get("final_answer") is None:
                # compatibility: fallback to model_response for old JSONL rows
                row["final_answer"] = final_answer if final_answer else model_response
                changed = True
            if row.get("response_parse_status") is None:
                row["response_parse_status"] = parse_status
                changed = True

            if changed:
                updated += 1

            out_f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Done. total={total}, updated={updated}, output={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
