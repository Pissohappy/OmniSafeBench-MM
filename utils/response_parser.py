import re
from typing import Dict, Optional


_TAG_PAIR_PATTERN = re.compile(
    r"<\s*(think|answer|result)\s*>(.*?)<\s*/\s*\1\s*>",
    re.IGNORECASE | re.DOTALL,
)
_THINK_PAIR_PATTERN = re.compile(
    r"<\s*think\s*>(.*?)<\s*/\s*think\s*>",
    re.IGNORECASE | re.DOTALL,
)
_THINK_ORPHAN_TAG_PATTERN = re.compile(r"</?\s*think\s*>", re.IGNORECASE)


def _normalize_text(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    cleaned = value.strip()
    return cleaned if cleaned else None


def parse_reasoning_and_answer(text: str, strategy: str = "auto") -> Dict[str, Optional[str]]:
    """Parse model output into reasoning and final answer.

    Returns keys:
    - reasoning_trace
    - final_answer
    - parse_status
    - raw_text
    """
    raw_text = "" if text is None else str(text)

    if strategy == "raw_only":
        return {
            "reasoning_trace": None,
            "final_answer": raw_text,
            "parse_status": "fallback_raw",
            "raw_text": raw_text,
        }

    if strategy not in {"auto", "tag_only", "raw_only"}:
        raise ValueError(
            f"Unsupported parse strategy: {strategy}. Use one of: auto, tag_only, raw_only"
        )

    reasoning_segments = []
    answer_segments = []

    for match in _TAG_PAIR_PATTERN.finditer(raw_text):
        tag_name = match.group(1).lower()
        content = _normalize_text(match.group(2))
        if not content:
            continue
        if tag_name == "think":
            reasoning_segments.append(content)
        elif tag_name in {"answer", "result"}:
            answer_segments.append(content)

    reasoning_trace = (
        "\n\n".join(reasoning_segments) if reasoning_segments else None
    )

    # 1) Explicit <answer>/<result> wins.
    if answer_segments:
        final_answer = "\n\n".join(answer_segments)
        return {
            "reasoning_trace": reasoning_trace,
            "final_answer": final_answer,
            "parse_status": "parsed_explicit_tags",
            "raw_text": raw_text,
        }

    # tag_only mode stops here if no explicit final answer tags.
    if strategy == "tag_only":
        return {
            "reasoning_trace": reasoning_trace,
            "final_answer": raw_text,
            "parse_status": "fallback_raw",
            "raw_text": raw_text,
        }

    # 2) Think-only: remove think segments and use remaining text.
    has_think_signal = bool(
        reasoning_segments
        or re.search(r"<\s*/?\s*think\b", raw_text, flags=re.IGNORECASE)
    )
    if has_think_signal:
        text_without_think_blocks = _THINK_PAIR_PATTERN.sub("", raw_text)
        text_without_think = _THINK_ORPHAN_TAG_PATTERN.sub("", text_without_think_blocks)
        final_answer = _normalize_text(text_without_think)

        return {
            "reasoning_trace": reasoning_trace,
            "final_answer": final_answer or "",
            "parse_status": "parsed_think_only",
            "raw_text": raw_text,
        }

    # 3) No tags fallback.
    return {
        "reasoning_trace": None,
        "final_answer": raw_text,
        "parse_status": "fallback_raw",
        "raw_text": raw_text,
    }
