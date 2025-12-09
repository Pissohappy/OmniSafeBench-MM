"""
Data utility functions
"""

import json
from pathlib import Path
from typing import Dict, List, Any


def load_json(file_path: str) -> Any:
    """Load JSON file"""
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Any, file_path: str, indent: int = 2) -> str:
    """Save data to JSON file"""
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)

    return str(file_path)


def validate_data_format(data: Dict[str, Any], expected_keys: List[str]) -> bool:
    """Validate data format"""
    return all(key in data for key in expected_keys)
