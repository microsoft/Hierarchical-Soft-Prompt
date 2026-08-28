"""
Model output parser: text → dict.

Tries in order:
  1. Direct JSON parse
  2. Extract from markdown code block (```json ... ```)
  3. Extract first {...} block via regex
  4. Failure — returns empty dict and increments failure count
"""

from __future__ import annotations
import json
import re
from typing import Optional


def parse_json_output(text: str) -> tuple[Optional[dict], bool]:
    """
    Parse model output text to a JSON dict.

    Returns:
        (parsed_dict, success)
        If success=False, parsed_dict is {} (empty dict, not None).
    """
    if not text or not isinstance(text, str):
        return {}, False

    text = text.strip()

    # Attempt 1: direct parse
    try:
        result = json.loads(text)
        if isinstance(result, dict):
            return result, True
    except json.JSONDecodeError:
        pass

    # Attempt 2: extract from markdown fences
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence_match:
        try:
            result = json.loads(fence_match.group(1))
            if isinstance(result, dict):
                return result, True
        except json.JSONDecodeError:
            pass

    # Attempt 3: find outermost { ... } block
    # Find first { and match to its closing }
    start = text.find("{")
    if start >= 0:
        # Walk to find matching brace
        depth = 0
        for i, ch in enumerate(text[start:], start=start):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start : i + 1]
                    try:
                        result = json.loads(candidate)
                        if isinstance(result, dict):
                            return result, True
                    except json.JSONDecodeError:
                        # Try fixing common issues: trailing commas
                        fixed = re.sub(r",\s*([}\]])", r"\1", candidate)
                        try:
                            result = json.loads(fixed)
                            if isinstance(result, dict):
                                return result, True
                        except json.JSONDecodeError:
                            pass
                    break

    return {}, False


def batch_parse(texts: list[str]) -> tuple[list[dict], int]:
    """
    Parse a list of model outputs.

    Returns:
        (parsed_list, n_failures)
        parsed_list has {} for failed parses (not excluded).
    """
    parsed = []
    failures = 0
    for text in texts:
        result, success = parse_json_output(text)
        parsed.append(result)
        if not success:
            failures += 1
    return parsed, failures
