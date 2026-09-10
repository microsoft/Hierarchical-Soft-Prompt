"""
Schema utilities: DFS traversal, key-path operations, nested dict access.

The DFS pre-order key list is the single source of truth for:
  - SchemaHierarchicalSoftPrompt token ordering
  - FieldLoss key iteration
  - Metric computation
"""

from __future__ import annotations
from typing import Any


def dfs_keys(schema: dict, prefix: str = "", depth: int = 0) -> list[tuple[str, int]]:
    """
    DFS pre-order traversal of a JSON schema dict.

    Returns list of (key_path, depth) tuples where key_path uses dot notation.

    Example:
        schema = {"record": {"id": "str", "subject": {"name": "str"}}}
        -> [("record", 0), ("record.id", 1), ("record.subject", 1),
            ("record.subject.name", 2)]
    """
    result = []
    if isinstance(schema, dict):
        for key, value in schema.items():
            full_key = f"{prefix}.{key}" if prefix else key
            result.append((full_key, depth))
            result.extend(dfs_keys(value, full_key, depth + 1))
    elif isinstance(schema, list) and len(schema) > 0:
        # For arrays, traverse the element schema (first element as representative)
        result.extend(dfs_keys(schema[0], prefix, depth))
    return result


def dfs_key_paths(schema: dict) -> list[str]:
    """Returns only the key paths in DFS pre-order (no depths)."""
    return [kp for kp, _ in dfs_keys(schema)]


def dfs_leaf_keys(schema: dict, prefix: str = "", depth: int = 0) -> list[tuple[str, int]]:
    """
    DFS traversal returning only leaf keys (keys with scalar values).
    Used for field-level metric computation where only leaf values exist.
    """
    result = []
    if isinstance(schema, dict):
        for key, value in schema.items():
            full_key = f"{prefix}.{key}" if prefix else key
            if isinstance(value, dict) and value:
                result.extend(dfs_leaf_keys(value, full_key, depth + 1))
            elif isinstance(value, list) and value and isinstance(value[0], dict):
                result.extend(dfs_leaf_keys(value[0], full_key, depth + 1))
            else:
                result.append((full_key, depth))
    return result


def get_nested(obj: dict, key_path: str) -> Any:
    """
    Get value at dot-notation key path from nested dict.

    Returns None if path doesn't exist (does not raise).
    Handles list elements by taking the first element.
    """
    if obj is None:
        return None
    parts = key_path.split(".")
    current = obj
    for part in parts:
        if isinstance(current, dict):
            current = current.get(part)
        elif isinstance(current, list) and len(current) > 0:
            current = current[0].get(part) if isinstance(current[0], dict) else None
        else:
            return None
        if current is None:
            return None
    return current


def set_nested(obj: dict, key_path: str, value: Any) -> None:
    """Set value at dot-notation key path in nested dict (creates intermediate dicts)."""
    parts = key_path.split(".")
    current = obj
    for part in parts[:-1]:
        if part not in current or not isinstance(current[part], dict):
            current[part] = {}
        current = current[part]
    current[parts[-1]] = value


def flatten_json(obj: dict, prefix: str = "") -> dict[str, Any]:
    """Flatten a nested dict to {dot.key.path: value} mapping."""
    result = {}
    if isinstance(obj, dict):
        for key, value in obj.items():
            full_key = f"{prefix}.{key}" if prefix else key
            if isinstance(value, dict):
                result.update(flatten_json(value, full_key))
            elif isinstance(value, list):
                for i, item in enumerate(value):
                    if isinstance(item, dict):
                        result.update(flatten_json(item, f"{full_key}[{i}]"))
                    else:
                        result[f"{full_key}[{i}]"] = item
            else:
                result[full_key] = value
    return result


def max_depth(schema: dict, depth: int = 0) -> int:
    """Return maximum nesting depth of a schema dict."""
    if not isinstance(schema, dict) or not schema:
        return depth
    return max(
        max_depth(v, depth + 1) if isinstance(v, dict) else depth + 1
        for v in schema.values()
    )
