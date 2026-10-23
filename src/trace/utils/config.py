"""Small configuration helpers for simple YAML-like files."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a restricted YAML subset used by Milestone 1 configs."""
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    root: dict[str, Any] = {}
    stack: list[tuple[int, Any]] = [(-1, root)]
    last_key_at_indent: dict[int, str] = {}

    for raw in lines:
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        indent = len(raw) - len(raw.lstrip(" "))
        text = raw.strip()
        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        if text.startswith("- "):
            value = _parse_scalar(text[2:])
            if not isinstance(parent, list):
                key = last_key_at_indent.get(stack[-1][0])
                if key is None:
                    continue
                new_list: list[Any] = []
                stack[-1][1][key] = new_list
                stack.append((indent, new_list))
                parent = new_list
            parent.append(value)
            continue
        key, _, value_text = text.partition(":")
        key = key.strip()
        value_text = value_text.strip()
        if value_text:
            parent[key] = _parse_scalar(value_text)
        else:
            parent[key] = {}
            stack.append((indent, parent[key]))
        last_key_at_indent[indent] = key
    return root


def _parse_scalar(value: str) -> Any:
    if value in {"true", "false"}:
        return value == "true"
    if value.startswith("[") and value.endswith("]"):
        try:
            return ast.literal_eval(value)
        except (ValueError, SyntaxError):
            inner = value[1:-1].strip()
            if not inner:
                return []
            return [part.strip().strip("\"'") for part in inner.split(",")]
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value
