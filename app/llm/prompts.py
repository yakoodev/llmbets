"""Load & render prompt templates from /prompts/*.yaml.

Prompts live outside code so they can change without a redeploy (TZ §14).
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

PROMPTS_DIR = Path(__file__).resolve().parents[2] / "prompts"


@lru_cache
def load_prompt(name: str) -> dict[str, Any]:
    path = PROMPTS_DIR / f"{name}.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def render(template: str, **values: Any) -> str:
    out = template
    for key, val in values.items():
        rendered = val if isinstance(val, str) else json.dumps(val, ensure_ascii=False)
        out = out.replace(f"{{{{ {key} }}}}", rendered)
    return out


def clear_cache() -> None:
    """For the /reload_prompts command."""
    load_prompt.cache_clear()
