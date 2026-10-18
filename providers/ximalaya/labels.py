"""Standalone label fallbacks; MA's translations override these after upstream inclusion."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from music_assistant_models.config_entries import ConfigEntry

STRINGS = json.loads(Path(__file__).with_name("zh_CN.json").read_text())


def config_entry(**kwargs: Any) -> ConfigEntry:
    """Supply display values without editing MA's compiled translation files."""
    entry = ConfigEntry(**kwargs)
    text = STRINGS.get("config_entries", {}).get(entry.key, {})
    if label := text.get("label"):
        entry.label = label.format(*(entry.translation_params or []))
    entry.description = text.get("description", entry.description)
    for option in entry.options or ():
        option.title = text.get("options", {}).get(str(option.value), option.title)
    return entry
