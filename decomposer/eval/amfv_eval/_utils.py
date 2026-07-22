"""Shared utilities for LLM response parsing."""

from __future__ import annotations

import re

__all__ = ["extract_json"]


def extract_json(text: str) -> str:
    """Extract a JSON object from an LLM response.

    Handles three common output styles:
    - Plain JSON
    - JSON wrapped in markdown code fences (```json ... ``` or ``` ... ```)
    - JSON preceded or followed by prose (finds the outermost ``{...}`` block)

    Also strips ``<think>...</think>`` blocks before searching.
    """
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"```(?:json)?\s*", "", text)
    text = text.strip()
    start = text.find("{")
    end = text.rfind("}") + 1
    if start != -1 and end > start:
        return text[start:end]
    return text