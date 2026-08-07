"""Shared utilities for LLM response parsing."""

from __future__ import annotations

import re

__all__ = ["extract_json", "split_thinking"]


def split_thinking(text: str) -> tuple[str, str]:
    """Return ``(thinking, response)`` split from a raw completion.

    Handles both ``<think>…</think>`` embedded blocks and vllm's
    ``reasoning_content`` field (pass the field value directly as ``text``).
    The response is the content with all think blocks stripped.
    """
    thinking_parts = re.findall(r"<think>(.*?)</think>", text, re.DOTALL)
    response = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    return "\n".join(thinking_parts), response


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