"""Decompose a passage into atomic claims using a vLLM-hosted model."""

from __future__ import annotations

import re

import openai
from pydantic import BaseModel, Field

from amfv_eval._cache import GenerationCache

__all__ = ["decompose"]

_SYSTEM = """\
You are a scientific claim decomposer. Given a passage, think carefully about its \
structure and extract every atomic factual claim it makes.

Respond with valid JSON:
  {"atoms": ["<claim 1>", "<claim 2>", ...]}
"""


class _Output(BaseModel):
    atoms: list[str] = Field(description="All atomic claims in the passage.")


def decompose(
    passage: str,
    *,
    client: openai.OpenAI,
    model: str,
    cache: GenerationCache,
) -> tuple[list[str], str]:
    """Decompose a passage into atomic claims.

    Args:
        passage: The passage to decompose.
        client: OpenAI-compatible client pointed at a vLLM server.
        model: Model name.
        cache: Disk cache.

    Returns:
        Tuple of ``(extracted_atoms, thinking_trace)``.  Returns ``([], "")``
        on failure.
    """
    cache_key = cache.key("decompose", model, passage)
    cached = cache.get(cache_key)
    if cached is not None:
        return cached["atoms"], cached.get("thinking_trace", "")

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _SYSTEM},
                {"role": "user", "content": f"Passage: {passage}"},
            ],
            response_format={"type": "json_object"},
            max_tokens=1024,
        )
        raw = response.choices[0].message.content or ""
        thinking = _extract_thinking(response, raw)
        clean = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
        atoms = _Output.model_validate_json(clean).atoms
    except Exception:
        return [], ""

    cache.set(cache_key, {"atoms": atoms, "thinking_trace": thinking})
    return atoms, thinking


def _extract_thinking(response: openai.types.chat.ChatCompletion, raw: str) -> str:
    """Extract thinking trace from a vLLM response.

    Handles two formats:
    - ``reasoning_content`` field (newer vLLM with thinking-capable models)
    - ``<think>...</think>`` tags embedded in the content
    """
    msg = response.choices[0].message
    rc = getattr(msg, "reasoning_content", None)
    if rc:
        return rc
    m = re.search(r"<think>(.*?)</think>", raw, re.DOTALL)
    return m.group(1).strip() if m else ""
