"""Decompose a passage into atomic claims using a vLLM-hosted model."""

from __future__ import annotations

import re

import openai
from pydantic import BaseModel, Field

from amfv_eval._cache import GenerationCache
from amfv_eval._utils import extract_json

__all__ = ["decompose"]

_SYSTEM = """\
You are a scientific claim decomposer. Given a passage, extract every atomic factual \
claim it makes — statements that are independently verifiable and make a concrete \
assertion about the world.

Do NOT extract:
- Background or framing sentences (e.g. "X has been extensively studied.")
- Methodological context (e.g. "This study used a murine model.")
- Statements that describe what was done without asserting a verifiable fact

Each atom must be a standalone, grammatically complete declarative sentence that \
could in principle be verified true or false against evidence.

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
    enable_thinking: bool = False,
    debug: bool = False,
) -> tuple[list[str], str]:
    """Decompose a passage into atomic claims.

    Args:
        passage: The passage to decompose.
        client: OpenAI-compatible client pointed at a vLLM server.
        model: Model name.
        cache: Disk cache.
        enable_thinking: Pass ``enable_thinking=True`` in the request body for
            vLLM models that require it to produce a reasoning trace.

    Returns:
        Tuple of ``(extracted_atoms, thinking_trace)``.  Returns ``([], "")``
        on failure.
    """
    cache_key = cache.key("decompose", model, passage, str(enable_thinking))
    cached = cache.get(cache_key)
    if cached is not None:
        return cached["atoms"], cached.get("thinking_trace", "")

    extra = {"chat_template_kwargs": {"enable_thinking": True}} if enable_thinking else {}
    max_tokens = 8192 if enable_thinking else 1024
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _SYSTEM},
                {"role": "user", "content": f"Passage: {passage}"},
            ],
            max_tokens=max_tokens,
            extra_body=extra,
        )
        raw = response.choices[0].message.content or ""
        if debug:
            print(f"[decompose] completion returned:\n{raw}", flush=True)
        thinking = _extract_thinking(response, raw)
        atoms = _Output.model_validate_json(extract_json(raw)).atoms
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
    rc = getattr(msg, "reasoning", None)
    if rc:
        return rc
    m = re.search(r"<think>(.*?)</think>", raw, re.DOTALL)
    return m.group(1).strip() if m else ""
