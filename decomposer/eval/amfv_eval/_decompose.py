"""Decompose a passage into atomic claims using a vLLM-hosted model."""

from __future__ import annotations

import json

import openai
from pydantic import BaseModel, Field

from amfv_eval._cache import GenerationCache
from amfv_eval._utils import extract_json, split_thinking

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
    extra = {"chat_template_kwargs": {"enable_thinking": True}} if enable_thinking else {}
    max_tokens = 8192 if enable_thinking else 1024
    messages = [
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": f"Passage: {passage}"},
    ]
    cache_key = cache.key(model, json.dumps(messages), str(enable_thinking))
    cached = cache.get(cache_key)
    if cached is not None:
        return cached["atoms"], cached.get("thinking_trace", "")

    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,  # type: ignore[arg-type]
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
    msg = response.choices[0].message
    rc = getattr(msg, "reasoning_content", None) or getattr(msg, "reasoning", None)
    if rc:
        return rc
    thinking, _ = split_thinking(raw)
    return thinking
