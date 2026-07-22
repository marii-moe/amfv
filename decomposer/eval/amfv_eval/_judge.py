"""LLM judge for atom coverage and operator-awareness evaluation."""

from __future__ import annotations

import json

import openai
from pydantic import BaseModel

from amfv_eval._cache import GenerationCache
from amfv_eval.operators import get_operator
from amfv_eval._utils import extract_json

__all__ = ["judge_atoms", "judge_operators", "judge_context_extraction"]

_ATOM_SYSTEM = """\
You are evaluating whether a set of extracted atomic claims covers a set of gold \
atomic claims.

For each gold atom, output true if it is semantically covered by any of the extracted \
atoms — even if worded differently — and false otherwise.

Respond with valid JSON:
  {"atom_covered": [<bool>, <bool>, ...]}
One boolean per gold atom, in the same order.
"""

_OPERATOR_SYSTEM = """\
You are evaluating whether a reasoning trace demonstrates awareness of specific \
linguistic relationships present in a scientific passage.

For each operator listed below, output true if the reasoning trace shows the model \
noticed that kind of relationship — it does not need to name the operator explicitly.

Operator reference with examples:

Non-Relation Adding (no new relational fact introduced):
  coreference_fusion    — "Aspirin reduces inflammation, and it also thins the blood."
  apposition            — "BRCA1, a tumor suppressor gene, is located on chromosome 17."
  relative_clause       — "The drug, which targets EGFR, was approved in 2015."
  conjunction_reduction — "The treatment lowered blood pressure and cholesterol."
  ellipsis              — "Group A received the vaccine, and Group B the placebo."

Relation Adding (each introduces one new relational fact):
  causal      — "The tumor shrank because the drug inhibited EGFR."
  contrastive — "Group A improved, whereas Group B declined."
  temporal    — "The patient received the vaccine and developed immunity later."

Respond with valid JSON:
  {"operators_noticed": {"<operator_name>": <bool>, ...}}
"""


_CONTEXT_SYSTEM = """\
You are checking whether a model incorrectly extracted non-verifiable context \
sentences as atomic claims.

Context sentences are background, methodological, or framing statements that a \
correct decomposer should NOT extract — they are not independently verifiable facts.

Examples of context sentences:
  - "Regulatory T cells are central to immune homeostasis."
  - "This study used a murine model of experimental autoimmune encephalomyelitis."
  - "The role of transcription factors in T cell differentiation has been extensively studied."
  - "Participants were recruited from a tertiary care centre between 2010 and 2015."

For each context sentence, output true if it was extracted as an atomic claim \
(i.e., any extracted atom is semantically equivalent to or subsumes it), \
false otherwise.

Respond with valid JSON:
  {"context_extracted": [<bool>, <bool>, ...]}
One boolean per context sentence, in the same order.
"""


class _AtomOutput(BaseModel):
    atom_covered: list[bool]


class _OperatorOutput(BaseModel):
    operators_noticed: dict[str, bool]


class _ContextOutput(BaseModel):
    context_extracted: list[bool]


def judge_atoms(
    gold_atoms: list[str],
    extracted_atoms: list[str],
    *,
    client: openai.OpenAI,
    model: str,
    cache: GenerationCache,
) -> list[bool]:
    """Judge which gold atoms are semantically covered by extracted atoms.

    Args:
        gold_atoms: Reference atoms from the eval example.
        extracted_atoms: Atoms produced by the decomposer model.
        client: OpenAI-compatible client.
        model: Judge model name.
        cache: Disk cache.

    Returns:
        One bool per gold atom; ``True`` means covered.
    """
    cache_key = cache.key(
        "judge_atoms", model,
        json.dumps(gold_atoms),
        json.dumps(extracted_atoms),
    )
    cached = cache.get(cache_key)
    if cached is not None:
        return cached["atom_covered"]

    gold_text = "\n".join(f"{i + 1}. {a}" for i, a in enumerate(gold_atoms))
    extr_text = "\n".join(f"- {a}" for a in extracted_atoms) if extracted_atoms else "(none)"
    user_msg = f"Gold atoms:\n{gold_text}\n\nExtracted atoms:\n{extr_text}"

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _ATOM_SYSTEM},
                {"role": "user", "content": user_msg},
            ],
            max_tokens=256,
        )
        content = response.choices[0].message.content or ""
        covered = _AtomOutput.model_validate_json(extract_json(content)).atom_covered
        # Align to gold length in case the model under/over-counts
        covered = (covered + [False] * len(gold_atoms))[: len(gold_atoms)]
    except Exception:
        covered = [False] * len(gold_atoms)

    cache.set(cache_key, {"atom_covered": covered})
    return covered


def judge_operators(
    operators: list[str],
    thinking_trace: str,
    *,
    client: openai.OpenAI,
    model: str,
    cache: GenerationCache,
) -> dict[str, bool]:
    """Judge whether a thinking trace shows awareness of each operator relationship.

    Args:
        operators: Operator names used in the eval example.
        thinking_trace: The decomposer model's extended reasoning text.
        client: OpenAI-compatible client.
        model: Judge model name.
        cache: Disk cache.

    Returns:
        Mapping from operator name to noticed boolean.
    """
    cache_key = cache.key("judge_operators", model, json.dumps(operators), thinking_trace)
    cached = cache.get(cache_key)
    if cached is not None:
        return cached["operators_noticed"]

    op_desc = "\n".join(
        f"- {name}: {get_operator(name).description}" for name in operators
    )
    user_msg = (
        f"Operators:\n{op_desc}\n\n"
        f"Reasoning trace:\n{thinking_trace or '(no trace available)'}"
    )

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _OPERATOR_SYSTEM},
                {"role": "user", "content": user_msg},
            ],
            max_tokens=256,
        )
        content = response.choices[0].message.content or ""
        noticed = _OperatorOutput.model_validate_json(extract_json(content)).operators_noticed
        for op in operators:
            noticed.setdefault(op, False)
    except Exception:
        noticed = {op: False for op in operators}

    cache.set(cache_key, {"operators_noticed": noticed})
    return noticed


def judge_context_extraction(
    context_sentences: list[str],
    extracted_atoms: list[str],
    *,
    client: openai.OpenAI,
    model: str,
    cache: GenerationCache,
) -> list[bool]:
    """Judge whether context sentences were incorrectly extracted as atomic claims.

    Args:
        context_sentences: Non-verifiable sentences from the passage.
        extracted_atoms: Atoms produced by the decomposer model.
        client: OpenAI-compatible client.
        model: Judge model name.
        cache: Disk cache.

    Returns:
        One bool per context sentence; ``True`` means it was wrongly extracted.
    """
    if not context_sentences:
        return []

    cache_key = cache.key(
        "judge_context", model,
        json.dumps(context_sentences),
        json.dumps(extracted_atoms),
    )
    cached = cache.get(cache_key)
    if cached is not None:
        return cached["context_extracted"]

    ctx_text = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(context_sentences))
    extr_text = "\n".join(f"- {a}" for a in extracted_atoms) if extracted_atoms else "(none)"
    user_msg = f"Context sentences:\n{ctx_text}\n\nExtracted atoms:\n{extr_text}"

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _CONTEXT_SYSTEM},
                {"role": "user", "content": user_msg},
            ],
            max_tokens=128,
        )
        content = response.choices[0].message.content or ""
        extracted = _ContextOutput.model_validate_json(extract_json(content)).context_extracted
        extracted = (extracted + [False] * len(context_sentences))[: len(context_sentences)]
    except Exception:
        extracted = [False] * len(context_sentences)

    cache.set(cache_key, {"context_extracted": extracted})
    return extracted
