"""LLM-based eval example generation pipeline.

Sampling strategy
-----------------
For each split we target ``n_claims * 4`` examples (the "4x" target).  We
sample (claim_subset, operator_list) pairs from available :class:`ClaimGroup`
objects, weighted by group size, until the target is reached.

Operator count distribution (stratified):
  - 1 operator  ~ 60 % of examples
  - 2 operators ~ 30 %
  - 3 operators ~ 10 %

For each sampled pair we call a vLLM-hosted model via the OpenAI-compatible
API with JSON-mode structured output.  LLM outputs are cached on disk so
reruns are free.
"""

from __future__ import annotations

import json
import random
from typing import Literal

import openai
from pydantic import BaseModel, Field

from amfv_eval._cache import GenerationCache
from amfv_eval.operators import NON_RELATION_OPERATORS, OPERATORS, RELATION_OPERATORS
from amfv_eval.types import ClaimGroup, EvalExample, ScifactClaim

__all__ = ["generate_split", "generate_example"]

_OP_COUNT_WEIGHTS = {1: 0.60, 2: 0.30, 3: 0.10}

_SYSTEM_PROMPT = """\
You are an expert at constructing scientific claim-decomposition training examples.

You will receive a set of atomic scientific claims (seed claims) and a list of \
linguistic operators that describe how to fuse them into a single complex passage. \
Your job is to:

1. Write a PASSAGE that embeds all seed claims using each specified operator, in a \
way that reads naturally as scientific prose.

2. Return GOLD_ATOMS — the complete, minimal set of atomic claims that a perfect \
decomposer must extract from the passage:
   - Include a contextualized (concrete, unambiguous) version of every seed claim.
   - For each RELATION operator applied (causal, contrastive, temporal), add exactly \
one additional relational atom capturing the new logical link introduced.
   - Do NOT add atoms that weren't implied by the seed claims or required by a \
RELATION operator.

3. Return SEED_ATOMS — the contextualized versions of the original seed claims \
(subset of gold_atoms, listed in the same order as the input seeds).

Rules:
- Every entity must be named explicitly. Never write "the drug", "the treatment", \
"it" etc. without first establishing the referent—use the specific name.
- Keep the passage to 1–3 sentences.
- Each atom must be a standalone, grammatically complete declarative sentence.
- Do not hallucinate facts not present in the seed claims.
- Respond with valid JSON matching this schema exactly:
  {
    "passage": "<fused passage>",
    "seed_atoms": ["<atom 1>", ...],
    "relational_atoms": ["<atom>"]  // empty list if no RELATION operators
  }
"""


class _LLMOutput(BaseModel):
    """Structured output schema for a single generation call."""

    passage: str = Field(description="The fused passage embedding all seed claims.")
    seed_atoms: list[str] = Field(
        description=(
            "Contextualized versions of the input seed claims, in the same order "
            "as the input seeds."
        )
    )
    relational_atoms: list[str] = Field(
        default_factory=list,
        description=(
            "One additional atom per RELATION operator applied, capturing the new "
            "logical link.  Empty list when no RELATION operators are used."
        ),
    )


def _build_user_message(claims: list[ScifactClaim], operator_names: list[str]) -> str:
    """Format the per-request user message."""
    claim_lines = "\n".join(f"  {i + 1}. {c.claim}" for i, c in enumerate(claims))
    op_lines = "\n".join(
        f"  - {name}: {OPERATORS[name].description}"
        for name in operator_names
    )
    return (
        f"Seed claims:\n{claim_lines}\n\n"
        f"Operators to apply:\n{op_lines}\n\n"
        "Produce the passage and gold atoms as JSON."
    )


def generate_example(
    claims: list[ScifactClaim],
    operator_names: list[str],
    *,
    client: openai.OpenAI,
    model: str,
    cache: GenerationCache,
    split: Literal["train", "validation", "test"],
    example_index: int,
) -> EvalExample | None:
    """Generate a single eval example, using the disk cache if available.

    Args:
        claims: Seed claims to fuse (2 or 3).
        operator_names: Ordered list of operator names to apply.
        client: OpenAI-compatible client pointed at a vLLM server.
        model: Model name as registered in the vLLM server.
        cache: Disk cache instance.
        split: Dataset split.
        example_index: Monotonic counter used to build the example ID.

    Returns:
        A fully populated :class:`EvalExample`, or ``None`` if generation
        fails (e.g. the model returns invalid JSON).
    """
    cache_key = cache.key(
        model,
        json.dumps([c.id for c in claims], sort_keys=True),
        json.dumps(operator_names),
    )

    cached = cache.get(cache_key)
    if cached is None:
        user_msg = _build_user_message(claims, operator_names)
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                response_format={"type": "json_object"},
                max_tokens=2048,
            )
            content = response.choices[0].message.content or ""
            output = _LLMOutput.model_validate_json(content)
        except Exception:
            return None

        cached = {
            "passage": output.passage,
            "seed_atoms": output.seed_atoms,
            "relational_atoms": output.relational_atoms,
        }
        cache.set(cache_key, cached)

    gold_atoms: list[str] = list(cached["seed_atoms"]) + list(cached.get("relational_atoms", []))

    return EvalExample(
        id=f"{split}_{example_index:05d}",
        split=split,
        passage=cached["passage"],
        operators=operator_names,
        gold_atoms=gold_atoms,
        source_claims=list(claims),
    )


def generate_split(
    groups: list[ClaimGroup],
    n_claims: int,
    split: Literal["train", "validation", "test"],
    *,
    client: openai.OpenAI,
    model: str,
    cache: GenerationCache,
    seed: int = 42,
) -> list[EvalExample]:
    """Generate eval examples for one split.

    Args:
        groups: Connected-component groups (>= 2 claims each).
        n_claims: Total number of claims in the split (used to compute
            the 4x target).
        split: Dataset split label.
        client: OpenAI-compatible client pointed at a vLLM server.
        model: Model name as registered in the vLLM server.
        cache: Disk cache instance.
        seed: Random seed for reproducibility.

    Returns:
        List of generated :class:`EvalExample` objects.
    """
    target = n_claims * 4
    rng = random.Random(seed)

    weights = [len(g.claims) for g in groups]
    if sum(weights) == 0:
        return []

    examples: list[EvalExample] = []
    seen: set[str] = set()
    attempt_cap = target * 20
    attempts = 0
    example_index = 0

    while len(examples) < target and attempts < attempt_cap:
        attempts += 1

        group = rng.choices(groups, weights=weights, k=1)[0]

        n_ops = rng.choices(
            list(_OP_COUNT_WEIGHTS.keys()),
            weights=list(_OP_COUNT_WEIGHTS.values()),
            k=1,
        )[0]

        n_claims_needed = min(3, max(2, n_ops))
        if len(group.claims) < n_claims_needed:
            n_claims_needed = min(len(group.claims), 2)

        sampled_claims = list(rng.sample(list(group.claims), n_claims_needed))
        op_names = _sample_operators(rng, n_ops)

        dedup_key = json.dumps(
            [sorted(c.id for c in sampled_claims), sorted(op_names)], sort_keys=True
        )
        if dedup_key in seen:
            continue
        seen.add(dedup_key)

        example = generate_example(
            sampled_claims,
            op_names,
            client=client,
            model=model,
            cache=cache,
            split=split,
            example_index=example_index,
        )
        if example is not None:
            examples.append(example)
            example_index += 1

    return examples


def _sample_operators(rng: random.Random, n_ops: int) -> list[str]:
    """Sample ``n_ops`` distinct operators, ensuring variety.

    For n_ops == 1: pick any operator uniformly.
    For n_ops >= 2: pick at most one RELATION operator to keep things coherent.
    """
    all_ops = list(OPERATORS.keys())

    if n_ops == 1:
        return [rng.choice(all_ops)]

    n_relation = rng.choices([0, 1], weights=[0.4, 0.6], k=1)[0]
    n_non_relation = n_ops - n_relation

    chosen: list[str] = []
    if n_relation:
        chosen.append(rng.choice(RELATION_OPERATORS))
    if n_non_relation:
        available = [op for op in NON_RELATION_OPERATORS if op not in chosen]
        chosen.extend(rng.sample(available, min(n_non_relation, len(available))))

    rng.shuffle(chosen)
    return chosen
