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

import concurrent.futures
import json
import random
from typing import Literal

import openai
from tqdm import tqdm
from pydantic import BaseModel, Field

from amfv_eval._cache import GenerationCache
from amfv_eval._utils import extract_json
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
way that reads naturally as scientific prose.  You have significant latitude to \
paraphrase: change word order, substitute synonyms, restructure sentences, and vary \
the level of specificity — as long as every factual claim from the seeds is preserved \
exactly.  Do not add, remove, or alter any fact.  Weave in 1–2 short context \
sentences that provide background framing or methodological detail but do NOT \
themselves make a new verifiable factual claim.  These context sentences should read \
as a natural part of the passage.

Example context sentences (illustrative only — do not reuse these verbatim):
  - "Regulatory T cells are central to immune homeostasis."
  - "This study used a murine model of experimental autoimmune encephalomyelitis."

2. Return GOLD_ATOMS — the complete, minimal set of atomic claims that a perfect \
decomposer must extract from the passage:
   - For each seed claim, write a seed atom that captures how the fact appears in the \
passage (after any paraphrasing), not the original wording of the seed.
   - For each RELATION operator applied (causal, contrastive, temporal), add exactly \
one additional relational atom capturing the new logical link introduced.
   - Do NOT include the context sentences — they are not verifiable claims.

3. Return SEED_ATOMS — the paraphrased seed claims as they appear in the passage \
(subset of gold_atoms, listed in the same order as the input seeds).

4. Return CONTEXT_SENTENCES — copy the context sentences verbatim as they appear in \
the passage.  A correct decomposer must NOT extract these as atomic claims.

Rules:

- Keep the passage to 6–8 sentences.
- Each atom must be a standalone, grammatically complete declarative sentence.
- Do not hallucinate facts not present in the seed claims.

Operator reference:

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

- Respond with valid JSON matching this schema exactly:
  {
    "passage": "<fused passage with context woven in>",
    "seed_atoms": ["<atom 1>", ...],
    "relational_atoms": ["<atom>"],
    "context_sentences": ["<non-verifiable sentence 1>", ...]
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
    context_sentences: list[str] = Field(
        default_factory=list,
        description=(
            "Non-verifiable context sentences woven into the passage "
            "(background, methodology, framing).  Copied verbatim from the passage."
        ),
    )


def _token_jaccard(a: str, b: str) -> float:
    ta, tb = set(a.lower().split()), set(b.lower().split())
    union = ta | tb
    return len(ta & tb) / len(union) if union else 0.0


def _any_too_similar(claims: list[ScifactClaim], threshold: float = 0.75) -> bool:
    """Return True if any pair of claims are trivially opposing near-duplicates."""
    for i in range(len(claims)):
        for j in range(i + 1, len(claims)):
            if _token_jaccard(claims[i].claim, claims[j].claim) >= threshold:
                return True
    return False


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
    debug: bool = False,
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
    thinking_trace = ""
    if cached is None:
        user_msg = _build_user_message(claims, operator_names)
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                max_tokens=16384,
            )
            content = response.choices[0].message.content or ""
            # Capture thinking trace before extract_json strips it.
            # vllm may expose it as reasoning_content or embed it as <think>…</think>.
            thinking_trace = getattr(response.choices[0].message, "reasoning_content", None) or ""
            if not thinking_trace:
                m = __import__("re").search(r"<think>(.*?)</think>", content, __import__("re").DOTALL)
                if m:
                    thinking_trace = m.group(1).strip()
            if debug:
                print(f"[generate] completion returned:\n{content}", flush=True)
            output = _LLMOutput.model_validate_json(extract_json(content))
        except Exception:
            return None

        cached = {
            "passage": output.passage,
            "seed_atoms": output.seed_atoms,
            "relational_atoms": output.relational_atoms,
            "context_sentences": output.context_sentences,
        }
        cache.set(cache_key, cached)

    gold_atoms: list[str] = list(cached["seed_atoms"]) + list(cached.get("relational_atoms", []))

    return EvalExample(
        id=f"{split}_{example_index:05d}",
        split=split,
        passage=cached["passage"],
        operators=operator_names,
        gold_atoms=gold_atoms,
        context_sentences=list(cached.get("context_sentences", [])),
        source_claims=list(claims),
        thinking_trace=thinking_trace,
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
    target: int | None = None,
    concurrency: int = 8,
    debug: bool = False,
) -> list[EvalExample]:
    """Generate eval examples for one split.

    Args:
        groups: Connected-component groups (>= 2 claims each).
        n_claims: Total number of claims in the split (used to compute
            the 4x target when ``target`` is not given).
        split: Dataset split label.
        client: OpenAI-compatible client pointed at a vLLM server.
        model: Model name as registered in the vLLM server.
        cache: Disk cache instance.
        seed: Random seed for reproducibility.
        target: Override the number of examples to generate.  Defaults to
            ``n_claims * 4``.
        concurrency: Number of concurrent requests to the vLLM server.

    Returns:
        List of generated :class:`EvalExample` objects.
    """
    target = target if target is not None else n_claims * 4
    rng = random.Random(seed)

    weights = [len(g.claims) for g in groups]
    if sum(weights) == 0:
        return []

    examples: list[EvalExample] = []
    seen: set[str] = set()
    attempt_cap = target * 20
    attempts = 0
    example_index = 0

    def _next_candidate() -> tuple[list[ScifactClaim], list[str]] | None:
        nonlocal attempts
        while attempts < attempt_cap:
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
            if _any_too_similar(sampled_claims):
                continue
            op_names = _sample_operators(rng, n_ops)
            dedup_key = json.dumps(
                [sorted(c.id for c in sampled_claims), sorted(op_names)], sort_keys=True
            )
            if dedup_key in seen:
                continue
            seen.add(dedup_key)
            return sampled_claims, op_names
        return None

    with tqdm(total=target, desc=f"[{split}] generating", unit="ex") as bar:
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
            pending: dict[concurrent.futures.Future, None] = {}

            def _submit_one() -> None:
                cand = _next_candidate()
                if cand is None:
                    return
                sampled_claims, op_names = cand
                f = executor.submit(
                    generate_example,
                    sampled_claims, op_names,
                    client=client, model=model, cache=cache,
                    split=split, example_index=0, debug=debug,
                )
                pending[f] = None

            for _ in range(concurrency):
                if len(examples) + len(pending) >= target:
                    break
                _submit_one()

            while pending and len(examples) < target:
                done, _ = concurrent.futures.wait(
                    list(pending), return_when=concurrent.futures.FIRST_COMPLETED
                )
                for f in done:
                    del pending[f]
                    result = f.result()
                    if result is not None and len(examples) < target:
                        result.id = f"{split}_{example_index:05d}"
                        examples.append(result)
                        example_index += 1
                        bar.update(1)
                    if len(examples) + len(pending) < target:
                        _submit_one()

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
