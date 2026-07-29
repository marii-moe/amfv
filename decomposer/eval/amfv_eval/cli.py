"""CLI entry point for decomposer eval-set generation and filtering.

Commands
--------
``amfv-eval generate``
    Generate passages from SciFact, then self-evaluate using the same model.
    Outputs passing examples; optionally writes failures separately.

``amfv-eval filter``
    Load examples from a previous step, run a new judge model on them,
    and output only those that pass.

Typical workflow (swap vLLM model between steps)::

    # Model #1 in vLLM
    amfv-eval generate --model <m1> --out pass1.jsonl --failures fail1.jsonl

    # Swap to model #2
    amfv-eval filter --model <m2> --in pass1.jsonl --out pass2.jsonl --failures fail2.jsonl

    # Swap to model #3
    amfv-eval filter --model <m3> --in pass2.jsonl --out pass3.jsonl --failures fail3.jsonl

    # Repeat until enough examples accumulate in the final output.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Literal

import openai
from tqdm import tqdm

from amfv_eval._cache import GenerationCache
from amfv_eval._decompose import decompose
from amfv_eval._judge import (
    judge_atoms,
    judge_context_extraction,
    judge_operator_awareness,
    judge_operators,
    judge_source_claims,
)
from amfv_eval.generate import generate_split
from amfv_eval.metrics import compute_metrics, print_report
from amfv_eval.scifact import group_claims, load_claims

_DEFAULT_BASE_URL = "http://localhost:8000/v1"


def _normalise_base_url(url: str) -> str:
    """Ensure the base URL has a scheme and ends at /v1, not a full path.

    Handles common mistakes like omitting ``http://`` or appending
    ``/chat/completions``.
    """
    if "://" not in url:
        url = "http://" + url
    # Strip any path suffix beyond /v1
    for suffix in ("/v1/chat/completions", "/v1/completions", "/v1/embeddings"):
        if url.endswith(suffix):
            url = url[: -len(suffix) + 3]  # keep up to "/v1"
            break
    return url


# ---------------------------------------------------------------------------
# Decomposition and judging (run separately so different models can be used)
# ---------------------------------------------------------------------------


def _decompose_examples(
    examples: list[dict],
    *,
    client: openai.OpenAI,
    model: str,
    cache: GenerationCache,
    enable_thinking: bool = False,
    debug: bool = False,
) -> list[dict]:
    """Run decomposition on each passage and store the result.

    Appends a new entry to each record's ``"decompositions"`` list.
    """
    results = []
    for ex in tqdm(examples, desc="decomposing", unit="ex"):
        atoms, trace = decompose(
            ex["passage"], client=client, model=model, cache=cache,
            enable_thinking=enable_thinking, debug=debug,
        )
        record = dict(ex)
        record.setdefault("decompositions", [])
        record["decompositions"].append({
            "model": model,
            "extracted_atoms": atoms,
            "thinking_trace": trace,
        })
        results.append(record)
    return results


def _judge_examples(
    examples: list[dict],
    *,
    client: openai.OpenAI,
    model: str,
    cache: GenerationCache,
) -> list[dict]:
    """Judge the latest decomposition in each record.

    Reads ``record["decompositions"][-1]`` and runs atom coverage, context
    extraction, and operator awareness judges.  Appends a new entry to each
    record's ``"evaluations"`` list and sets the top-level ``"passed"`` flag.
    """
    results = []
    for ex in tqdm(examples, desc="judging", unit="ex"):
        decomps = ex.get("decompositions", [])
        if not decomps:
            print(
                f"[WARNING] record {ex.get('id', '?')} has no decompositions; skipping.",
                file=sys.stderr, flush=True,
            )
            continue
        latest = decomps[-1]
        atoms: list[str] = latest["extracted_atoms"]
        trace: str = latest["thinking_trace"]

        atom_covered = judge_atoms(
            ex["gold_atoms"], atoms, client=client, model=model, cache=cache
        )
        context_extracted = judge_context_extraction(
            ex.get("context_sentences", []), atoms, client=client, model=model, cache=cache
        )
        operators_noticed = judge_operator_awareness(
            ex["operators"], trace, client=client, model=model, cache=cache
        )

        passed = all(atom_covered) and not any(context_extracted)

        n_gold = len(atom_covered)
        recall = sum(atom_covered) / n_gold if n_gold else 1.0
        n_ctx = len(context_extracted)
        ctx_precision = 1.0 - (sum(context_extracted) / n_ctx) if n_ctx else 1.0
        op_awareness = (
            sum(operators_noticed.values()) / len(operators_noticed)
            if operators_noticed
            else 1.0
        )

        evaluation = {
            "decomposition_model": latest["model"],
            "judge_model": model,
            "atom_covered": atom_covered,
            "context_extracted": context_extracted,
            "operators_noticed": operators_noticed,
            "atom_recall": recall,
            "context_precision": ctx_precision,
            "operator_awareness_score": op_awareness,
            "passed": passed,
        }

        record = dict(ex)
        record.setdefault("evaluations", [])
        record["evaluations"].append(evaluation)
        record["passed"] = passed
        results.append(record)

    return results


# ---------------------------------------------------------------------------
# Dataset validation (run once during generation)
# ---------------------------------------------------------------------------


def _validate_dataset_examples(
    examples: list[dict],
    *,
    client: openai.OpenAI,
    model: str,
    cache: GenerationCache,
) -> tuple[list[dict], list[dict]]:
    """Verify each example meets dataset quality requirements.

    Checks that source claims are faithfully represented in the passage and
    gold atoms, and that the passage correctly applies its operators.

    Returns:
        ``(passed, failed)`` — both lists have a ``"dataset_validation"``
        field attached for inspection.
    """
    passed: list[dict] = []
    failed: list[dict] = []
    for ex in tqdm(examples, desc="validating dataset", unit="ex"):
        source_claims = [c["claim"] for c in ex.get("source_claims", [])]

        source_coverage = judge_source_claims(
            source_claims,
            ex["passage"],
            ex["gold_atoms"],
            client=client,
            model=model,
            cache=cache,
        )
        operators_applied = judge_operators(
            ex["operators"],
            ex["passage"],
            client=client,
            model=model,
            cache=cache,
        )

        valid = (
            all(source_coverage["in_passage"])
            and all(source_coverage["in_gold"])
            and all(operators_applied.values())
        )

        record = dict(ex)
        record["dataset_validation"] = {
            "model": model,
            "source_in_passage": source_coverage["in_passage"],
            "source_in_gold": source_coverage["in_gold"],
            "operators_applied": operators_applied,
            "passed": valid,
        }
        (passed if valid else failed).append(record)

    return passed, failed


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def _write_jsonl(records: list[dict], path: Path) -> None:
    with path.open("w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _load_jsonl(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def _ensure_parent(path: Path | None) -> None:
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------


def _cmd_generate(args: argparse.Namespace) -> None:
    _ensure_parent(args.out)
    _ensure_parent(args.failures)

    client = openai.OpenAI(base_url=_normalise_base_url(args.base_url), api_key="EMPTY")
    cache = GenerationCache(cache_dir=args.cache_dir)

    all_pass: list[dict] = []
    all_fail: list[dict] = []

    for split in args.splits:
        split_key: Literal["train", "validation", "test"] = split  # type: ignore[assignment]
        print(f"\n[{split}] Loading SciFact claims…", flush=True)
        claims = load_claims(split_key, data_dir=args.data_dir)
        n_claims = len(claims)
        groups = group_claims(claims)
        target = args.n_examples if args.n_examples is not None else n_claims * 4
        print(
            f"[{split}] {n_claims} claims · {len(groups)} groups ≥2 "
            f"· target {target} examples",
            flush=True,
        )

        print(f"[{split}] Generating passages…", flush=True)
        examples = generate_split(
            groups,
            n_claims=n_claims,
            split=split_key,
            client=client,
            model=args.model,
            cache=cache,
            seed=args.seed,
            target=target,
            debug=args.debug,
        )
        raw_dicts = [ex.to_dict() for ex in examples]
        print(f"[{split}] {len(raw_dicts)} passages generated. Validating…", flush=True)
        split_pass, split_fail = _validate_dataset_examples(
            raw_dicts, client=client, model=args.model, cache=cache
        )
        print(f"[{split}] {len(split_pass)}/{len(raw_dicts)} passed dataset validation.", flush=True)
        all_pass.extend(split_pass)
        all_fail.extend(split_fail)

    _write_jsonl(all_pass, args.out)
    print(f"Passed : {len(all_pass):,} → {args.out}")
    if args.failures:
        _write_jsonl(all_fail, args.failures)
        print(f"Failed : {len(all_fail):,} → {args.failures}")


def _cmd_filter(args: argparse.Namespace) -> None:
    _ensure_parent(args.out)
    _ensure_parent(args.failures)

    examples = _load_jsonl(args.input)
    print(f"Loaded {len(examples):,} examples from {args.input}", flush=True)

    client = openai.OpenAI(base_url=_normalise_base_url(args.base_url), api_key="EMPTY")
    cache = GenerationCache(cache_dir=args.cache_dir)

    print(f"Validating dataset with {args.model}…", flush=True)
    passed, failed = _validate_dataset_examples(
        examples, client=client, model=args.model, cache=cache
    )
    print(f"{len(passed)}/{len(examples)} passed dataset validation.", flush=True)

    _write_jsonl(passed, args.out)
    print(f"Passed : {len(passed):,} → {args.out}")
    if args.failures:
        _write_jsonl(failed, args.failures)
        print(f"Failed : {len(failed):,} → {args.failures}")


def _cmd_decompose(args: argparse.Namespace) -> None:
    _ensure_parent(args.out)

    examples = _load_jsonl(args.input)
    print(f"Loaded {len(examples):,} examples from {args.input}", flush=True)

    client = openai.OpenAI(base_url=_normalise_base_url(args.base_url), api_key="EMPTY")
    cache = GenerationCache(cache_dir=args.cache_dir)

    print(f"Decomposing with {args.model}…", flush=True)
    records = _decompose_examples(
        examples,
        client=client,
        model=args.model,
        cache=cache,
        enable_thinking=args.enable_thinking,
        debug=args.debug,
    )

    _write_jsonl(records, args.out)
    print(f"Decomposed : {len(records):,} → {args.out}")


def _cmd_eval(args: argparse.Namespace) -> None:
    _ensure_parent(args.out)
    _ensure_parent(args.failures)

    examples = _load_jsonl(args.input)
    print(f"Loaded {len(examples):,} examples from {args.input}", flush=True)

    client = openai.OpenAI(base_url=_normalise_base_url(args.base_url), api_key="EMPTY")
    cache = GenerationCache(cache_dir=args.cache_dir)

    print(f"Judging with {args.model}…", flush=True)
    records = _judge_examples(
        examples,
        client=client,
        model=args.model,
        cache=cache,
    )

    passed = [r for r in records if r["passed"]]
    failed = [r for r in records if not r["passed"]]

    print_report(compute_metrics(records), title=f"eval · {args.model}")

    _write_jsonl(passed, args.out)
    print(f"Passed : {len(passed):,} → {args.out}")
    if args.failures:
        _write_jsonl(failed, args.failures)
        print(f"Failed : {len(failed):,} → {args.failures}")


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def _add_common_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--model",
        required=True,
        help="Model name as registered in the vLLM server.",
    )
    p.add_argument(
        "--base-url",
        default=_DEFAULT_BASE_URL,
        help=f"vLLM OpenAI-compatible endpoint (default: {_DEFAULT_BASE_URL}).",
    )
    p.add_argument("--cache-dir", type=Path, default=None, help="LLM output cache directory.")
    p.add_argument("--out", type=Path, required=True, help="Output JSONL for passing examples.")
    p.add_argument(
        "--failures",
        type=Path,
        default=None,
        help="Output JSONL for failing examples (optional).",
    )
    p.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="Print each chat completion response as it is returned.",
    )


def main(argv: list[str] | None = None) -> None:
    """Generate, filter, and evaluate decomposer eval examples."""
    parser = argparse.ArgumentParser(
        prog="amfv-eval",
        description="Decomposer eval-set generation, dataset filtering, and model evaluation.",
    )
    subs = parser.add_subparsers(dest="command", required=True)

    # ------------------------------------------------------------------
    # generate: produce passages and validate dataset quality
    # ------------------------------------------------------------------
    gen = subs.add_parser(
        "generate",
        help="Generate passages from SciFact and validate dataset quality.",
    )
    _add_common_args(gen)
    gen.add_argument("--seed", type=int, default=42, help="Random seed (default: 42).")
    gen.add_argument(
        "--splits",
        nargs="+",
        choices=["train", "validation", "test"],
        default=["train", "validation", "test"],
    )
    gen.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help=(
            "Directory containing SciFact JSONL files (claims_train.jsonl, "
            "claims_dev.jsonl, claims_test_abstract.jsonl). "
            "Omit to download from GitHub (cached in ~/.cache/amfv_eval/scifact/)."
        ),
    )
    gen.add_argument(
        "--n-examples",
        type=int,
        default=None,
        help=(
            "Total number of examples to generate per split.  "
            "Defaults to n_claims \u00d7 4.  Use a small value (e.g. 50) to do a "
            "quick quality check before a full run."
        ),
    )

    # ------------------------------------------------------------------
    # filter: re-validate an existing dataset with a different model
    # ------------------------------------------------------------------
    filt = subs.add_parser(
        "filter",
        help="Re-validate an existing dataset with a different judge model.",
    )
    _add_common_args(filt)
    filt.add_argument(
        "--input", "-i",
        type=Path,
        required=True,
        help="Input JSONL (output from generate or a previous filter step).",
    )

    # ------------------------------------------------------------------
    # decompose: run decomposition on a dataset with the model under test
    # ------------------------------------------------------------------
    dec = subs.add_parser(
        "decompose",
        help="Decompose passages in a dataset using the model under test.",
    )
    _add_common_args(dec)
    dec.add_argument(
        "--input", "-i",
        type=Path,
        required=True,
        help="Input JSONL (output from generate or filter).",
    )
    dec.add_argument(
        "--enable-thinking",
        action="store_true",
        default=False,
        help=(
            "Pass enable_thinking=True to the vLLM server. "
            "Required for models that disable extended thinking by default."
        ),
    )

    # ------------------------------------------------------------------
    # eval: judge decomposition results with a (potentially different) model
    # ------------------------------------------------------------------
    ev = subs.add_parser(
        "eval",
        help="Judge decomposition results produced by the decompose command.",
    )
    _add_common_args(ev)
    ev.add_argument(
        "--input", "-i",
        type=Path,
        required=True,
        help="Input JSONL (output from decompose).",
    )

    args = parser.parse_args(argv)
    {
        "generate": _cmd_generate,
        "filter": _cmd_filter,
        "decompose": _cmd_decompose,
        "eval": _cmd_eval,
    }[args.command](args)


if __name__ == "__main__":
    main(sys.argv[1:])
