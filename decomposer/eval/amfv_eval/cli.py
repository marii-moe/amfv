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
from amfv_eval._judge import judge_atoms, judge_context_extraction, judge_operators
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
# Shared evaluation logic
# ---------------------------------------------------------------------------


def _evaluate_examples(
    examples: list[dict],
    *,
    client: openai.OpenAI,
    model: str,
    cache: GenerationCache,
) -> list[dict]:
    """Decompose each passage and judge atoms + operator awareness.

    Appends a new entry to each record's ``"evaluations"`` list and sets the
    top-level ``"passed"`` flag.

    Args:
        examples: Eval records (may already contain prior evaluations).
        client: OpenAI-compatible client.
        model: Model to use for decomposition and judging.
        cache: Disk cache.

    Returns:
        Same records with a new evaluation appended.
    """
    results = []
    for ex in tqdm(examples, desc="evaluating", unit="ex"):
        atoms, trace = decompose(ex["passage"], client=client, model=model, cache=cache)
        atom_covered = judge_atoms(
            ex["gold_atoms"], atoms, client=client, model=model, cache=cache
        )
        operators_noticed = judge_operators(
            ex["operators"], trace, client=client, model=model, cache=cache
        )
        context_extracted = judge_context_extraction(
            ex.get("context_sentences", []), atoms, client=client, model=model, cache=cache
        )

        passed = (
            all(atom_covered)
            and all(operators_noticed.values())
            and not any(context_extracted)
        )

        n_gold = len(atom_covered)
        recall = sum(atom_covered) / n_gold if n_gold else 1.0
        op_recall = (
            sum(operators_noticed.values()) / len(operators_noticed)
            if operators_noticed
            else 1.0
        )
        n_ctx = len(context_extracted)
        ctx_precision = 1.0 - (sum(context_extracted) / n_ctx) if n_ctx else 1.0

        evaluation = {
            "model": model,
            "extracted_atoms": atoms,
            "thinking_trace": trace,
            "atom_covered": atom_covered,
            "operators_noticed": operators_noticed,
            "context_extracted": context_extracted,
            "atom_recall": recall,
            "operator_recall": op_recall,
            "context_precision": ctx_precision,
            "passed": passed,
        }

        record = dict(ex)
        record.setdefault("evaluations", [])
        record["evaluations"].append(evaluation)
        record["passed"] = passed
        results.append(record)

    return results


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
        )
        print(f"[{split}] {len(examples)} passages generated. Evaluating…", flush=True)

        records = _evaluate_examples(
            [ex.to_dict() for ex in examples],
            client=client,
            model=args.model,
            cache=cache,
        )

        split_pass = [r for r in records if r["passed"]]
        split_fail = [r for r in records if not r["passed"]]
        all_pass.extend(split_pass)
        all_fail.extend(split_fail)

        print_report(
            compute_metrics(records),
            title=f"generate · {split} · {args.model}",
        )

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

    print(f"Evaluating with {args.model}…", flush=True)
    records = _evaluate_examples(examples, client=client, model=args.model, cache=cache)

    passed = [r for r in records if r["passed"]]
    failed = [r for r in records if not r["passed"]]

    print_report(compute_metrics(records), title=f"filter · {args.model}")

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
    p.add_argument("--seed", type=int, default=42, help="Random seed (default: 42).")


def main(argv: list[str] | None = None) -> None:
    """Generate and iteratively filter decomposer eval examples."""
    parser = argparse.ArgumentParser(
        prog="amfv-eval",
        description="Decomposer eval-set generation and multi-model filtering.",
    )
    subs = parser.add_subparsers(dest="command", required=True)

    gen = subs.add_parser("generate", help="Generate passages and self-evaluate.")
    _add_common_args(gen)
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

    filt = subs.add_parser("filter", help="Filter examples using a second judge model.")
    _add_common_args(filt)
    filt.add_argument(
        "--input",
        "-i",
        type=Path,
        required=True,
        help="Input JSONL (output from a previous generate or filter step).",
    )

    args = parser.parse_args(argv)
    {"generate": _cmd_generate, "filter": _cmd_filter}[args.command](args)


if __name__ == "__main__":
    main(sys.argv[1:])
