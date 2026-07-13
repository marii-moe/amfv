"""CLI entry point for decomposer eval-set generation.

Usage
-----
    amfv-eval --model <model-name> --out data/eval --splits train validation test

The command generates JSONL files (one per split) under the specified output
directory.  LLM calls are cached in ``~/.cache/amfv_eval/`` so reruns only
spend tokens on new examples.

The ``--model`` flag must match the model name registered in the running vLLM
server.  Use ``--base-url`` to point at a non-default vLLM endpoint.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Literal

import openai

from amfv_eval._cache import GenerationCache
from amfv_eval.generate import generate_split
from amfv_eval.scifact import group_claims, load_claims

_DEFAULT_BASE_URL = "http://localhost:8000/v1"


def main(argv: list[str] | None = None) -> None:
    """Generate decomposer eval examples and write them to JSONL files."""
    parser = argparse.ArgumentParser(
        prog="amfv-eval",
        description="Generate decomposer eval-set from SciFact via a vLLM server.",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Model name as registered in the vLLM server (e.g. meta-llama/Llama-3.1-70B-Instruct).",
    )
    parser.add_argument(
        "--base-url",
        default=_DEFAULT_BASE_URL,
        help=f"vLLM OpenAI-compatible endpoint (default: {_DEFAULT_BASE_URL}).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/eval"),
        help="Output directory for JSONL files (default: data/eval).",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=["train", "validation", "test"],
        default=["train", "validation", "test"],
        help="Splits to generate (default: all three).",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Override the LLM output cache directory.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling (default: 42).",
    )
    args = parser.parse_args(argv)

    out_dir: Path = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    client = openai.OpenAI(base_url=args.base_url, api_key="EMPTY")
    cache = GenerationCache(cache_dir=args.cache_dir)

    for split in args.splits:
        split_key: Literal["train", "validation", "test"] = split  # type: ignore[assignment]
        print(f"[{split}] Loading SciFact claims…", flush=True)
        claims = load_claims(split_key)
        n_claims = len(claims)
        print(f"[{split}] {n_claims} claims loaded.", flush=True)

        groups = group_claims(claims)
        print(
            f"[{split}] {len(groups)} groups with ≥2 claims "
            f"(target: {n_claims * 4} examples).",
            flush=True,
        )

        examples = generate_split(
            groups,
            n_claims=n_claims,
            split=split_key,
            client=client,
            model=args.model,
            cache=cache,
            seed=args.seed,
        )
        print(f"[{split}] Generated {len(examples)} examples.", flush=True)

        out_path = out_dir / f"{split}.jsonl"
        with out_path.open("w") as f:
            for ex in examples:
                f.write(json.dumps(ex.to_dict(), ensure_ascii=False) + "\n")
        print(f"[{split}] Written to {out_path}", flush=True)

    print("Done.", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
