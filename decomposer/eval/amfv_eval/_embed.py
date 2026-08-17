"""Embed SciFact claims and identify semantically similar pairs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

__all__ = ["embed_and_find_similar"]


def embed_and_find_similar(
    splits: list[Literal["train", "validation", "test"]],
    *,
    model_name: str,
    threshold: float,
    out: Path,
    data_dir: Path | None = None,
    print_histogram: bool = False,
) -> None:
    """Embed all claims across splits and save similar pairs to *out*.

    Output JSON schema::

        {
          "model": "<model_name>",
          "threshold": 0.85,
          "n_claims": 1109,
          "histogram": [{"bin": [-1.0, -0.9], "count": 0}, ...],
          "similar_pairs": {"<claim_id>": [<similar_id>, ...], ...}
        }

    The histogram covers the full [-1, 1] cosine similarity range in 20 bins
    of width 0.1. It is always saved and optionally printed.
    """
    import torch
    import torch.nn.functional as F
    from transformers import AutoModel, AutoTokenizer

    from amfv_eval.scifact import load_claims

    all_claims = []
    for split in splits:
        all_claims.extend(load_claims(split, data_dir=data_dir))

    n = len(all_claims)
    claim_ids = [c.id for c in all_claims]
    texts = [c.claim for c in all_claims]

    print(f"[embed] Loaded {n} claims from splits: {splits}", flush=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[embed] Loading {model_name} on {device}…", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    print(f"[embed] Embedding {n} claims…", flush=True)
    embeddings = _embed_texts(texts, tokenizer, model, device)  # [n, d] normalized

    print("[embed] Computing pairwise cosine similarities…", flush=True)
    sims = (embeddings @ embeddings.T).cpu()  # [n, n]

    # Upper triangle indices (exclude diagonal)
    idx = torch.triu_indices(n, n, offset=1)
    upper = sims[idx[0], idx[1]]

    histogram = _build_histogram(upper)

    if print_histogram:
        _print_histogram(histogram)

    # Threshold to similar pairs
    mask = upper >= threshold
    similar_pairs: dict[str, list[int]] = {}
    pair_i = idx[0][mask].tolist()
    pair_j = idx[1][mask].tolist()
    for i, j in zip(pair_i, pair_j):
        ci, cj = str(claim_ids[i]), str(claim_ids[j])
        similar_pairs.setdefault(ci, []).append(claim_ids[j])
        similar_pairs.setdefault(cj, []).append(claim_ids[i])

    n_pairs = len(pair_i)
    print(f"[embed] Found {n_pairs} similar pairs above threshold {threshold}.", flush=True)

    result = {
        "model": model_name,
        "threshold": threshold,
        "n_claims": n,
        "histogram": histogram,
        "similar_pairs": similar_pairs,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(f"[embed] Saved → {out}", flush=True)


def _embed_texts(texts, tokenizer, model, device: str, batch_size: int = 64):
    import torch
    import torch.nn.functional as F

    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        encoded = tokenizer(
            batch, padding=True, truncation=True, max_length=512, return_tensors="pt"
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}
        with torch.no_grad():
            output = model(**encoded)
        mask = encoded["attention_mask"].unsqueeze(-1).float()
        pooled = (output.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1)
        pooled = F.normalize(pooled, p=2, dim=-1)
        all_embeddings.append(pooled.cpu())
        if (i // batch_size) % 5 == 0:
            print(f"[embed]   {min(i + batch_size, len(texts))}/{len(texts)}", flush=True)
    return torch.cat(all_embeddings, dim=0)


def _build_histogram(upper_triangle) -> list[dict]:
    """20 bins of width 0.1 covering [-1, 1]."""
    bins = [round(-1.0 + i * 0.1, 1) for i in range(21)]
    histogram = []
    for k in range(len(bins) - 1):
        lo, hi = bins[k], bins[k + 1]
        if k < len(bins) - 2:
            count = int(((upper_triangle >= lo) & (upper_triangle < hi)).sum().item())
        else:
            count = int((upper_triangle >= lo).sum().item())
        histogram.append({"bin": [lo, hi], "count": count})
    return histogram


def _print_histogram(histogram: list[dict]) -> None:
    max_count = max(h["count"] for h in histogram) or 1
    bar_width = 40
    print("\nCosine similarity distribution (pairwise upper triangle):")
    print(f"  {'Bin':<14}  {'Count':>8}  Bar")
    print(f"  {'-'*14}  {'-'*8}  {'-'*bar_width}")
    for h in histogram:
        lo, hi = h["bin"]
        count = h["count"]
        bar = "█" * int(count / max_count * bar_width)
        print(f"  [{lo:+.1f}, {hi:+.1f})  {count:>8,}  {bar}")
    print()
