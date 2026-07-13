"""SciFact dataset loading and claim grouping via connected components."""

from __future__ import annotations

from collections import defaultdict
from typing import Literal

from amfv_eval.types import ClaimGroup, ScifactClaim

__all__ = ["load_claims", "group_claims"]

_SPLIT_MAP = {
    "train": "train",
    "validation": "validation",
    "test": "test",
}

_LABEL_MAP = {
    "SUPPORTS": "SUPPORTED",
    "SUPPORTED": "SUPPORTED",
    "CONTRADICT": "REFUTED",
    "REFUTED": "REFUTED",
    "NOT_ENOUGH_INFO": "NEI",
    "NEI": "NEI",
}


def load_claims(split: Literal["train", "validation", "test"]) -> list[ScifactClaim]:
    """Load SciFact claims for a split from HuggingFace.

    Args:
        split: One of ``train``, ``validation``, or ``test``.

    Returns:
        List of :class:`ScifactClaim` objects.
    """
    try:
        import datasets as hf_datasets
    except ImportError as e:
        raise ImportError(
            "The 'datasets' package is required. Install it with: uv add datasets"
        ) from e

    ds = hf_datasets.load_dataset("allenai/scifact", "claims", split=split, trust_remote_code=True)

    claims: list[ScifactClaim] = []
    for row in ds:
        evidence = row.get("evidence") or {}
        if isinstance(evidence, str):
            import json

            evidence = json.loads(evidence)

        cited_doc_ids: list[int] = [int(d) for d in (row.get("cited_doc_ids") or [])]

        # Derive veracity label from evidence
        label = _infer_label(evidence, row.get("label"))

        claims.append(
            ScifactClaim(
                id=int(row["id"]),
                claim=row["claim"],
                evidence=evidence,
                cited_doc_ids=cited_doc_ids,
                split=split,
                label=label,
            )
        )

    return claims


def group_claims(claims: list[ScifactClaim]) -> list[ClaimGroup]:
    """Partition claims into connected components by shared cited doc IDs.

    Two claims belong to the same group if they share at least one cited
    document, transitively.  Claims with no cited documents form singleton
    groups and are excluded from the returned list (they cannot be fused).

    Args:
        claims: Claims from a single split.

    Returns:
        List of :class:`ClaimGroup` objects with two or more members,
        sorted by descending group size.
    """
    if not claims:
        return []

    buckets = _bucket_by_parent(claims)

    return _buckets_to_groups(buckets)

def _bucket_by_parent(claims: list[ScifactClaim]) -> dict[int, list[ScifactClaim]]:
    """Group claims by their connected component parent ID."""
    claim_ids = [c.id for c in claims]
    parent: dict[int, int] = {cid: cid for cid in claim_ids}

    def find_parent(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]  # path compression
            x = parent[x]
        return x

    def union(x: int, y: int) -> None:
        px, py = find_parent(x), find_parent(y)
        if px != py:
            parent[px] = py

    doc_to_claims: dict[int, list[int]] = defaultdict(list)
    for c in claims:
        for doc_id in c.cited_doc_ids:
            doc_to_claims[doc_id].append(c.id)

    for members in doc_to_claims.values():
        for i in range(1, len(members)):
            union(members[0], members[i])

    buckets: dict[int, list[ScifactClaim]] = defaultdict(list)
    by_id = {c.id: c for c in claims}
    for cid in claim_ids:
        buckets[find_parent(cid)].append(by_id[cid])

    return buckets

def _buckets_to_groups(buckets: dict[int, list[ScifactClaim]]) -> list[ClaimGroup]:
    """Convert claim ID bucketed by parent to ClaimGroup objects."""
    groups: list[ClaimGroup] = []
    for members in buckets.values():
        if len(members) < 2:
            continue
        all_doc_ids: frozenset[int] = frozenset(
            doc_id for c in members for doc_id in c.cited_doc_ids
        )
        split = members[0].split
        groups.append(ClaimGroup(claims=tuple(members), doc_ids=all_doc_ids, split=split))

    groups.sort(key=lambda g: len(g.claims), reverse=True)
    return groups

def _infer_label(
    evidence: dict,
    raw_label: str | None,
) -> Literal["SUPPORTED", "REFUTED", "NEI"]:
    """Derive a single veracity label from evidence or explicit label field."""
    if raw_label:
        mapped = _LABEL_MAP.get(raw_label.upper())
        if mapped:
            return mapped  # type: ignore[return-value]

    if not evidence:
        return "NEI"

    labels: list[str] = []
    for evs in evidence.values():
        if isinstance(evs, list):
            for ev in evs:
                if isinstance(ev, dict) and "label" in ev:
                    labels.append(ev["label"].upper())

    if not labels:
        return "NEI"

    # CONTRADICT dominates; otherwise SUPPORTS
    if any(lb in ("CONTRADICT", "REFUTED") for lb in labels):
        return "REFUTED"
    return "SUPPORTED"
