"""Fusion des classements dense et BM25 avec Reciprocal Rank Fusion (RRF)."""

from __future__ import annotations

import hashlib
from typing import Any


def chunk_identity(chunk: dict[str, Any]) -> str:
    """Construit une clé stable commune aux résultats dense et lexicaux."""

    metadata = chunk.get("metadata", {}) or {}
    identity_parts = [
        metadata.get("file_hash"),
        metadata.get("source"),
        metadata.get("page"),
        metadata.get("chunk_id"),
    ]

    if any(value not in (None, "") for value in identity_parts):
        return "|".join(str(value or "") for value in identity_parts)

    # Repli défensif pour d'anciens chunks sans métadonnées complètes.
    text = str(chunk.get("text", ""))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def reciprocal_rank_fusion(
    dense_results: list[dict],
    bm25_results: list[dict],
    *,
    rrf_k: int = 60,
    dense_weight: float = 1.0,
    bm25_weight: float = 1.0,
) -> list[dict]:
    """Fusionne deux listes classées sans mélanger leurs scores bruts.

    Les distances Chroma et les scores BM25 n'ont pas la même échelle. RRF ne
    considère donc que la position de chaque chunk dans chaque liste :

        poids / (rrf_k + rang)

    Un chunk retrouvé par les deux moteurs cumule les deux contributions.
    """

    if rrf_k < 1:
        raise ValueError("rrf_k doit être supérieur ou égal à 1.")

    merged: dict[str, dict] = {}

    for rank, result in enumerate(dense_results, start=1):
        identity = chunk_identity(result)
        candidate = merged.setdefault(
            identity,
            {
                "text": result.get("text", ""),
                "metadata": result.get("metadata", {}),
                "dense_rank": None,
                "bm25_rank": None,
                "dense_distance": None,
                "semantic_score": None,
                "bm25_score": None,
                "rrf_score": 0.0,
            },
        )
        candidate["dense_rank"] = rank
        candidate["dense_distance"] = result.get("score")
        candidate["semantic_score"] = result.get("semantic_score")
        candidate["rrf_score"] += dense_weight / (rrf_k + rank)

    for rank, result in enumerate(bm25_results, start=1):
        identity = chunk_identity(result)
        candidate = merged.setdefault(
            identity,
            {
                "text": result.get("text", ""),
                "metadata": result.get("metadata", {}),
                "dense_rank": None,
                "bm25_rank": None,
                "dense_distance": None,
                "semantic_score": None,
                "bm25_score": None,
                "rrf_score": 0.0,
            },
        )
        candidate["bm25_rank"] = rank
        candidate["bm25_score"] = result.get("bm25_score")
        candidate["rrf_score"] += bm25_weight / (rrf_k + rank)

    fused_results = list(merged.values())

    # En cas d'égalité, on favorise d'abord les chunks trouvés par les deux
    # moteurs, puis ceux dont le meilleur rang individuel est le plus faible.
    fused_results.sort(
        key=lambda item: (
            item["rrf_score"],
            int(item["dense_rank"] is not None and item["bm25_rank"] is not None),
            -min(
                rank
                for rank in (item["dense_rank"], item["bm25_rank"])
                if rank is not None
            ),
        ),
        reverse=True,
    )

    return fused_results
