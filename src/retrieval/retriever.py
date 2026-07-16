"""Fonctions de retrieval dense, réécrit et hybride."""

from __future__ import annotations

from src.generation.query_rewriter import rewrite_query_for_retrieval
from src.retrieval.bm25_index import search_bm25
from src.retrieval.hybrid_fusion import reciprocal_rank_fusion
from src.retrieval.reranker import rerank_results
from src.vectorstore.chroma_store import load_vector_store

# RRF utilise traditionnellement une constante autour de 60. Elle atténue les
# écarts entre les tout premiers rangs et rend la fusion robuste aux listes plus
# longues. Les poids restent égaux afin de mesurer l'effet d'une hybridation
# neutre avant toute optimisation supplémentaire.
RRF_K = 60
DENSE_WEIGHT = 1.0
BM25_WEIGHT = 1.0


def is_broad_question(query: str) -> bool:
    """Détecte les questions qui nécessitent un éventail plus large de chunks."""

    broad_terms = [
        "caractéristiques",
        "spécifications",
        "performances",
        "résume",
        "résumer",
        "liste",
        "différentes",
        "principales",
        "toutes",
        "vue d'ensemble",
        "synthèse",
    ]

    query_lower = query.casefold()
    return any(term in query_lower for term in broad_terms)


def get_search_k(query: str, k: int) -> int:
    """Détermine le nombre de candidats récupérés avant le classement final."""

    if is_broad_question(query):
        return max(30, k * 6)
    return max(12, k * 4)


def _retrieve_dense_candidates(query: str, *, candidate_k: int) -> list[dict]:
    """Récupère et normalise les candidats issus de Chroma/BGE-M3."""

    vector_store = load_vector_store()
    raw_results = vector_store.similarity_search_with_score(query, k=candidate_k)
    reranked_results = rerank_results(query, raw_results)

    return [
        {
            "text": doc.page_content,
            "metadata": doc.metadata,
            "score": distance,  # Distance Chroma : plus petit = meilleur.
            "semantic_score": semantic_score,  # Plus grand = meilleur.
            "dense_rank": rank,
        }
        for rank, (doc, distance, semantic_score) in enumerate(
            reranked_results,
            start=1,
        )
    ]



def _combine_original_and_rewritten_query(
    original_query: str,
    rewritten_query: str,
) -> str:
    """Concatène les requêtes sans dupliquer un rewrite identique au texte initial."""

    cleaned_original = original_query.strip()
    cleaned_rewrite = rewritten_query.strip()

    if not cleaned_rewrite or cleaned_rewrite.casefold() == cleaned_original.casefold():
        return cleaned_original

    return f"{cleaned_original} {cleaned_rewrite}"


def _attach_query_information(
    chunks: list[dict],
    *,
    original_query: str,
    rewritten_query: str | None,
    search_query: str,
    retrieval_mode: str,
    search_k: int,
) -> list[dict]:
    """Ajoute les informations nécessaires au debug et aux rapports JSON."""

    for chunk in chunks:
        chunk["original_query"] = original_query
        chunk["rewritten_query"] = rewritten_query
        chunk["search_query"] = search_query
        chunk["query_used"] = search_query
        chunk["retrieval_mode"] = retrieval_mode
        chunk["search_k"] = search_k

    return chunks


def retrieve_relevant_chunks(query: str, k: int = 3) -> list[dict]:
    """Recherche dense BGE-M3, utilisée comme baseline historique."""

    search_k = get_search_k(query=query, k=k)
    chunks = _retrieve_dense_candidates(query, candidate_k=search_k)[:k]

    return _attach_query_information(
        chunks,
        original_query=query,
        rewritten_query=None,
        search_query=query,
        retrieval_mode="dense",
        search_k=search_k,
    )


def retrieve_relevant_chunks_with_rewrite(
    query: str,
    k: int = 3,
) -> list[dict]:
    """Recherche dense sur question originale + reformulation Gemma."""

    rewritten_query = rewrite_query_for_retrieval(query)
    search_query = _combine_original_and_rewritten_query(query, rewritten_query)
    search_k = get_search_k(query=search_query, k=k)
    chunks = _retrieve_dense_candidates(search_query, candidate_k=search_k)[:k]

    return _attach_query_information(
        chunks,
        original_query=query,
        rewritten_query=rewritten_query,
        search_query=search_query,
        retrieval_mode="dense_rewrite",
        search_k=search_k,
    )


def _retrieve_hybrid(
    *,
    original_query: str,
    search_query: str,
    rewritten_query: str | None,
    retrieval_mode: str,
    k: int,
) -> list[dict]:
    """Exécute dense + BM25 puis fusionne leurs rangs avec RRF."""

    search_k = get_search_k(query=search_query, k=k)

    dense_candidates = _retrieve_dense_candidates(
        search_query,
        candidate_k=search_k,
    )
    bm25_candidates = search_bm25(search_query, k=search_k)

    fused_candidates = reciprocal_rank_fusion(
        dense_candidates,
        bm25_candidates,
        rrf_k=RRF_K,
        dense_weight=DENSE_WEIGHT,
        bm25_weight=BM25_WEIGHT,
    )

    # ``score`` reste présent pour la compatibilité avec l'interface et le
    # script d'évaluation. En mode hybride, il correspond au score RRF et non à
    # une distance Chroma.
    chunks: list[dict] = []
    for fused_rank, candidate in enumerate(fused_candidates[:k], start=1):
        chunks.append(
            {
                **candidate,
                "score": candidate["rrf_score"],
                "fused_rank": fused_rank,
                "rrf_k": RRF_K,
                "dense_weight": DENSE_WEIGHT,
                "bm25_weight": BM25_WEIGHT,
            }
        )

    return _attach_query_information(
        chunks,
        original_query=original_query,
        rewritten_query=rewritten_query,
        search_query=search_query,
        retrieval_mode=retrieval_mode,
        search_k=search_k,
    )


def retrieve_relevant_chunks_hybrid(query: str, k: int = 3) -> list[dict]:
    """Recherche hybride BGE-M3 + BM25 avec fusion RRF."""

    return _retrieve_hybrid(
        original_query=query,
        search_query=query,
        rewritten_query=None,
        retrieval_mode="hybrid",
        k=k,
    )


def retrieve_relevant_chunks_hybrid_with_rewrite(
    query: str,
    k: int = 3,
) -> list[dict]:
    """Recherche hybride sur question originale + reformulation Gemma."""

    rewritten_query = rewrite_query_for_retrieval(query)
    search_query = _combine_original_and_rewritten_query(query, rewritten_query)

    return _retrieve_hybrid(
        original_query=query,
        search_query=search_query,
        rewritten_query=rewritten_query,
        retrieval_mode="hybrid_rewrite",
        k=k,
    )
