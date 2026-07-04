from src.vectorstore.chroma_store import load_vector_store
from src.retrieval.reranker import rerank_results


def is_broad_question(query: str) -> bool:
    """
    Détecte les questions larges qui demandent une synthèse
    plutôt qu'une information précise.
    """

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

    query_lower = query.lower()

    return any(term in query_lower for term in broad_terms)


def get_search_k(query: str, k: int) -> int:
    """
    Définit combien de chunks récupérer en interne depuis Chroma.

    - Question précise : récupération modérée
    - Question large : récupération plus large
    """

    if is_broad_question(query):
        return max(30, k * 6)

    return max(12, k * 4)


def retrieve_relevant_chunks(query: str, k: int = 3) -> list[dict]:
    """
    Recherche les chunks les plus pertinents pour une question.

    Principe :
    - on récupère plus de résultats que nécessaire dans Chroma ;
    - on applique un scoring/reranking ;
    - on retourne seulement les k meilleurs chunks.
    """

    vector_store = load_vector_store()

    search_k = get_search_k(query=query, k=k)

    raw_results = vector_store.similarity_search_with_score(
        query,
        k=search_k,
    )

    reranked_results = rerank_results(query, raw_results)

    chunks = []

    for doc, distance, semantic_score in reranked_results[:k]:
        chunks.append(
            {
                "text": doc.page_content,
                "metadata": doc.metadata,
                "score": distance,  # distance Chroma : plus petit = meilleur
                "semantic_score": semantic_score,  # score lisible : plus grand = meilleur
                "search_k": search_k,
                "query_used": query,
            }
        )

    return chunks