from langchain_core.documents import Document


def rerank_results(
    query: str,
    results: list[tuple[Document, float]],
) -> list[tuple[Document, float, float]]:
    """
    Scoring neutre des résultats Chroma.

    Chroma renvoie une distance :
    - plus petit = meilleur

    On transforme cette distance en score lisible :
    - plus grand = meilleur

    Aucun bonus, aucune heuristique pour l'instant.
    """

    scored_results = []

    for doc, distance in results:
        semantic_score = 1 / (1 + distance)

        scored_results.append((doc, distance, semantic_score))

    scored_results.sort(key=lambda x: x[2], reverse=True)

    return scored_results