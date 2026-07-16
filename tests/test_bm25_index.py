from src.retrieval.bm25_index import BM25Index


def test_bm25_prioritises_exact_model_reference() -> None:
    chunks = [
        {
            "id": "generic",
            "text": "Pompe de piscine avec filtre à sable et entretien courant.",
            "metadata": {"source": "generic.pdf", "page": 1, "chunk_id": 0},
        },
        {
            "id": "target",
            "text": "Le modèle SF70220-2 utilise 35 kg de sable de silice.",
            "metadata": {"source": "pump.pdf", "page": 9, "chunk_id": 0},
        },
    ]

    index = BM25Index(chunks)
    results = index.search("quantité média filtrant SF70220-2", k=2)

    assert results
    assert results[0].chunk["id"] == "target"
    assert results[0].score > 0
