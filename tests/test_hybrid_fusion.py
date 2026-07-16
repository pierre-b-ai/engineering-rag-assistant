from src.retrieval.hybrid_fusion import reciprocal_rank_fusion


def make_chunk(chunk_id: int) -> dict:
    return {
        "text": f"chunk {chunk_id}",
        "metadata": {
            "source": "manual.pdf",
            "page": chunk_id,
            "chunk_id": 0,
        },
    }


def test_rrf_rewards_a_chunk_found_by_both_retrievers() -> None:
    dense = [
        {**make_chunk(1), "score": 0.2, "semantic_score": 0.8},
        {**make_chunk(2), "score": 0.3, "semantic_score": 0.7},
    ]
    bm25 = [
        {**make_chunk(2), "bm25_score": 5.0},
        {**make_chunk(3), "bm25_score": 4.0},
    ]

    fused = reciprocal_rank_fusion(dense, bm25, rrf_k=60)

    assert fused[0]["metadata"]["page"] == 2
    assert fused[0]["dense_rank"] == 2
    assert fused[0]["bm25_rank"] == 1
    assert fused[0]["rrf_score"] > fused[1]["rrf_score"]
