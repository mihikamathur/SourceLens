from src.retrieval.hybrid import reciprocal_rank_fusion


def test_rrf_rewards_consensus_docs():
    bm25_ranked = ["docA", "docB", "docC"]
    vector_ranked = ["docB", "docA", "docD"]

    fused = reciprocal_rank_fusion([bm25_ranked, vector_ranked], k=60)
    fused_ids = [doc_id for doc_id, _ in fused]

    # docA and docB appear near the top of BOTH lists -> should outrank docC/docD
    assert fused_ids[0] in {"docA", "docB"}
    assert fused_ids[1] in {"docA", "docB"}
    assert "docC" in fused_ids and "docD" in fused_ids
    assert fused_ids.index("docA") < fused_ids.index("docC")
    assert fused_ids.index("docB") < fused_ids.index("docD")


def test_rrf_handles_disjoint_lists():
    fused = reciprocal_rank_fusion([["x", "y"], ["z", "w"]], k=60)
    assert {doc_id for doc_id, _ in fused} == {"x", "y", "z", "w"}


def test_rrf_score_decreases_with_rank():
    fused = reciprocal_rank_fusion([["a", "b", "c"]], k=60)
    scores = [s for _, s in fused]
    assert scores == sorted(scores, reverse=True)