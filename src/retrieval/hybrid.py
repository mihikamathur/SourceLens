from __future__ import annotations
from src.config import settings
from src.retrieval.types import Candidate
from src.retrieval.bm25_retriever import bm25_retrieve
from src.retrieval.image_retriever import image_retrieve, ImageCandidate
from src.retrieval.vector_retriever import vector_retrieve


def reciprocal_rank_fusion(
    ranked_lists: list[list[str]], k: int = 60
) -> list[tuple[str, float]]:
    """ranked_lists: each is a list of doc IDs, already ordered best-first.
    Returns [(doc_id, fused_score), ...] sorted best-first."""
    scores: dict[str, float] = {}
    for ranked_list in ranked_lists:
        for rank, doc_id in enumerate(ranked_list, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)


def hybrid_retrieve(
    query: str,
    top_k_bm25: int = settings.top_k_bm25,
    top_k_vector: int = settings.top_k_vector,
    top_n_fused: int = 15,
    rrf_k: int = 60,
) -> list[Candidate]:
    bm25_candidates = bm25_retrieve(query, top_k_bm25)
    vector_candidates = vector_retrieve(query, top_k_vector)

    bm25_ids = [c.chunk_id for c in bm25_candidates]
    vector_ids = [c.chunk_id for c in vector_candidates]

    fused = reciprocal_rank_fusion([bm25_ids, vector_ids], k=rrf_k)

    # lookup table so we can re-attach text/metadata after fusing on IDs alone
    by_id = {c.chunk_id: c for c in (bm25_candidates + vector_candidates)}

    results = []
    for doc_id, fused_score in fused[:top_n_fused]:
        base = by_id[doc_id]
        results.append(
            Candidate(
                chunk_id=base.chunk_id, text=base.text,
                source=base.source, page=base.page,
                score=fused_score,
            )
        )
    return results


def retrieve_all(query: str, top_n_fused: int = 15, top_m_images: int = settings.top_m_images):
    """One call that returns both fused text candidates and image candidates —
    this is what the rerank/generation layers in Day 3 will call."""
    text_candidates = hybrid_retrieve(query, top_n_fused=top_n_fused)
    images: list[ImageCandidate] = image_retrieve(query, top_m=top_m_images)
    return text_candidates, images