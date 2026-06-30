from __future__ import annotations
from functools import lru_cache

from src.config import settings
from src.retrieval.types import Candidate


@lru_cache(maxsize=1)
def get_cross_encoder():
    from sentence_transformers import CrossEncoder
    return CrossEncoder(settings.cross_encoder_model)


def rerank(
    query: str,
    candidates: list[Candidate],
    top_k: int = settings.top_k_reranked,
    model=None,
) -> list[Candidate]:
    if not candidates:
        return []

    model = model or get_cross_encoder()
    pairs = [(query, c.text) for c in candidates]
    scores = model.predict(pairs)

    rescored = [
        Candidate(chunk_id=c.chunk_id, text=c.text, source=c.source, page=c.page, score=float(s))
        for c, s in zip(candidates, scores)
    ]
    rescored.sort(key=lambda c: c.score, reverse=True)
    return rescored[:top_k]