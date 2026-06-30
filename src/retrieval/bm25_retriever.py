from __future__ import annotations
from functools import lru_cache
import pickle
import numpy as np

from src.config import settings, processed_dir
from src.retrieval.types import Candidate


@lru_cache(maxsize=1)
def _load_bm25_store():
    path = processed_dir() / "bm25_index.pkl"
    with open(path, "rb") as f:
        return pickle.load(f)


def _tokenize(text: str) -> list[str]:
    return text.lower().split()


def bm25_retrieve(query: str, top_k: int = settings.top_k_bm25) -> list[Candidate]:
    store = _load_bm25_store()
    bm25 = store["bm25"]

    scores = bm25.get_scores(_tokenize(query))
    top_indices = np.argsort(scores)[::-1][:top_k]

    return [
        Candidate(
            chunk_id=store["chunk_ids"][i],
            text=store["documents"][i],
            source=store["metadatas"][i]["source"],
            page=store["metadatas"][i].get("page"),
            score=float(scores[i]),
        )
        for i in top_indices
        if scores[i] > 0   # drop zero-relevance matches rather than padding with noise
    ]