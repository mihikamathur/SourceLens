from __future__ import annotations
from functools import lru_cache

from src.config import settings
from src.ingest.embed import embed_image_query
from src.retrieval.types import ImageCandidate


@lru_cache(maxsize=1)
def _get_image_collection():
    import chromadb
    client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
    return client.get_or_create_collection("images")


def image_retrieve(query: str, top_m: int = settings.top_m_images) -> list[ImageCandidate]:
    collection = _get_image_collection()
    if collection.count() == 0:
        return []

    query_vec = embed_image_query(query)
    result = collection.query(query_embeddings=[query_vec.tolist()], n_results=top_m)

    candidates = []
    ids = result["ids"][0]
    metas = result["metadatas"][0]
    distances = result["distances"][0]

    for image_id, meta, dist in zip(ids, metas, distances):
        candidates.append(
            ImageCandidate(
                image_id=image_id,
                path=meta["path"],
                source=meta["source"],
                page=meta.get("page"),
                score=1.0 - dist,
            )
        )
    return candidates
