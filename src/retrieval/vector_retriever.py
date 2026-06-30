from __future__ import annotations
from functools import lru_cache

from src.config import settings
from src.ingest.embed import embed_query
from src.retrieval.types import Candidate


@lru_cache(maxsize=1)
def _get_collection():
    import chromadb
    client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
    return client.get_or_create_collection("text_chunks")


def vector_retrieve(query: str, top_k: int = settings.top_k_vector) -> list[Candidate]:
    collection = _get_collection()
    query_vec = embed_query(query)

    result = collection.query(
        query_embeddings=[query_vec.tolist()],
        n_results=top_k,
    )

    candidates = []
    ids = result["ids"][0]
    docs = result["documents"][0]
    metas = result["metadatas"][0]
    distances = result["distances"][0]   # cosine distance, lower = more similar

    for chunk_id, text, meta, dist in zip(ids, docs, metas, distances):
        candidates.append(
            Candidate(
                chunk_id=chunk_id,
                text=text,
                source=meta["source"],
                page=meta.get("page"),
                score=1.0 - dist,   # convert distance -> similarity for readability
            )
        )
    return candidates