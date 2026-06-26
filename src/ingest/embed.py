from __future__ import annotations
from functools import lru_cache
from io import BytesIO
import numpy as np

from src.config import settings


@lru_cache(maxsize=1)
def get_text_embedder():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(settings.text_embedding_model)


@lru_cache(maxsize=1)
def get_clip_embedder():
    from sentence_transformers import SentenceTransformer
    # clip-ViT-B-32 can .encode() both PIL Images and raw strings into one shared space
    return SentenceTransformer(settings.image_embedding_model)


def embed_texts(texts: list[str]) -> np.ndarray:
    model = get_text_embedder()
    # bge models recommend a query instruction prefix at *query* time, not index time
    return model.encode(texts, normalize_embeddings=True, show_progress_bar=False)


def embed_query(query: str) -> np.ndarray:
    model = get_text_embedder()
    instructed = f"Represent this sentence for searching relevant passages: {query}"
    return model.encode([instructed], normalize_embeddings=True)[0]


def embed_images(image_bytes_list: list[bytes]) -> np.ndarray:
    from PIL import Image
    model = get_clip_embedder()
    images = [Image.open(BytesIO(b)).convert("RGB") for b in image_bytes_list]
    return model.encode(images, normalize_embeddings=True, show_progress_bar=False)


def embed_image_query(query: str) -> np.ndarray:
    """Embed a plain text query into the CLIP space, to search images by text."""
    model = get_clip_embedder()
    return model.encode([query], normalize_embeddings=True)[0]