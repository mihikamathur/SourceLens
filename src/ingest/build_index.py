from __future__ import annotations
import argparse
import json
import pickle
from pathlib import Path

import chromadb
from rank_bm25 import BM25Okapi

from src.config import settings, processed_dir
from src.ingest.loaders import load_directory
from src.ingest.chunking import chunk_records, Chunk
from src.ingest.embed import embed_texts, embed_images


def _tokenize(text: str) -> list[str]:
    return text.lower().split()


def build_text_index(chunks: list[Chunk]) -> None:
    client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
    collection = client.get_or_create_collection("text_chunks")

    vectors = embed_texts([c.text for c in chunks])
    collection.upsert(
        ids=[c.chunk_id for c in chunks],
        embeddings=vectors.tolist(),
        documents=[c.text for c in chunks],
        metadatas=[c.metadata for c in chunks],
    )

    # BM25 needs the raw corpus on hand at query time too — persist it alongside the index
    tokenized_corpus = [_tokenize(c.text) for c in chunks]
    bm25 = BM25Okapi(tokenized_corpus)
    bm25_path = processed_dir() / "bm25_index.pkl"
    with open(bm25_path, "wb") as f:
        pickle.dump(
            {
                "bm25": bm25,
                "chunk_ids": [c.chunk_id for c in chunks],
                "documents": [c.text for c in chunks],
                "metadatas": [c.metadata for c in chunks],
            },
            f,
        )
    print(f"[text] indexed {len(chunks)} chunks -> Chroma + {bm25_path}")


def build_image_index(image_records) -> None:
    if not image_records:
        print("[image] no images found, skipping")
        return

    images_dir = processed_dir() / "images"
    image_ids, image_paths, metadatas, image_bytes_list = [], [], [], []

    for idx, rec in enumerate(image_records):
        image_id = f"img_{idx:05d}"
        out_path = images_dir / f"{image_id}.{rec.ext}"
        out_path.write_bytes(rec.image_bytes)

        image_ids.append(image_id)
        image_paths.append(str(out_path))
        image_bytes_list.append(rec.image_bytes)
        metadatas.append(
            {"source": rec.source, "page": rec.page, "path": str(out_path)}
        )

    vectors = embed_images(image_bytes_list)
    client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
    collection = client.get_or_create_collection("images")
    collection.upsert(
        ids=image_ids,
        embeddings=vectors.tolist(),
        metadatas=metadatas,
    )
    print(f"[image] indexed {len(image_ids)} images -> Chroma + {images_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Directory of raw docs")
    parser.add_argument("--chunk-size", type=int, default=settings.chunk_size)
    parser.add_argument("--overlap", type=int, default=settings.chunk_overlap)
    args = parser.parse_args()

    text_records, image_records = load_directory(Path(args.input))
    print(f"loaded {len(text_records)} text records, {len(image_records)} images")

    chunks = chunk_records(text_records, args.chunk_size, args.overlap)
    print(f"chunked into {len(chunks)} chunks")

    # also dump chunks.jsonl for inspection / eval-set authoring later
    chunks_path = processed_dir() / "chunks.jsonl"
    with open(chunks_path, "w") as f:
        for c in chunks:
            f.write(json.dumps({
                "chunk_id": c.chunk_id, "text": c.text,
                "source": c.source, "page": c.page,
            }) + "\n")

    build_text_index(chunks)
    build_image_index(image_records)
    print("done.")


if __name__ == "__main__":
    main()