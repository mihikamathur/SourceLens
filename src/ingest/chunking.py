from __future__ import annotations
from dataclasses import dataclass, field
import hashlib
import re

from src.ingest.loaders import TextRecord

SEPARATORS = ["\n\n", "\n", ". ", " "]  # tried in order, coarsest first


@dataclass
class Chunk:
    chunk_id: str
    text: str
    source: str
    page: int | None
    chunk_index: int            # position within the source document
    metadata: dict = field(default_factory=dict)


def _split_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Greedy recursive split: try each separator until pieces fit chunk_size,
    then pack pieces back together up to chunk_size with `overlap` chars repeated."""

    def split_on_seps(s: str, seps: list[str]) -> list[str]:
        if not seps:
            return [s]
        sep = seps[0]
        parts = [p for p in s.split(sep) if p.strip()]
        if len(parts) <= 1:
            return split_on_seps(s, seps[1:])
        return parts

    pieces = split_on_seps(text, SEPARATORS)

    chunks: list[str] = []
    current = ""
    for piece in pieces:
        candidate = (current + " " + piece).strip() if current else piece
        if len(candidate) <= chunk_size:
            current = candidate
        else:
            if current:
                chunks.append(current)
            # start new chunk, carrying overlap from the tail of the previous one
            tail = current[-overlap:] if overlap and current else ""
            current = (tail + " " + piece).strip()
            # if a single piece is itself bigger than chunk_size, hard-cut it
            while len(current) > chunk_size:
                chunks.append(current[:chunk_size])
                current = current[chunk_size - overlap:]
    if current.strip():
        chunks.append(current)
    return chunks


def _make_chunk_id(source: str, page: int | None, chunk_index: int) -> str:
    raw = f"{source}|{page}|{chunk_index}"
    return hashlib.sha1(raw.encode()).hexdigest()[:12]


def chunk_records(
    records: list[TextRecord], chunk_size: int = 800, overlap: int = 120
) -> list[Chunk]:
    chunks: list[Chunk] = []
    per_source_index: dict[str, int] = {}

    for rec in records:
        normalized = re.sub(r"[ \t]+", " ", rec.text).strip()
        if not normalized:
            continue
        for piece in _split_text(normalized, chunk_size, overlap):
            idx = per_source_index.get(rec.source, 0)
            per_source_index[rec.source] = idx + 1
            chunks.append(
                Chunk(
                    chunk_id=_make_chunk_id(rec.source, rec.page, idx),
                    text=piece,
                    source=rec.source,
                    page=rec.page,
                    chunk_index=idx,
                    metadata={"source": rec.source, "page": rec.page or 0},
                )
            )
    return chunks