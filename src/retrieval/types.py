from dataclasses import dataclass


@dataclass
class Candidate:
    chunk_id: str
    text: str
    source: str
    page: int | None
    score: float                # retriever-specific score; not comparable across retrievers


@dataclass
class ImageCandidate:
    image_id: str
    path: str
    source: str
    page: int | None
    score: float
