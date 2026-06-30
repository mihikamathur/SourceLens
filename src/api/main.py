from __future__ import annotations
from pathlib import Path
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.config import settings, processed_dir
from src.retrieval.hybrid import hybrid_retrieve
from src.retrieval.image_retriever import image_retrieve
from src.rerank.cross_encoder import rerank
from src.generation.generate import generate_answer

app = FastAPI(title="Ask My Docs", version="0.1.0")

images_dir = processed_dir() / "images"
images_dir.mkdir(parents=True, exist_ok=True)
app.mount("/images", StaticFiles(directory=str(images_dir)), name="images")


class AskRequest(BaseModel):
    query: str
    top_n_fused: int = 15
    top_k_reranked: int = settings.top_k_reranked
    top_m_images: int = settings.top_m_images


class CitationOut(BaseModel):
    chunk_id: str
    source: str
    page: int | None
    snippet: str


class ImageOut(BaseModel):
    image_id: str
    source: str
    page: int | None
    url: str


class AskResponse(BaseModel):
    answer: str
    citations: list[CitationOut]
    images: list[ImageOut]
    is_grounded: bool


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    fused = hybrid_retrieve(req.query, top_n_fused=req.top_n_fused)
    top_candidates = rerank(req.query, fused, top_k=req.top_k_reranked)
    images = image_retrieve(req.query, top_m=req.top_m_images)

    result = generate_answer(req.query, top_candidates)

    return AskResponse(
        answer=result.answer,
        citations=[
            CitationOut(chunk_id=c.chunk_id, source=c.source, page=c.page, snippet=c.text[:200])
            for c in result.citations
        ],
        images=[
            ImageOut(image_id=i.image_id, source=i.source, page=i.page,
                      url=f"/images/{Path(i.path).name}")
            for i in images
        ],
        is_grounded=result.report.is_grounded,
    )