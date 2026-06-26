# Ask My Docs

Production RAG system with hybrid retrieval (BM25 + vector), image retrieval,
cross-encoder reranking, citation enforcement, and a CI-gated evaluation pipeline.

## Status
🚧 Under active build — see `/docs` (or project board) for the 4-day build plan.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # add your ANTHROPIC_API_KEY
python -m src.ingest.build_index --input data/samples
uvicorn src.api.main:app --reload
```
