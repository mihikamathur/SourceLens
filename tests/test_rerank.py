from src.retrieval.types import Candidate
from src.rerank.cross_encoder import rerank


class FakeCrossEncoder:
    """Returns higher scores for passages containing the word 'fusion'."""
    def predict(self, pairs):
        return [2.0 if "fusion" in passage.lower() else 0.1 for _, passage in pairs]


def test_rerank_reorders_by_model_score():
    candidates = [
        Candidate(chunk_id="c1", text="The weather was nice.", source="d", page=1, score=0.5),
        Candidate(chunk_id="c2", text="RRF performs rank fusion of retrievers.", source="d", page=2, score=0.4),
    ]
    reranked = rerank("how does fusion work", candidates, top_k=2, model=FakeCrossEncoder())
    assert reranked[0].chunk_id == "c2"   # fusion-relevant passage promoted to top


def test_rerank_respects_top_k():
    candidates = [
        Candidate(chunk_id=f"c{i}", text="fusion " * i, source="d", page=1, score=0.0)
        for i in range(5)
    ]
    reranked = rerank("fusion", candidates, top_k=2, model=FakeCrossEncoder())
    assert len(reranked) == 2