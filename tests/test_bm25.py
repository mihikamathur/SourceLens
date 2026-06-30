from rank_bm25 import BM25Okapi


def test_bm25_ranks_exact_term_match_higher():
    corpus = [
        "reciprocal rank fusion combines bm25 and vector search".split(),
        "the weather today is sunny with light wind".split(),
        "cross encoder reranking improves retrieval precision".split(),
    ]
    bm25 = BM25Okapi(corpus)
    scores = bm25.get_scores("vector search fusion".split())
    assert scores[0] == max(scores)   # doc 0 mentions all three query terms 
    