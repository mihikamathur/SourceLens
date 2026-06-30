from src.retrieval.types import Candidate
from src.generation.generate import validate_citations, extract_citations

CANDS = [
    Candidate(chunk_id="a1b2c3d4e5f6", text="Revenue grew 12% in Q3.", source="report.pdf", page=2, score=1.0),
    Candidate(chunk_id="111122223333", text="Headcount increased to 50.", source="report.pdf", page=3, score=0.9),
]


def test_extract_citations_finds_all_ids():
    answer = "Revenue grew 12% [a1b2c3d4e5f6]. Headcount rose too [111122223333]."
    assert extract_citations(answer) == {"a1b2c3d4e5f6", "111122223333"}


def test_fully_grounded_answer_passes():
    answer = "Revenue grew 12% in Q3 [a1b2c3d4e5f6]. Headcount increased to 50 [111122223333]."
    report = validate_citations(answer, CANDS)
    assert report.is_grounded is True
    assert report.invalid_ids == set()
    assert report.uncited_sentences == []


def test_hallucinated_citation_id_is_flagged_invalid():
    answer = "Revenue grew 12% in Q3 [ffffffffffff]."
    report = validate_citations(answer, CANDS)
    assert "ffffffffffff" in report.invalid_ids
    assert report.is_grounded is False


def test_uncited_factual_sentence_is_flagged():
    answer = "Revenue grew 12% in Q3. This is a strong result."
    report = validate_citations(answer, CANDS)
    assert len(report.uncited_sentences) == 2
    assert report.is_grounded is False


def test_refusal_sentence_does_not_need_a_citation():
    answer = "I don't know — the context does not contain information about headcount in 2019."
    report = validate_citations(answer, CANDS)
    assert report.uncited_sentences == []
    assert report.is_grounded is True