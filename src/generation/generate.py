from __future__ import annotations
from dataclasses import dataclass
import re
import anthropic

from src.config import settings
from src.retrieval.types import Candidate
from src.generation.prompt import SYSTEM_PROMPT, build_user_prompt

CITATION_PATTERN = re.compile(r"\[([a-f0-9]{12})\]")


@dataclass
class CitationReport:
    cited_ids: set[str]
    valid_ids: set[str]            # cited AND present among the supplied candidates
    invalid_ids: set[str]          # cited but NOT among supplied candidates (hallucinated ID)
    uncited_sentences: list[str]   # sentences with no citation at all
    is_grounded: bool              # True iff no invalid_ids and no uncited factual sentences


@dataclass
class AskResult:
    answer: str
    citations: list[Candidate]      # the candidates that were actually cited, in cite order
    report: CitationReport


def extract_citations(answer: str) -> set[str]:
    return set(CITATION_PATTERN.findall(answer))


def _split_sentences(text: str) -> list[str]:
    # good enough for citation auditing; not a general-purpose sentence splitter
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


REFUSAL_MARKERS = ("don't know", "do not know", "doesn't contain", "not enough information",
                    "cannot answer", "no information")


def validate_citations(answer: str, candidates: list[Candidate]) -> CitationReport:
    valid_chunk_ids = {c.chunk_id for c in candidates}
    cited_ids = extract_citations(answer)

    valid_ids = cited_ids & valid_chunk_ids
    invalid_ids = cited_ids - valid_chunk_ids

    uncited_sentences = []
    for sentence in _split_sentences(answer):
        is_refusal = any(marker in sentence.lower() for marker in REFUSAL_MARKERS)
        has_citation = bool(CITATION_PATTERN.search(sentence))
        if not has_citation and not is_refusal:
            uncited_sentences.append(sentence)

    is_grounded = (len(invalid_ids) == 0) and (len(uncited_sentences) == 0)

    return CitationReport(
        cited_ids=cited_ids, valid_ids=valid_ids, invalid_ids=invalid_ids,
        uncited_sentences=uncited_sentences, is_grounded=is_grounded,
    )


def generate_answer(query: str, candidates: list[Candidate], client: anthropic.Anthropic | None = None) -> AskResult:
    if not candidates:
        answer = "I don't know — no relevant context was retrieved for this question."
        return AskResult(answer=answer, citations=[], report=validate_citations(answer, candidates))

    client = client or anthropic.Anthropic(api_key=settings.anthropic_api_key)
    user_prompt = build_user_prompt(query, candidates)

    response = client.messages.create(
        model=settings.claude_model,
        max_tokens=800,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_prompt}],
    )
    answer_text = "".join(block.text for block in response.content if block.type == "text")

    report = validate_citations(answer_text, candidates)
    by_id = {c.chunk_id: c for c in candidates}
    cited_candidates = [by_id[cid] for cid in report.valid_ids if cid in by_id]

    return AskResult(answer=answer_text, citations=cited_candidates, report=report)