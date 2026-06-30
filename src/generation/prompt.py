from __future__ import annotations
from src.retrieval.types import Candidate

SYSTEM_PROMPT = """You are "Ask My Docs", a citation-strict Q&A assistant.

RULES (follow exactly):
1. Answer ONLY using the CONTEXT block below. Never use outside knowledge.
2. After every sentence that states a fact from the context, append the chunk ID
   it came from in square brackets, e.g. "Revenue grew 12% in Q3 [a1b2c3d4e5f6]."
3. If a sentence draws on multiple chunks, cite all of them: "... [id1][id2]."
4. If the context does not contain enough information to answer, say so plainly —
   do NOT guess, and do NOT attach a citation to a claim the context doesn't support.
5. Keep the answer concise and directly responsive to the question.
"""

USER_TEMPLATE = """CONTEXT:
{context_block}

QUESTION: {query}

Answer the question following all citation rules above."""


def build_context_block(candidates: list[Candidate]) -> str:
    blocks = []
    for c in candidates:
        page_str = f" p.{c.page}" if c.page else ""
        blocks.append(f"[{c.chunk_id}] ({c.source}{page_str}):\n{c.text}")
    return "\n\n".join(blocks)


def build_user_prompt(query: str, candidates: list[Candidate]) -> str:
    return USER_TEMPLATE.format(context_block=build_context_block(candidates), query=query)