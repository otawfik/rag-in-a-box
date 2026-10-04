"""Eval set schema and helpers for the 10-K RAG eval (steps 4-5).

Schema (``eval/eval_set.json``)::

    {
      "meta": {"created": ..., "corpus": [...], "chunking": {...}},
      "questions": [
        {"id": "aapl-01",
         "question": "What was Apple's total net sales in FY2025?",
         "answer": "$416.2 billion",
         "type": "factual",          # factual | comparison | unanswerable
         "gold_chunks": ["AAPL_10K_FY2025#s7#c42", ...],
         "gold_sources": [{"doc_id": ..., "section": ...}],
         "verified": false}          # set true only by human verification
      ]
    }

``gold_chunks`` are chunk IDs from the pinned chunking config in ``meta``;
step 5's harness measures recall@5 as the fraction of questions with at
least one gold chunk in the top-5 retrieved. ``unanswerable`` questions
have empty ``gold_chunks`` and expect ``refused=True``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class EvalQuestion:
    id: str
    question: str
    answer: str
    type: str = "factual"  # factual | comparison | unanswerable
    gold_chunks: list[str] = field(default_factory=list)
    gold_sources: list[dict] = field(default_factory=list)
    verified: bool = False

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "question": self.question,
            "answer": self.answer,
            "type": self.type,
            "gold_chunks": self.gold_chunks,
            "gold_sources": self.gold_sources,
            "verified": self.verified,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "EvalQuestion":
        return cls(
            id=data["id"],
            question=data["question"],
            answer=data.get("answer", ""),
            type=data.get("type", "factual"),
            gold_chunks=data.get("gold_chunks", []),
            gold_sources=data.get("gold_sources", []),
            verified=data.get("verified", False),
        )


def load_eval_set(path: str | Path) -> tuple[dict, list[EvalQuestion]]:
    """Return (meta, questions) from an eval set JSON file."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return data.get("meta", {}), [EvalQuestion.from_dict(q) for q in data["questions"]]


def save_eval_set(questions: list[EvalQuestion], path: str | Path, meta: dict) -> Path:
    """Write the eval set JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {"meta": meta, "questions": [q.to_dict() for q in questions]}
    path.write_text(json.dumps(data, indent=1), encoding="utf-8")
    return path


def find_gold_candidates(pipe, question: str, top_k: int = 5) -> list[dict]:
    """Drafting helper: retrieve top-k chunks for a candidate question.

    Returns chunk metadata for a human to pick gold chunks from. Used when
    building the eval set (step 4), not in the harness itself.
    """
    return [
        {"chunk_id": r.chunk.chunk_id,
         "doc_id": r.chunk.doc_id,
         "section": r.chunk.section,
         "score": round(r.score, 4),
         "text": r.chunk.text}
        for r in pipe.retrieve(question, top_k=top_k)
    ]
