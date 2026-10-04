"""LLM answer generation grounded in retrieved passages.

Three-layer refusal design (see :class:`Generator`):

1. **Retrieval gate** (in :meth:`RAGPipeline.generate`): if retrieval finds
   nothing relevant, refuse *without* calling the LLM. Cheap, deterministic.
   For TF-IDF/dense this is a cosine floor (``MIN_RELEVANCE``); for hybrid
   RRF the fused scores are not cosine similarities, so any non-empty
   ranking counts and layer 2 becomes the backstop.
2. **Relevance judge** (:meth:`Generator.judge`): a binary YES/NO call asking
   whether the passages contain the answer. NO -> refuse with
   :data:`REFUSAL_TEXT`, no generation call.
3. **Grounded generation** (:meth:`Generator.generate`): answers from the
   passages with ``[n]`` citations. No refusal instruction lives in this
   prompt (see below).

Why the split? On Qwen2.5-1.5B-Instruct, a *joint* answer-or-refuse prompt
collapses: any mention of the refusal option makes the model refuse
*everything*, including questions the passages directly answer (measured).
A binary relevance judgment is an easier task for a small model and is
reliable both ways, while a pure generation prompt (no refusal option)
answers and cites reliably. Splitting judgment from generation fixes the
over-refusal without losing the refusal capability.

Known failure modes (defend these in an interview):

- *Over-refusal*: the judge says NO even though the answer is present, or
  the retrieval gate's cosine floor is set too high. The judge is the main
  residual risk; it can be ablated (``judge=False``) to measure its effect.
- *Parametric leakage*: the generator answers from its own weights despite
  "only use the passages". The judge passing first makes this less likely
  (the passages do contain the answer), but the constraint is still a
  request, not a guarantee — small models leak.
- *Citation granularity*: the 1.5B model does not reliably emit per-claim
  ``[n]`` markers (seven prompt variants tested), so citations are
  passage-level: the answer was conditioned solely on the cited passages
  and the judge confirmed they contain the answer. Per-sentence provenance
  remains available via the extractive :meth:`RAGPipeline.answer`. A larger
  model would likely follow per-claim citation instructions; the
  ``llm_model`` parameter is the swap-in path.
- *Threshold brittleness*: the layer-1 cosine floor is a magic number. Too
  high causes false refusals; too low passes garbage to the judge (which is
  exactly what the judge is for).
- *Untrusted passages*: filing text is data, not instructions. A hostile
  passage could try prompt injection ("ignore previous instructions").
  10-Ks are low-risk, but the architecture treats passages as untrusted.
- *Latency*: up to two LLM calls per question (judge + generate). On CPU
  with a 1.5B model that is ~30-60s per answer; a GPU or a larger model
  changes the tradeoff.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

#: Exact sentence the model is instructed to emit when the passages lack
#: the answer. Detection is tolerant (see is_refusal), but generation aims
#: for this verbatim so downstream code can match reliably.
REFUSAL_TEXT = "I cannot answer this from the provided filings."

_REFUSAL_RE = re.compile(
    r"\bi\s+cannot\s+answer\b.{0,40}\bfrom the provided filings\b",
    re.IGNORECASE,
)

_CITATION_RE = re.compile(r"\[(\d+)\]")


def extract_citations(text: str, n_passages: int) -> list[int]:
    """0-based passage indices cited in ``text``, in order of first
    appearance. Out-of-range citations (hallucinated) are dropped."""
    cited: list[int] = []
    for m in _CITATION_RE.finditer(text or ""):
        idx = int(m.group(1)) - 1
        if 0 <= idx < n_passages and idx not in cited:
            cited.append(idx)
    return cited


def is_refusal(text: str) -> bool:
    """True if the model emitted the refusal sentence."""
    return bool(_REFUSAL_RE.search(text or ""))


def build_prompt(question: str, passages: list[str]) -> list[dict]:
    """ChatML messages for grounded generation. Passages are numbered [1..n].

    The instruction is deliberately light. Measured on Qwen2.5-1.5B-Instruct:
    the model answers fluently from the passages but does not reliably emit
    ``[n]`` markers no matter how the citation format is instructed (seven
    prompt variants tested: per-sentence instruction, format example,
    few-shot demo, bullet-prefix instruction — all ignored on realistic
    contexts). Citations are therefore passage-level (see
    :meth:`Generator.generate`): the answer is conditioned only on the cited
    passages, and the judge has confirmed they contain the answer. If the
    model happens to emit ``[n]`` markers, they are captured best-effort.
    """
    numbered = "\n\n".join(f"[{i + 1}]\n{p}" for i, p in enumerate(passages))
    return [
        {"role": "system", "content": (
            "Answer the question using ONLY the passages below. "
            "Do not use any outside knowledge."
        )},
        {"role": "user", "content": f"Passages:\n{numbered}\n\nQuestion: {question}"},
    ]


def build_judge_prompt(question: str, passages: list[str]) -> list[dict]:
    """ChatML messages for the binary relevance judgment.

    A YES/NO question is a far easier task for a small model than a joint
    answer-or-refuse, and it is reliable in both directions.
    """
    numbered = "\n\n".join(f"[{i + 1}]\n{p}" for i, p in enumerate(passages))
    return [
        {"role": "user", "content": (
            f"Passages:\n{numbered}\n\nQuestion: {question}\n\n"
            "Do the passages contain the information needed to answer the "
            "question? Reply with exactly one word: YES or NO."
        )},
    ]


@dataclass
class GeneratedAnswer:
    question: str
    text: str
    refused: bool = False
    #: passage indices (0-based) actually cited, in order of first appearance
    cited: list[int] = field(default_factory=list)
    #: the passages that were given to the model, for source rendering
    passages: list[str] = field(default_factory=list)
    #: per-passage metadata (doc_id, section) parallel to ``passages``
    sources: list[dict] = field(default_factory=list)
    latency_ms: float = 0.0

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "answer": self.text,
            "refused": self.refused,
            "cited": self.cited,
            "sources": self.sources,
            "latency_ms": round(self.latency_ms, 1),
        }


class Generator:
    """Local instruction-tuned LLM for grounded answer generation.

    Default model is Qwen2.5-1.5B-Instruct: small enough for CPU demos,
    strong enough to follow cite-or-refuse instructions. Swap via
    ``model_name``. Greedy decoding (``do_sample=False``) keeps answers
    deterministic for later eval.
    """

    def __init__(self, model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
                 max_new_tokens: int = 256):
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "Generator needs the 'transformers' package: pip install transformers"
            ) from exc
        import torch

        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModelForCausalLM.from_pretrained(model_name)
        self._model.eval()
        self._torch = torch

    def _complete(self, messages: list[dict], max_new_tokens: int) -> str:
        """Run one chat completion, return the decoded new tokens."""
        prompt = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        inputs = self._tokenizer(prompt, return_tensors="pt")
        with self._torch.no_grad():
            out = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,          # greedy: deterministic for eval
                pad_token_id=self._tokenizer.eos_token_id,
            )
        new_tokens = out[0][inputs["input_ids"].shape[1]:]
        return self._tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    def judge(self, question: str, passages: list[str]) -> bool:
        """Binary relevance judgment: do the passages contain the answer?

        Small models answer YES/NO reliably in both directions, where a
        joint answer-or-refuse prompt collapses to always-refuse.
        """
        text = self._complete(build_judge_prompt(question, passages),
                              max_new_tokens=8)
        return text.strip().upper().startswith("YES")

    def generate(self, question: str, passages: list[str],
                 sources: list[dict] | None = None) -> GeneratedAnswer:
        """Answer ``question`` from ``passages``.

        Assumes the passages were judged relevant already. Citations are
        passage-level: the answer was conditioned solely on these passages,
        so ``cited`` lists all of them (plus any ``[n]`` markers the model
        happened to emit, best-effort). An empty generation is refused.
        """
        started = time.perf_counter()
        text = self._complete(build_prompt(question, passages),
                              max_new_tokens=self.max_new_tokens)
        refused = not text.strip()
        if refused:
            text = REFUSAL_TEXT
        cited = extract_citations(text, len(passages)) or list(range(len(passages)))
        if refused:
            cited = []
        return GeneratedAnswer(
            question=question,
            text=text,
            refused=refused,
            cited=cited,
            passages=list(passages),
            sources=list(sources or []),
            latency_ms=(time.perf_counter() - started) * 1000,
        )
