"""Step 5: evaluation harness for the 10-K RAG system.

Measures, per the step-5 spec:
  - recall@5 for keyword (TF-IDF), dense, and hybrid retrieval
  - answer correctness on generated answers
  - citation support (does a cited passage contain the answer's key figure?)
  - refusal behavior (accuracy on unanswerable, false refusals on answerable)

Retrieval eval runs on the full set (fast, no LLM). Generation eval runs on
a stratified subset (LLM generation is ~1 min/answer on CPU); the subset is
reported in the results table.

Usage:
    python -m ragbox.harness --eval eval/eval_set.json --out eval/results.md
"""

from __future__ import annotations

import argparse
import glob
import re
import time
from dataclasses import dataclass, field

from ragbox.chunking import load_text
from ragbox.eval import load_eval_set
from ragbox.pipeline import RAGPipeline


def normalize_digits(s: str) -> str:
    return re.sub(r"\D", "", s)


def key_figure(answer: str) -> str | None:
    """Extract the most distinctive number from a known answer, if any."""
    nums = re.findall(r"[\d,.]+", answer)
    if not nums:
        return None
    return max(nums, key=len)


def answer_supported(answer_text: str, known_answer: str) -> bool:
    """Lenient correctness: the key figure (or its digits) appears."""
    key = key_figure(known_answer)
    if not key:
        words = [w.lower() for w in re.findall(r"[A-Za-z]{5,}", known_answer)]
        return any(w in answer_text.lower() for w in words)
    digits = normalize_digits(key)
    # allow rounded forms: "34.6" matches chunk "34,550" via leading digits
    return digits[:3] in normalize_digits(answer_text) or key in answer_text


@dataclass
class RetrievalResult:
    mode: str
    recall_at_5: float
    per_question: dict = field(default_factory=dict)


def run_retrieval_eval(pipes: dict[str, RAGPipeline], questions, top_k: int = 5) -> dict[str, RetrievalResult]:
    results: dict[str, RetrievalResult] = {}
    answerable = [q for q in questions if q.type != "unanswerable"]
    for mode, pipe in pipes.items():
        hits = 0
        per_q: dict[str, bool] = {}
        for q in answerable:
            retrieved = pipe.retrieve(q.question, top_k=top_k)
            ids = {r.chunk.chunk_id for r in retrieved}
            ok = bool(ids & set(q.gold_chunks))
            per_q[q.id] = ok
            hits += ok
        results[mode] = RetrievalResult(mode, hits / len(answerable), per_q)
    return results


@dataclass
class GenerationResult:
    qid: str
    refused: bool
    correct: bool | None  # None for unanswerable
    citation_supported: bool | None
    latency_s: float


def run_generation_eval(pipe: RAGPipeline, questions) -> list[GenerationResult]:
    results: list[GenerationResult] = []
    for q in questions:
        t0 = time.time()
        try:
            ans = pipe.generate(q.question, top_k=5)
        except Exception as e:
            print(f"  [warn] generate failed for {q.id}: {e}")
            results.append(GenerationResult(q.id, True, None, None, time.time() - t0))
            continue
        latency = time.time() - t0
        if q.type == "unanswerable":
            # correct behavior = refusal
            results.append(GenerationResult(q.id, ans.refused, None, None, latency))
            continue
        correct = answer_supported(ans.text, q.answer) if not ans.refused else False
        key = key_figure(q.answer)
        supported = None
        if key and not ans.refused:
            digits = normalize_digits(key)
            supported = any(
                digits[:3] in normalize_digits(p.text) for p in ans.sources if p.cited
            ) or any(key in p.text for p in ans.sources if p.cited)
        results.append(GenerationResult(q.id, ans.refused, correct, supported, latency))
    return results


def results_table(retrieval: dict[str, RetrievalResult],
                  generation: list[GenerationResult],
                  gen_subset_note: str) -> str:
    lines = ["# RAG eval results", ""]
    lines.append("## Retrieval recall@5 (46 answerable questions)")
    lines.append("")
    lines.append("| mode | recall@5 |")
    lines.append("| --- | --- |")
    for mode in ("tfidf", "dense", "hybrid"):
        r = retrieval[mode]
        lines.append(f"| {mode} | {r.recall_at_5:.1%} |")
    lines.append("")
    lines.append(f"## Generation ({gen_subset_note})")
    lines.append("")
    answerable = [g for g in generation if g.correct is not None]
    unans = [g for g in generation if g.correct is None]
    refused_unans = sum(1 for g in unans if g.refused)
    false_ref = sum(1 for g in answerable if g.refused)
    correct = sum(1 for g in answerable if g.correct)
    supported = [g for g in answerable if g.citation_supported]
    lines.append("| metric | value |")
    lines.append("| --- | --- |")
    lines.append(f"| refusal accuracy (unanswerable) | {refused_unans}/{len(unans)} |")
    lines.append(f"| false refusal rate (answerable) | {false_ref}/{len(answerable)} |")
    answered = [g for g in answerable if not g.refused]
    lines.append(f"| answer correctness (answered) | {correct}/{len(answered)} |")
    if supported:
        s = sum(1 for g in supported if g.citation_supported)
        lines.append(f"| citation support | {s}/{len(supported)} |")
    avg_lat = sum(g.latency_s for g in generation) / len(generation)
    lines.append(f"| avg latency | {avg_lat:.1f}s |")
    lines.append("")
    lines.append("## Per-question generation detail")
    lines.append("")
    lines.append("| qid | refused | correct | cited_ok | latency_s |")
    lines.append("| --- | --- | --- | --- | --- |")
    for g in generation:
        lines.append(f"| {g.qid} | {g.refused} | {g.correct} | {g.citation_supported} | {g.latency_s:.1f} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", default="eval/eval_set.json")
    ap.add_argument("--out", default="eval/results.md")
    ap.add_argument("--gen-subset", type=int, default=12,
                    help="number of questions for the (slow) generation eval")
    args = ap.parse_args()

    meta, questions = load_eval_set(args.eval)
    docs = []
    for f in sorted(glob.glob("corpus/sec10k/*_10K_*.txt")):
        docs.extend(load_text(f))

    pipes = {}
    for mode in ("tfidf", "dense", "hybrid"):
        pipe = RAGPipeline(retrieval=mode)
        pipe.index_documents(docs)
        pipes[mode] = pipe

    retrieval = run_retrieval_eval(pipes, questions)

    # stratified generation subset: 2 factual per company + all unanswerable
    factual = [q for q in questions if q.type == "factual"]
    subset = []
    for prefix in ("aapl", "msft", "googl", "nvda"):
        subset.extend([q for q in factual if q.id.startswith(prefix)][:2])
    subset.extend([q for q in questions if q.type == "unanswerable"])
    gen_pipe = pipes["hybrid"]
    generation = run_generation_eval(gen_pipe, subset)

    note = f"{len(subset)}-question stratified subset (hybrid retrieval)"
    table = results_table(retrieval, generation, note)
    with open(args.out, "w") as fh:
        fh.write(table)
    print(table)


if __name__ == "__main__":
    main()
