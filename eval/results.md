# RAG eval results

Generated 2026-10-04. Eval set: `eval/eval_set.json` (50 questions: 40 factual,
6 comparison, 4 unanswerable; all verified against filing text).

## Retrieval recall@5 (46 answerable questions)

| mode | recall@5 |
| --- | --- |
| tfidf | 45.7% |
| dense | 50.0% |
| hybrid | 52.2% |

Recall@5 = fraction of questions with at least one gold chunk in the top-5.

**Diagnosis.** The biggest miss pattern is company disambiguation: e.g. "How much
did Apple spend on R&D?" retrieves Microsoft's, Alphabet's, and NVIDIA's R&D
chunks above Apple's. The company name appears once in the question while
"research and development" matches many chunks. A production fix would be a
company entity filter or name-boost before ranking.

## Extractive baseline (8-question subset, TF-IDF retrieval)

| metric | value |
| --- | --- |
| answer correctness | 5/8 |

Uses the extractive `RAGPipeline.answer()` (no LLM). Misses are the same
company-disambiguation failures as above.

## LLM generation eval

Built in `ragbox/harness.py` (`run_generation_eval`) but not run here: the
Qwen2.5-1.5B model needs ~3GB RAM and the VM had <1GB available at eval time.
The harness measures refusal accuracy, false-refusal rate, answer correctness,
and citation support on a 12-question stratified subset. Re-run with
`python -m ragbox.harness` on a machine with headroom.
