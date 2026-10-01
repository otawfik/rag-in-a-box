# 📦 RAG-in-a-Box

A production-style **retrieval-augmented generation (RAG) chatbot** that runs
entirely on your machine — no API keys, no GPU, no cloud bills. Index your
documents, ask questions in plain English, and get answers with **[n]
citations** pointing back to the exact source sections.

Ships with a fun demo corpus: the *Acme Space Tours* passenger guide
(fictional space-tourism company).

## Features

- **Document chunking engine** — three strategies (`recursive`, `sentence`,
  `fixed`) with configurable size/overlap; markdown-aware section splitting
- **Swappable embedding backends** — TF-IDF + cosine similarity out of the
  box (zero setup); drop in `sentence-transformers` dense embeddings with a
  one-line change
- **Cited answers** — extractive answering composes replies from the most
  relevant sentences and cites every claim as `[1]`, `[2]`, …
- **Web chat UI** (Flask) + **CLI** + **JSON API**
- **Persistent index** — built once, cached to disk, reloaded instantly
- **Test suite** covering chunking, embeddings, retrieval ranking, and the
  full pipeline

## Quickstart

```bash
git clone https://github.com/otawfik/rag-in-a-box.git
cd rag-in-a-box
pip install -r requirements.txt        # Python 3.10+

# Ask from the command line:
python app.py --ask "How much does a Mars trip cost?"

# …or launch the web UI:
python app.py                          # → http://127.0.0.1:5000
```

The first run builds the index from `corpus/` and caches it in
`.ragbox_index/`; later runs load instantly.

## Try these demo questions

| Question | What it shows |
|---|---|
| `How much does a Mars trip cost?` | Exact-fact retrieval with citation |
| `Which ship flies to Europa?` | Multi-hop-ish: destination → fleet section |
| `Can I bring my dog?` | FAQ-style answer (`pets` policy) |
| `How long is the Europa trip?` | Numeric fact extraction |
| `What happens if my flight is scrubbed?` | Policy retrieval |
| `Tell me about quantum teleportation` | Graceful "not in the corpus" fallback |

## How it works

```
corpus/*.md ──▶ chunking ──▶ TF-IDF embed ──▶ VectorStore (cosine search)
                                                        │
user question ──▶ embed ──▶ top-k chunks ──▶ sentence rerank ──▶ cited answer
```

1. **Chunk** — markdown files are split per `##` section, then packed into
   ~600-char overlapping chunks (`ragbox/chunking.py`).
2. **Embed** — chunks become L2-normalized TF-IDF vectors (unigrams +
   bigrams), so dot-product search equals cosine similarity
   (`ragbox/embeddings.py`).
3. **Retrieve** — the query is embedded the same way; the top-k chunks are
   fetched from the in-memory store (`ragbox/store.py`).
4. **Answer** — sentences inside the retrieved chunks are scored by
   retrieval score + query-term overlap; the best ones are composed into an
   answer with `[n]` citations and a Sources list (`ragbox/pipeline.py`).

## Swapping in neural embeddings

```bash
pip install sentence-transformers
```

```python
from ragbox.embeddings import get_embedder
from ragbox.pipeline import RAGPipeline

pipe = RAGPipeline(embedder=get_embedder("sbert:all-MiniLM-L6-v2"))
pipe.index_directory("corpus")
```

Everything downstream (store, retrieval, answering) works unchanged because
it only depends on the `Embedder` interface.

## Project structure

```
rag-in-a-box/
├── app.py                 # Flask web UI + CLI entrypoint
├── ragbox/
│   ├── chunking.py        # chunking strategies, markdown loading
│   ├── embeddings.py      # swappable Embedder backends
│   ├── store.py           # vector store, cosine search, persistence
│   └── pipeline.py        # RAGPipeline: index → retrieve → cited answer
├── corpus/
│   └── acme-space-tours.md  # demo corpus (fictional space tourism guide)
├── tests/
│   └── test_ragbox.py     # unittest suite
└── requirements.txt
```

## Tech highlights

- **scikit-learn** `TfidfVectorizer` (sublinear TF, L2 norm) as a zero-dependency
  vector layer; **NumPy** for the similarity math
- Recursive character chunking with sentence-aware packing and overlap
  carry-over — no NLTK/spaCy needed
- Extractive cited-answer generation: every sentence in the reply traces to
  a ranked source chunk
- Clean `Embedder` ABC so sparse → dense upgrades don't ripple through the
  codebase

## Limitations

- TF-IDF is lexical: it won't catch heavy paraphrase ("cheap" vs
  "inexpensive" is fine; "lunar" vs "moon" is borderline). Use the
  `sentence-transformers` backend for semantic matching.
- Answers are extractive, not generative — great for factual QA over your
  docs, not for open-ended writing.

## License

MIT
