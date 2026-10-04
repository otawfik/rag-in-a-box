"""End-to-end RAG pipeline: index documents, retrieve, answer with citations.

The answering step is *extractive*: it selects the most query-relevant
sentences from the retrieved chunks and presents them with ``[n]`` citations
pointing at the source document and section — so every claim is traceable.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from .chunking import Chunk, chunk_documents, load_markdown, split_sentences
from .embeddings import Embedder, get_embedder
from .hybrid import HybridVectorStore
from .store import SearchResult, VectorStore

#: Below this cosine similarity the corpus is considered to have no answer.
MIN_RELEVANCE = 0.05

_CONTENT_WORD = re.compile(r"[a-z][a-z\-']{2,}")
_PRICE_RE = re.compile(r"\$\s?[\d,]+")
_NUMBER_RE = re.compile(r"\d[\d,]*")

# Lightweight question-type cues so "how much" finds "$48,000" even though
# the words "cost" and "price" never co-occur lexically. Documented, tiny,
# and strictly a tie-breaker on top of retrieval scores.
_PRICE_WORDS = {"much", "cost", "price", "prices", "pay", "pays", "expensive", "cheap", "fare", "fares"}
_TIME_WORDS = {"when", "long", "duration", "days", "years", "depart", "departs",
               "departure", "launch", "schedule"}


def _content_words(text: str) -> set[str]:
    return {w for w in _CONTENT_WORD.findall(text.lower())} - ENGLISH_STOP_WORDS


def _embed_text(chunk: Chunk) -> str:
    """Text actually embedded: section heading prepended so titles are searchable."""
    if chunk.section:
        return f"{chunk.section}. {chunk.text}"
    return chunk.text


def _sentence_bonus(sentence: str, query_words: set[str]) -> float:
    bonus = 0.0
    if query_words & _PRICE_WORDS:
        if _PRICE_RE.search(sentence):
            bonus += 2.0
        elif _NUMBER_RE.search(sentence):
            bonus += 0.5
    if query_words & _TIME_WORDS and _NUMBER_RE.search(sentence):
        bonus += 0.75
    return bonus


@dataclass
class Citation:
    number: int
    doc_id: str
    section: str
    source: str
    score: float


@dataclass
class CitedAnswer:
    question: str
    text: str
    citations: list[Citation] = field(default_factory=list)
    latency_ms: float = 0.0

    @property
    def has_answer(self) -> bool:
        return bool(self.citations)

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "answer": self.text,
            "citations": [
                {"n": c.number, "doc_id": c.doc_id, "section": c.section,
                 "source": c.source, "score": round(c.score, 3)}
                for c in self.citations
            ],
            "latency_ms": round(self.latency_ms, 1),
        }


class RAGPipeline:
    """Index-once, ask-many-times retrieval chatbot."""

    def __init__(
        self,
        embedder: Embedder | None = None,
        retrieval: str = "tfidf",
        dense_model: str = "all-MiniLM-L6-v2",
        hybrid_method: str = "rrf",
        hybrid_alpha: float = 0.5,
        chunk_size: int = 600,
        overlap: int = 120,
        chunk_strategy: str = "recursive",
        top_k: int = 4,
    ) -> None:
        """``retrieval``: ``"tfidf"`` (sparse, default), ``"dense"``
        (sentence-transformers), or ``"hybrid"`` (both, fused).

        ``hybrid_method`` is ``"rrf"`` or ``"weighted"`` (see
        :mod:`ragbox.hybrid`); ``hybrid_alpha`` only applies to ``"weighted"``.
        The dense model downloads on first use (~90MB) and needs the optional
        ``sentence-transformers`` package.
        """
        if retrieval not in ("tfidf", "dense", "hybrid"):
            raise ValueError(f"Unknown retrieval {retrieval!r}; "
                             "use 'tfidf', 'dense' or 'hybrid'")
        self.embedder = embedder or get_embedder("tfidf")  # sparse side
        self.retrieval = retrieval
        self._dense_model = dense_model
        self._dense_embedder: Embedder | None = None
        self.hybrid_method = hybrid_method
        self.hybrid_alpha = hybrid_alpha
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.chunk_strategy = chunk_strategy
        self.top_k = top_k
        self.store = HybridVectorStore() if retrieval == "hybrid" else VectorStore()

    @property
    def dense_embedder(self) -> Embedder:
        """Lazily built dense embedder: tfidf-only use never touches torch."""
        if self._dense_embedder is None:
            self._dense_embedder = get_embedder(f"sbert:{self._dense_model}")
        return self._dense_embedder

    # -- indexing ---------------------------------------------------------
    def index_documents(self, docs: list[dict]) -> int:
        """Chunk + embed a list of document dicts. Returns chunk count.

        ``"hybrid"`` embeds every chunk twice (sparse + dense);
        ``"dense"`` embeds dense-only; ``"tfidf"`` sparse-only.
        """
        chunks = chunk_documents(
            docs,
            chunk_size=self.chunk_size,
            overlap=self.overlap,
            strategy=self.chunk_strategy,
        )
        if not chunks:
            return 0
        embed_texts = [_embed_text(c) for c in chunks]
        self.embedder.fit(embed_texts)
        if self.retrieval == "hybrid":
            sparse_vectors = self.embedder.embed(embed_texts)
            dense_vectors = self.dense_embedder.embed(embed_texts)
            self.store.add(chunks, sparse_vectors, dense_vectors)
        else:
            active = self.dense_embedder if self.retrieval == "dense" else self.embedder
            self.store.add(chunks, active.embed(embed_texts))
        return len(chunks)

    def index_directory(self, corpus_dir: str | Path, pattern: str = "*.md") -> int:
        """Index every matching markdown file under ``corpus_dir``."""
        docs: list[dict] = []
        for path in sorted(Path(corpus_dir).glob(pattern)):
            docs.extend(load_markdown(path))
        return self.index_documents(docs)

    # -- retrieval --------------------------------------------------------
    def retrieve(self, question: str, top_k: int | None = None) -> list[SearchResult]:
        k = top_k or self.top_k
        if self.retrieval == "hybrid":
            qs = self.embedder.embed([question])[0]
            qd = self.dense_embedder.embed([question])[0]
            return self.store.search(qs, qd, top_k=k,
                                     method=self.hybrid_method,
                                     alpha=self.hybrid_alpha)
        active = self.dense_embedder if self.retrieval == "dense" else self.embedder
        qv = active.embed([question])[0]
        return self.store.search(qv, top_k=k)

    # -- answering --------------------------------------------------------
    def answer(self, question: str, top_k: int | None = None) -> CitedAnswer:
        started = time.perf_counter()
        results = self.retrieve(question, top_k=top_k)
        results = [r for r in results if r.score >= MIN_RELEVANCE]
        if not results:
            return CitedAnswer(
                question=question,
                text=("I couldn't find anything relevant in the indexed documents. "
                      "Try rephrasing, or index a corpus that covers the topic."),
                latency_ms=(time.perf_counter() - started) * 1000,
            )
        query_words = _content_words(question)
        # Rank every sentence in the retrieved chunks: lexical overlap with
        # the question first, then retrieval score, then question-type cues
        # (e.g. "how much" prefers sentences with prices).
        candidates: list[tuple[float, int, str, SearchResult]] = []
        for rank, res in enumerate(results):
            for sent in split_sentences(res.chunk.text):
                words = _content_words(sent)
                overlap = len(words & query_words)
                score = overlap * 1.0 + res.score * 3.0 + _sentence_bonus(sent, query_words)
                candidates.append((score, rank, sent, res))
        candidates.sort(key=lambda t: (-t[0], t[1]))

        picked: list[tuple[str, SearchResult]] = []
        per_chunk: dict[str, int] = {}
        for score, _, sent, res in candidates:
            if len(picked) >= 4 or score <= 0:
                break
            if any(sent == p[0] for p in picked):
                continue
            if per_chunk.get(res.chunk.chunk_id, 0) >= 2:
                continue  # at most 2 sentences per chunk: breadth over depth
            picked.append((sent, res))
            per_chunk[res.chunk.chunk_id] = per_chunk.get(res.chunk.chunk_id, 0) + 1
        if not picked:  # nothing overlapped the question; show the top chunk
            top = results[0]
            sents = split_sentences(top.chunk.text)
            if sents:
                picked.append((sents[0], top))

        # Number citations in order of first appearance.
        order: list[SearchResult] = []
        for _, res in picked:
            if res not in order:
                order.append(res)
        num = {id(r): i + 1 for i, r in enumerate(order)}
        citations = [
            Citation(number=i + 1, doc_id=r.chunk.doc_id,
                     section=r.chunk.section or r.chunk.doc_id,
                     source=r.chunk.source, score=r.score)
            for i, r in enumerate(order)
        ]

        lines = [f"{sent} [{num[id(res)]}]" for sent, res in picked]
        lines.append("")
        lines.append("Sources:")
        for c in citations:
            label = c.section if c.section else c.doc_id
            lines.append(f"  [{c.number}] {label} (relevance {c.score:.2f})")
        return CitedAnswer(
            question=question,
            text="\n".join(lines),
            citations=citations,
            latency_ms=(time.perf_counter() - started) * 1000,
        )

    # -- persistence ------------------------------------------------------
    def save(self, directory: str | Path) -> Path:
        """Persist the store plus the retrieval configuration."""
        path = self.store.save(directory)
        (path / "retriever.json").write_text(json.dumps({
            "retrieval": self.retrieval,
            "dense_model": self._dense_model,
            "hybrid_method": self.hybrid_method,
            "hybrid_alpha": self.hybrid_alpha,
        }, indent=1), encoding="utf-8")
        return path

    @classmethod
    def load(cls, directory: str | Path, **kwargs) -> "RAGPipeline":
        """Load a pipeline whose store was written with :meth:`save`.

        The retrieval mode is read from ``retriever.json``; explicit keyword
        arguments override the saved configuration.
        """
        directory = Path(directory)
        cfg: dict = {}
        rp = directory / "retriever.json"
        if rp.exists():
            cfg = json.loads(rp.read_text(encoding="utf-8"))
        cfg.update(kwargs)
        retrieval = cfg.get("retrieval", "tfidf")
        init_kwargs = {
            "retrieval": retrieval,
            "dense_model": cfg.get("dense_model", "all-MiniLM-L6-v2"),
            "hybrid_method": cfg.get("hybrid_method", "rrf"),
            "hybrid_alpha": cfg.get("hybrid_alpha", 0.5),
        }
        init_kwargs.update({k: v for k, v in kwargs.items()
                            if k in ("chunk_size", "overlap", "chunk_strategy", "top_k")})
        pipe = cls(**init_kwargs)
        pipe.store = (HybridVectorStore.load(directory) if retrieval == "hybrid"
                      else VectorStore.load(directory))
        if len(pipe.store):
            pipe.embedder.fit([_embed_text(c) for c in pipe.store.chunks])
        return pipe


def quickstart(corpus_dir: str | Path = "corpus") -> RAGPipeline:
    """Build a pipeline indexed on ``corpus_dir`` — the 3-line demo."""
    pipe = RAGPipeline()
    n = pipe.index_directory(corpus_dir)
    print(f"Indexed {n} chunks from {corpus_dir}")
    return pipe
