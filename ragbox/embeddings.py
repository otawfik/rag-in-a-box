"""Swappable embedding backends.

The retrieval layer only depends on the :class:`Embedder` interface
(``fit`` / ``embed`` / ``dimension``), so the vector representation can be
upgraded without touching chunking, storage, or answering code.

Included backends:

- :class:`TfidfEmbedder` (default): scikit-learn TF-IDF with word unigrams +
  bigrams. Runs anywhere, needs no API keys and no downloads. Vectors are
  L2-normalized, so a dot product equals cosine similarity.
- :class:`SentenceTransformerEmbedder`: dense neural embeddings via the
  ``sentence-transformers`` package. Optional dependency — install it and
  pass ``get_embedder("sbert:all-MiniLM-L6-v2")`` to swap it in.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Embedder(ABC):
    """Interface every embedding backend must implement."""

    @abstractmethod
    def fit(self, texts: list[str]) -> "Embedder":
        """Learn any corpus statistics (vocabulary, IDF weights, ...)."""

    @abstractmethod
    def embed(self, texts: list[str]) -> np.ndarray:
        """Return an ``(len(texts), dimension)`` float32 matrix.

        Rows MUST be L2-normalized so the store can use dot-product search.
        """

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Width of the embedding vectors."""


class TfidfEmbedder(Embedder):
    """Sparse TF-IDF embeddings. Zero setup, surprisingly strong for RAG."""

    def __init__(self, max_features: int = 20_000, ngram_range: tuple[int, int] = (1, 2)):
        from sklearn.feature_extraction.text import TfidfVectorizer

        self._vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            sublinear_tf=True,   # dampen ultra-frequent terms
            stop_words="english",
            norm="l2",           # dot product == cosine similarity
        )
        self._fitted = False

    def fit(self, texts: list[str]) -> "TfidfEmbedder":
        self._vectorizer.fit(texts)
        self._fitted = True
        return self

    def embed(self, texts: list[str]) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("TfidfEmbedder.embed() called before fit().")
        return self._vectorizer.transform(texts).toarray().astype(np.float32)

    @property
    def dimension(self) -> int:
        return len(self._vectorizer.vocabulary_)


class SentenceTransformerEmbedder(Embedder):
    """Dense neural embeddings (optional ``sentence-transformers`` package).

    Swap-in path::

        pip install sentence-transformers
        embedder = get_embedder("sbert:all-MiniLM-L6-v2")
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "SentenceTransformerEmbedder needs the optional "
                "'sentence-transformers' package: pip install sentence-transformers"
            ) from exc
        self._model = SentenceTransformer(model_name)
        self._model_name = model_name

    def fit(self, texts: list[str]) -> "SentenceTransformerEmbedder":
        return self  # pre-trained; nothing to fit

    def embed(self, texts: list[str]) -> np.ndarray:
        return np.asarray(
            self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False),
            dtype=np.float32,
        )

    @property
    def dimension(self) -> int:
        return int(self._model.get_sentence_embedding_dimension())


def get_embedder(spec: str = "tfidf") -> Embedder:
    """Build an embedder from a short spec string.

    - ``"tfidf"`` → :class:`TfidfEmbedder` (default)
    - ``"sbert:<model-name>"`` → :class:`SentenceTransformerEmbedder`
    """
    if spec == "tfidf":
        return TfidfEmbedder()
    if spec.startswith("sbert:"):
        return SentenceTransformerEmbedder(spec.split(":", 1)[1])
    raise ValueError(f"Unknown embedder spec {spec!r}; use 'tfidf' or 'sbert:<model>'")
