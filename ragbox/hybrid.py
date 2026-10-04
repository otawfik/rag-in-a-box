"""Hybrid sparse + dense retrieval.

Holds one TF-IDF (sparse, exact lexical match) and one sentence-transformer
(dense, semantic match) vector per chunk, and fuses the two rankings at query
time. Two fusion methods are supported — see :func:`fuse_scores`:

- ``"rrf"`` (default): Reciprocal Rank Fusion. Ranks are fused, never raw
  scores, so nothing needs normalizing. Robust when the two scorers live on
  different score scales (they do: sparse and dense cosine similarities have
  very different distributions over a corpus).
- ``"weighted"``: min-max normalize each scorer's scores to [0, 1], then
  ``alpha * dense + (1 - alpha) * sparse``. One interpretable knob (``alpha``),
  but the normalization is query-dependent: adding an outlier chunk to the
  corpus can rescale every score.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .chunking import Chunk
from .store import SearchResult


def _minmax(scores: np.ndarray) -> np.ndarray:
    lo, hi = float(scores.min()), float(scores.max())
    if hi - lo < 1e-9:
        return np.zeros_like(scores)
    return (scores - lo) / (hi - lo)


def rrf_fuse(rankings: list[np.ndarray], k: int = 60) -> np.ndarray:
    """Reciprocal Rank Fusion.

    ``rankings``: list of index arrays, each sorted best-first (e.g. the
    argsort of a score array). Returns a fused score per corpus index:
    ``sum(1 / (k + rank))`` over every ranking the chunk appears in.
    Higher is better. ``k`` dampens the influence of very top ranks;
    60 is the literature default and works fine here.
    """
    n = max((int(r.max()) for r in rankings if len(r)), default=-1) + 1
    fused = np.zeros(n, dtype=np.float64)
    for ranking in rankings:
        for rank, idx in enumerate(ranking):
            fused[int(idx)] += 1.0 / (k + rank + 1)
    return fused


def weighted_fuse(dense_scores: np.ndarray, sparse_scores: np.ndarray,
                  alpha: float = 0.5) -> np.ndarray:
    """Convex combination of min-max normalized scores.

    ``alpha`` is the weight on dense: 1.0 = pure dense, 0.0 = pure sparse.
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    return alpha * _minmax(dense_scores) + (1.0 - alpha) * _minmax(sparse_scores)


def fuse_scores(dense_scores: np.ndarray, sparse_scores: np.ndarray,
                method: str = "rrf", alpha: float = 0.5,
                rrf_k: int = 60) -> np.ndarray:
    """Fuse two per-chunk score arrays into one. Higher is better."""
    if method == "rrf":
        rankings = [np.argsort(s)[::-1] for s in (dense_scores, sparse_scores)]
        return rrf_fuse(rankings, k=rrf_k)
    if method == "weighted":
        return weighted_fuse(dense_scores, sparse_scores, alpha=alpha)
    raise ValueError(f"Unknown fusion method {method!r}; use 'rrf' or 'weighted'")


class HybridVectorStore:
    """One chunk list, two embedding matrices (sparse + dense)."""

    def __init__(self) -> None:
        self.chunks: list[Chunk] = []
        self._sparse: np.ndarray | None = None
        self._dense: np.ndarray | None = None

    def __len__(self) -> int:
        return len(self.chunks)

    def add(self, chunks: list[Chunk], sparse_vectors: np.ndarray,
            dense_vectors: np.ndarray) -> None:
        """Add chunks with their ``(n, dim)`` sparse and dense matrices."""
        sparse_vectors = np.asarray(sparse_vectors, dtype=np.float32)
        dense_vectors = np.asarray(dense_vectors, dtype=np.float32)
        if not (sparse_vectors.shape[0] == dense_vectors.shape[0] == len(chunks)):
            raise ValueError("chunks, sparse and dense vectors must have equal length")
        if self._sparse is None:
            self._sparse, self._dense = sparse_vectors, dense_vectors
        else:
            if sparse_vectors.shape[1] != self._sparse.shape[1]:
                raise ValueError("sparse dimension mismatch with existing index")
            if dense_vectors.shape[1] != self._dense.shape[1]:
                raise ValueError("dense dimension mismatch with existing index")
            self._sparse = np.vstack([self._sparse, sparse_vectors])
            self._dense = np.vstack([self._dense, dense_vectors])
        self.chunks.extend(chunks)

    def search(self, query_sparse: np.ndarray, query_dense: np.ndarray,
               top_k: int = 5, method: str = "rrf", alpha: float = 0.5,
               rrf_k: int = 60) -> list[SearchResult]:
        """Rank chunks by fused sparse+dense score. Higher is better."""
        if self._sparse is None or not self.chunks:
            return []
        qs = np.asarray(query_sparse, dtype=np.float32).ravel()
        qd = np.asarray(query_dense, dtype=np.float32).ravel()
        sparse_scores = self._sparse @ qs  # cosine sims; both sides normalized
        dense_scores = self._dense @ qd
        fused = fuse_scores(dense_scores, sparse_scores, method=method,
                            alpha=alpha, rrf_k=rrf_k)
        k = min(top_k, len(fused))
        top_idx = np.argsort(fused)[::-1][:k]
        return [SearchResult(chunk=self.chunks[i], score=float(fused[i]))
                for i in top_idx]

    # -- persistence ------------------------------------------------------
    def save(self, directory: str | Path) -> Path:
        """Persist chunks (JSON) + both matrices (NPZ) into ``directory``."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "chunks.json").write_text(
            json.dumps([c.to_dict() for c in self.chunks], indent=1),
            encoding="utf-8",
        )
        empty = np.zeros((0, 0), dtype=np.float32)
        np.savez_compressed(directory / "sparse.npz",
                            matrix=self._sparse if self._sparse is not None else empty)
        np.savez_compressed(directory / "dense.npz",
                            matrix=self._dense if self._dense is not None else empty)
        return directory

    @classmethod
    def load(cls, directory: str | Path) -> "HybridVectorStore":
        """Load a store previously written with :meth:`save`."""
        directory = Path(directory)
        store = cls()
        raw = json.loads((directory / "chunks.json").read_text(encoding="utf-8"))
        store.chunks = [Chunk.from_dict(d) for d in raw]
        if store.chunks:
            store._sparse = np.load(directory / "sparse.npz")["matrix"].astype(np.float32)
            store._dense = np.load(directory / "dense.npz")["matrix"].astype(np.float32)
        return store
