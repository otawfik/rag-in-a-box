"""In-memory vector store with cosine-similarity search and persistence.

Vectors are expected to be L2-normalized (guaranteed by every
:class:`~ragbox.embeddings.Embedder`), so ranking by dot product is exactly
ranking by cosine similarity — no distance library required.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .chunking import Chunk


@dataclass
class SearchResult:
    chunk: Chunk
    score: float  # cosine similarity in [-1, 1]


class VectorStore:
    """Append-only store of chunks + their embedding matrix."""

    def __init__(self) -> None:
        self.chunks: list[Chunk] = []
        self._matrix: np.ndarray | None = None

    def __len__(self) -> int:
        return len(self.chunks)

    def add(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        """Add chunks with their precomputed ``(n, dim)`` embedding matrix."""
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.shape[0] != len(chunks):
            raise ValueError("chunks and vectors must have the same length")
        if self._matrix is None:
            self._matrix = vectors
        else:
            if vectors.shape[1] != self._matrix.shape[1]:
                raise ValueError("embedding dimension mismatch with existing index")
            self._matrix = np.vstack([self._matrix, vectors])
        self.chunks.extend(chunks)

    def search(self, query_vector: np.ndarray, top_k: int = 5) -> list[SearchResult]:
        """Return the ``top_k`` chunks ranked by cosine similarity."""
        if self._matrix is None or not self.chunks:
            return []
        q = np.asarray(query_vector, dtype=np.float32).ravel()
        scores = self._matrix @ q  # cosine sim: both sides L2-normalized
        k = min(top_k, len(scores))
        top_idx = np.argsort(scores)[::-1][:k]
        return [SearchResult(chunk=self.chunks[i], score=float(scores[i]))
                for i in top_idx]

    # -- persistence ------------------------------------------------------
    def save(self, directory: str | Path) -> Path:
        """Persist chunks (JSON) + matrix (NPZ) into ``directory``."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "chunks.json").write_text(
            json.dumps([c.to_dict() for c in self.chunks], indent=1),
            encoding="utf-8",
        )
        np.savez_compressed(directory / "matrix.npz",
                            matrix=self._matrix if self._matrix is not None
                            else np.zeros((0, 0), dtype=np.float32))
        return directory

    @classmethod
    def load(cls, directory: str | Path) -> "VectorStore":
        """Load a store previously written with :meth:`save`."""
        directory = Path(directory)
        store = cls()
        raw = json.loads((directory / "chunks.json").read_text(encoding="utf-8"))
        store.chunks = [Chunk.from_dict(d) for d in raw]
        matrix = np.load(directory / "matrix.npz")["matrix"].astype(np.float32)
        store._matrix = matrix if len(store.chunks) else None
        return store
