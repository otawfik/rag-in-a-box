"""Document chunking engine.

Splits raw documents into overlapping, self-contained chunks that are small
enough for a retrieval context window but large enough to carry meaning on
their own. Three strategies are provided:

- ``"recursive"`` (default): greedily packs paragraphs, then sentences, then
  words into each chunk — the best general-purpose choice.
- ``"sentence"``: packs whole sentences; never cuts mid-sentence.
- ``"fixed"``: dumb sliding character window; useful as a baseline.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Chunk:
    """A single retrievable unit of text."""

    text: str
    doc_id: str
    chunk_id: str
    source: str = ""   # file path or origin label
    section: str = ""  # markdown section heading, when known
    start_char: int = 0

    @property
    def token_estimate(self) -> int:
        """Rough token count (~4 chars/token); good enough for sizing hints."""
        return max(1, len(self.text) // 4)

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "doc_id": self.doc_id,
            "chunk_id": self.chunk_id,
            "source": self.source,
            "section": self.section,
            "start_char": self.start_char,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Chunk":
        return cls(**{k: data[k] for k in
                      ("text", "doc_id", "chunk_id", "source", "section", "start_char")
                      if k in data})


def split_sentences(text: str) -> list[str]:
    """Split text into sentences without third-party NLP dependencies."""
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    parts = re.split(r"(?<=[.!?…])\s+(?=[A-Z0-9\"'‘“(\[])", text)
    return [p.strip() for p in parts if p.strip()]


def _pack(units: list[str], chunk_size: int, overlap: int) -> list[str]:
    """Greedily pack text units into chunks of ~chunk_size chars with overlap."""
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0
    for unit in units:
        unit = unit.strip()
        if not unit:
            continue
        # Oversized unit: split it on words rather than dropping it.
        while len(unit) > chunk_size:
            cut = unit.rfind(" ", 0, chunk_size)
            cut = cut if cut > 0 else chunk_size
            _flush_piece(chunks, current, unit[:cut].strip(), chunk_size, overlap)
            current, current_len = [], 0
            unit = unit[cut:].strip()
        if current and current_len + 1 + len(unit) > chunk_size:
            _flush_piece(chunks, current, None, chunk_size, overlap)
            # Overlap: carry trailing units that fit inside `overlap` chars.
            carried: list[str] = []
            carried_len = 0
            for u in reversed(current):
                if carried_len + len(u) + 1 > overlap:
                    break
                carried.insert(0, u)
                carried_len += len(u) + 1
            current, current_len = carried, carried_len
        current.append(unit)
        current_len += len(unit) + 1
    if current:
        chunks.append(" ".join(current).strip())
    return [c for c in chunks if c]


def _flush_piece(chunks: list[str], current: list[str], extra: str | None,
                 chunk_size: int, overlap: int) -> None:
    pieces = list(current)
    if extra:
        pieces.append(extra)
    text = " ".join(pieces).strip()
    if text:
        chunks.append(text)


def _chunk_recursive(text: str, chunk_size: int, overlap: int) -> list[str]:
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    units: list[str] = []
    for para in paragraphs:
        if len(para) <= chunk_size:
            units.append(para)
        else:
            units.extend(split_sentences(para))
    return _pack(units, chunk_size, overlap)


def _chunk_sentence(text: str, chunk_size: int, overlap: int) -> list[str]:
    return _pack(split_sentences(text), chunk_size, overlap)


def _chunk_fixed(text: str, chunk_size: int, overlap: int) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    chunks: list[str] = []
    start = 0
    step = max(1, chunk_size - overlap)
    while start < len(text):
        end = min(len(text), start + chunk_size)
        # Avoid cutting a word in half when we can help it.
        if end < len(text):
            cut = text.rfind(" ", start, end)
            if cut > start:
                end = cut
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= len(text):
            break
        start += step
    return chunks


_STRATEGIES = {
    "recursive": _chunk_recursive,
    "sentence": _chunk_sentence,
    "fixed": _chunk_fixed,
}


def chunk_text(
    text: str,
    doc_id: str,
    source: str = "",
    section: str = "",
    chunk_size: int = 600,
    overlap: int = 120,
    strategy: str = "recursive",
) -> list[Chunk]:
    """Split ``text`` into a list of :class:`Chunk` objects.

    ``chunk_size``/``overlap`` are in characters. ``overlap`` must be smaller
    than ``chunk_size``.
    """
    if strategy not in _STRATEGIES:
        raise ValueError(f"Unknown strategy {strategy!r}; choose from {sorted(_STRATEGIES)}")
    if not 0 <= overlap < chunk_size:
        raise ValueError("overlap must satisfy 0 <= overlap < chunk_size")
    pieces = _STRATEGIES[strategy](text, chunk_size, overlap)
    chunks = []
    cursor = 0
    for i, piece in enumerate(pieces):
        start = text.find(piece[:40], cursor)
        start = start if start != -1 else cursor
        chunks.append(Chunk(
            text=piece,
            doc_id=doc_id,
            chunk_id=f"{doc_id}#c{i}",
            source=source,
            section=section,
            start_char=start,
        ))
        cursor = start + 1
    return chunks


def load_markdown(path: str | Path, doc_id: str | None = None) -> list[dict]:
    """Load a markdown file, splitting it into one document per ``##`` section.

    Returns a list of dicts with keys ``doc_id``, ``title``, ``section``,
    ``text`` and ``source`` — ready for :func:`chunk_documents`.
    """
    path = Path(path)
    doc_id = doc_id or path.stem
    raw = path.read_text(encoding="utf-8")
    sections: list[tuple[str, list[str]]] = []
    current_title = ""
    current_lines: list[str] = []
    for line in raw.splitlines():
        m = re.match(r"^(#{1,3})\s+(.*)", line)
        if m and len(m.group(1)) >= 2:  # ## or ### starts a new section
            if current_lines or current_title:
                sections.append((current_title, current_lines))
            current_title = m.group(2).strip()
            current_lines = []
        elif m:  # single # is the document title
            title_line = m.group(2).strip()
            if not any(s for s, _ in sections) and not current_title:
                current_title = title_line
        else:
            current_lines.append(line)
    if current_lines or current_title:
        sections.append((current_title, current_lines))
    docs = []
    for idx, (title, lines) in enumerate(sections):
        text = "\n".join(lines).strip()
        if not text:
            continue
        docs.append({
            "doc_id": f"{doc_id}#s{idx}",
            "title": title or doc_id,
            "section": title,
            "text": text,
            "source": str(path),
        })
    return docs


def chunk_documents(
    docs: list[dict],
    chunk_size: int = 600,
    overlap: int = 120,
    strategy: str = "recursive",
) -> list[Chunk]:
    """Chunk a list of document dicts (as produced by :func:`load_markdown`)."""
    chunks: list[Chunk] = []
    for doc in docs:
        chunks.extend(chunk_text(
            doc["text"],
            doc_id=doc["doc_id"],
            source=doc.get("source", ""),
            section=doc.get("section", ""),
            chunk_size=chunk_size,
            overlap=overlap,
            strategy=strategy,
        ))
    return chunks
