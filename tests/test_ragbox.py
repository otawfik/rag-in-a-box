"""Test suite for RAG-in-a-Box. Run with:  python -m unittest discover -s tests -v"""

import unittest
from pathlib import Path

from ragbox.chunking import Chunk, chunk_documents, chunk_text, load_markdown, split_sentences
from ragbox.embeddings import TfidfEmbedder, get_embedder
from ragbox.store import VectorStore
from ragbox.pipeline import RAGPipeline

REPO_ROOT = Path(__file__).resolve().parent.parent

TINY_DOCS = [
    {"doc_id": "cats", "title": "Cats", "section": "Cats",
     "text": ("Cats are small domesticated mammals. " * 20).strip(),
     "source": "test"},
    {"doc_id": "rockets", "title": "Rockets", "section": "Rockets",
     "text": ("Rockets use controlled explosions for thrust. " * 20).strip(),
     "source": "test"},
]


class TestChunking(unittest.TestCase):
    def test_respects_chunk_size(self):
        text = " ".join(f"word{i}" for i in range(500))
        chunks = chunk_text(text, doc_id="d", chunk_size=200, overlap=40)
        self.assertGreater(len(chunks), 1)
        for c in chunks:
            self.assertLessEqual(len(c.text), 220)  # small slack for word boundaries

    def test_overlap_repeats_content(self):
        text = "alpha beta gamma delta epsilon zeta eta theta iota kappa " * 30
        chunks = chunk_text(text, doc_id="d", chunk_size=120, overlap=60, strategy="fixed")
        self.assertGreater(len(chunks), 1)
        words_first = set(chunks[0].text.split())
        words_second = set(chunks[1].text.split())
        self.assertTrue(words_first & words_second, "expected overlapping words")

    def test_sentence_strategy_keeps_sentences_whole(self):
        text = ("The sky is blue. " * 10 + "Rockets fly high. " * 10).strip()
        chunks = chunk_text(text, doc_id="d", chunk_size=120, overlap=0, strategy="sentence")
        for c in chunks:
            for sent in split_sentences(c.text):
                self.assertTrue(sent.rstrip().endswith((".", "!", "?")))

    def test_unknown_strategy_raises(self):
        with self.assertRaises(ValueError):
            chunk_text("hello", doc_id="d", strategy="nope")

    def test_chunk_ids_unique(self):
        chunks = chunk_text("lorem ipsum dolor sit amet " * 50, doc_id="doc")
        ids = [c.chunk_id for c in chunks]
        self.assertEqual(len(ids), len(set(ids)))

    def test_load_markdown_splits_sections(self):
        docs = load_markdown(REPO_ROOT / "corpus" / "acme-space-tours.md")
        self.assertGreaterEqual(len(docs), 4)
        self.assertTrue(all(d["section"] for d in docs))


class TestEmbeddings(unittest.TestCase):
    def test_tfidf_shape_and_norm(self):
        emb = TfidfEmbedder().fit(["the cat sat", "a rocket launched"])
        vecs = emb.embed(["the cat sat", "a rocket launched", "unseen words here"])
        self.assertEqual(vecs.shape, (3, emb.dimension))
        import numpy as np
        norms = np.linalg.norm(vecs, axis=1)
        # fitted docs are L2-normalized; unseen-only doc may be the zero vector
        self.assertAlmostEqual(norms[0], 1.0, places=5)

    def test_embed_before_fit_raises(self):
        with self.assertRaises(RuntimeError):
            TfidfEmbedder().embed(["hello"])

    def test_factory(self):
        self.assertIsInstance(get_embedder("tfidf"), TfidfEmbedder)
        with self.assertRaises(ValueError):
            get_embedder("mystery")


class TestVectorStore(unittest.TestCase):
    def _indexed(self):
        emb = TfidfEmbedder().fit([d["text"] for d in TINY_DOCS])
        chunks = chunk_documents(TINY_DOCS, chunk_size=400, overlap=80)
        store = VectorStore()
        store.add(chunks, emb.embed([c.text for c in chunks]))
        return store, emb

    def test_search_ranks_relevant_first(self):
        store, emb = self._indexed()
        results = store.search(emb.embed(["tell me about cats"])[0], top_k=3)
        self.assertTrue(results)
        self.assertIn("cats", results[0].chunk.doc_id)
        self.assertGreaterEqual(results[0].score, results[-1].score)

    def test_empty_store_search(self):
        self.assertEqual(VectorStore().search(__import__("numpy").zeros(10), top_k=3), [])

    def test_save_load_roundtrip(self):
        import tempfile
        store, emb = self._indexed()
        with tempfile.TemporaryDirectory() as tmp:
            store.save(tmp)
            loaded = VectorStore.load(tmp)
        self.assertEqual(len(loaded), len(store))
        r1 = store.search(emb.embed(["rockets"])[0], top_k=1)[0]
        r2 = loaded.search(emb.embed(["rockets"])[0], top_k=1)[0]
        self.assertEqual(r1.chunk.text, r2.chunk.text)


class TestPipeline(unittest.TestCase):
    def test_answer_has_citations(self):
        pipe = RAGPipeline()
        pipe.index_documents(TINY_DOCS)
        ans = pipe.answer("What are cats?")
        self.assertTrue(ans.has_answer)
        self.assertIn("[1]", ans.text)
        self.assertTrue(all(c.section for c in ans.citations))

    def test_no_answer_for_gibberish(self):
        pipe = RAGPipeline()
        pipe.index_documents(TINY_DOCS)
        ans = pipe.answer("xqzv blorpt wumpus zzzq")
        self.assertFalse(ans.has_answer)

    def test_demo_corpus_end_to_end(self):
        pipe = RAGPipeline()
        n = pipe.index_directory(REPO_ROOT / "corpus")
        self.assertGreater(n, 5)
        ans = pipe.answer("How much does a trip to Mars cost?")
        self.assertTrue(ans.has_answer)
        self.assertIn("[1]", ans.text)
        d = ans.to_dict()
        self.assertIn("citations", d)


if __name__ == "__main__":
    unittest.main()
