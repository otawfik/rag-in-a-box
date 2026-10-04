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


_FAKE_10K_HTML = """<html><head><title>10-K</title>
<script>var xbrl = 1;</script>
<style>.hidden{display:none}</style></head>
<body>
<div style="display:none">us-gaap:Revenue 12345 hidden xbrl junk</div>
<p>APPLE INC.</p>
<p>ITEM 1. Business</p>
<p>Apple designs a wide variety of consumer electronic devices.</p>
<p>42</p>
<p>ITEM 1A. Risk Factors</p>
<p>Our business is subject to intense competition.</p>
</body></html>"""


class TestEdgarCleaning(unittest.TestCase):
    def test_html_to_text_drops_markup_and_xbrl(self):
        from ragbox.edgar import html_to_text
        text = html_to_text(_FAKE_10K_HTML)
        self.assertIn("Apple designs a wide variety", text)
        self.assertNotIn("xbrl", text.lower())
        self.assertNotIn("var xbrl", text)
        self.assertNotIn("display:none", text)

    def test_html_to_text_drops_page_numbers(self):
        from ragbox.edgar import html_to_text
        text = html_to_text("<html><body><p>42</p><p>Real content here.</p></body></html>")
        self.assertNotIn("\n42\n", f"\n{text}\n")
        self.assertIn("Real content here.", text)

    def test_split_10k_items(self):
        from ragbox.chunking import split_10k_items
        sections = split_10k_items(
            "Cover page text here.\nITEM 1. Business\nBody one.\n"
            "ITEM 1A. Risk Factors\nBody two.\n"
        )
        titles = [t for t, _ in sections]
        self.assertIn("Cover", titles)
        self.assertTrue(any(t.startswith("ITEM 1.") for t in titles))
        self.assertTrue(any(t.startswith("ITEM 1A") for t in titles))
        body = dict(sections)["ITEM 1. Business"]
        self.assertIn("Body one.", body)

    def test_split_10k_items_no_headers(self):
        from ragbox.chunking import split_10k_items
        sections = split_10k_items("Just some plain text.")
        self.assertEqual(len(sections), 1)
        self.assertEqual(sections[0][0], "Full filing")

    def test_split_10k_items_dedupes_toc_phantoms(self):
        from ragbox.chunking import split_10k_items
        text = ("ITEM 1. Business\nITEM 1A. Risk Factors\n"  # table of contents
                "ITEM 1. Business\n" + "Real business body. " * 50 + "\n"
                "ITEM 1A. Risk Factors\n" + "Real risks body. " * 50 + "\n")
        sections = split_10k_items(text)
        titles = [t for t, _ in sections]
        self.assertEqual(len(sections), 2)
        self.assertTrue(any(t.startswith("ITEM 1.") for t in titles))
        body = dict(sections)[titles[0]]
        self.assertIn("Real business body.", body)

    def test_load_text_sections(self):
        import tempfile
        from ragbox.chunking import load_text
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as fh:
            fh.write("ITEM 7. Management Discussion\nMD&A body text here.\n")
            tmp = fh.name
        try:
            docs = load_text(tmp, doc_id="TEST_10K")
            self.assertEqual(len(docs), 1)
            self.assertTrue(docs[0]["section"].startswith("ITEM 7"))
            self.assertIn("MD&A body", docs[0]["text"])
        finally:
            Path(tmp).unlink()


class TestHybridFusion(unittest.TestCase):
    """Score-fusion math needs no neural models; test it directly."""

    def test_rrf_prefers_consensus(self):
        import numpy as np
        from ragbox.hybrid import rrf_fuse
        # chunk 1 is ranked 1st by dense but absent from sparse;
        # chunk 0 appears in both rankings -> consensus wins
        fused = rrf_fuse([np.array([1, 0]), np.array([0])], k=60)
        self.assertGreater(fused[0], fused[1])

    def test_rrf_ignores_score_scale(self):
        import numpy as np
        from ragbox.hybrid import fuse_scores
        dense = np.array([1000.0, 0.0, 0.0])   # wildly different scale
        sparse = np.array([0.9, 0.0, 0.0])
        fused = fuse_scores(dense, sparse, method="rrf")
        self.assertEqual(int(np.argmax(fused)), 0)

    def test_weighted_alpha_extremes(self):
        import numpy as np
        from ragbox.hybrid import fuse_scores
        dense = np.array([0.1, 0.9])
        sparse = np.array([0.9, 0.1])
        self.assertEqual(int(np.argmax(fuse_scores(dense, sparse, method="weighted", alpha=1.0))), 1)
        self.assertEqual(int(np.argmax(fuse_scores(dense, sparse, method="weighted", alpha=0.0))), 0)

    def test_weighted_rejects_bad_alpha(self):
        import numpy as np
        from ragbox.hybrid import weighted_fuse
        with self.assertRaises(ValueError):
            weighted_fuse(np.array([0.5]), np.array([0.5]), alpha=1.5)

    def test_hybrid_store_end_to_end_tfidf_only(self):
        # HybridVectorStore works with plain TF-IDF vectors on both sides;
        # exercises add/search/save/load without needing torch.
        import tempfile
        import numpy as np
        from ragbox.chunking import Chunk
        from ragbox.embeddings import TfidfEmbedder
        from ragbox.hybrid import HybridVectorStore
        texts = ["apple iphone revenue growth", "microsoft azure cloud services",
                 "nvidia gpu data center chips"]
        chunks = [Chunk(text=t, doc_id=f"d{i}", chunk_id=f"d{i}#c0") for i, t in enumerate(texts)]
        emb = TfidfEmbedder().fit(texts)
        vecs = emb.embed(texts)
        store = HybridVectorStore()
        store.add(chunks, vecs, vecs)
        qv = emb.embed(["cloud revenue"])[0]
        for method in ("rrf", "weighted"):
            res = store.search(qv, qv, top_k=2, method=method)
            self.assertEqual(len(res), 2)
            self.assertTrue(all(r.score >= 0 for r in res))
        with tempfile.TemporaryDirectory() as tmp:
            store.save(tmp)
            loaded = HybridVectorStore.load(tmp)
            self.assertEqual(len(loaded), 3)
            res = loaded.search(qv, qv, top_k=1, method="rrf")
            self.assertEqual(len(res), 1)

    def test_pipeline_rejects_bad_retrieval(self):
        from ragbox.pipeline import RAGPipeline
        with self.assertRaises(ValueError):
            RAGPipeline(retrieval="quantum")


if __name__ == "__main__":
    unittest.main()
