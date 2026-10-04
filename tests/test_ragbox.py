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
            "Cover page text here, long enough to survive the phantom filter. "
            "Padding padding padding.\nITEM 1. Business\n" + "Body one. " * 20 + "\n"
            "ITEM 1A. Risk Factors\n" + "Body two. " * 20 + "\n"
        )
        titles = [t for t, _ in sections]
        self.assertIn("Cover", titles)
        self.assertTrue(any(t.startswith("ITEM 1.") for t in titles))
        self.assertTrue(any(t.startswith("ITEM 1A") for t in titles))
        body = dict(sections)[[t for t in titles if t.startswith("ITEM 1.")][0]]
        self.assertIn("Body one.", body)

    def test_split_10k_items_no_headers(self):
        from ragbox.chunking import split_10k_items
        sections = split_10k_items("Just some plain text.")
        self.assertEqual(len(sections), 1)
        self.assertEqual(sections[0][0], "Full filing")

    def test_split_10k_items_merges_page_header_fragments(self):
        from ragbox.chunking import split_10k_items
        text = ("ITEM 1. Business\nITEM 1A. Risk Factors\n"  # table of contents
                "ITEM 1. Business\n" + "Real business body. " * 50 + "\n"
                "ITEM 1. Business\n" + "Page-header fragment body. " * 50 + "\n"
                "ITEM 1A. Risk Factors\n" + "Real risks body. " * 50 + "\n")
        sections = split_10k_items(text)
        titles = [t for t, _ in sections]
        self.assertEqual(len(sections), 2)
        self.assertTrue(any(t.startswith("ITEM 1.") for t in titles))
        body = dict(sections)[titles[0]]
        self.assertIn("Real business body.", body)
        self.assertIn("Page-header fragment body.", body)

    def test_load_text_sections(self):
        import tempfile
        from ragbox.chunking import load_text
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as fh:
            fh.write("ITEM 7. Management Discussion\n" + "MD&A body text here. " * 20 + "\n")
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


class StubGenerator:
    """Stands in for Generator in tests: no model download."""

    def __init__(self, text="Stub answer [1]", refused=False, judge_answer=True):
        self.text = text
        self.refused = refused
        self.judge_answer = judge_answer
        self.judge_calls = 0
        self.generate_calls = 0
        self.last_passages = None

    def judge(self, question, passages):
        self.judge_calls += 1
        return self.judge_answer

    def generate(self, question, passages, sources=None):
        from ragbox.generate import GeneratedAnswer, extract_citations
        self.generate_calls += 1
        self.last_passages = passages
        # mirror Generator.generate: best-effort markers, else all passages
        cited = extract_citations(self.text, len(passages)) or list(range(len(passages)))
        return GeneratedAnswer(
            question=question, text=self.text, refused=self.refused,
            cited=cited,
            passages=list(passages), sources=list(sources or []),
        )


class TestGeneration(unittest.TestCase):
    def test_is_refusal(self):
        from ragbox.generate import REFUSAL_TEXT, is_refusal
        self.assertTrue(is_refusal(REFUSAL_TEXT))
        self.assertTrue(is_refusal("i cannot answer this from the provided filings"))
        self.assertTrue(is_refusal("Sorry, I cannot answer that from the provided filings."))
        self.assertFalse(is_refusal("Revenue was $100 [1]."))
        self.assertFalse(is_refusal(""))

    def test_extract_citations(self):
        from ragbox.generate import extract_citations
        self.assertEqual(extract_citations("Blah [2] blah [1] blah [2].", 3), [1, 0])
        self.assertEqual(extract_citations("No citations here.", 3), [])
        # hallucinated citation [9] with only 2 passages: dropped
        self.assertEqual(extract_citations("Blah [9] blah [1].", 2), [0])

    def test_build_prompt_numbers_passages(self):
        from ragbox.generate import build_prompt
        msgs = build_prompt("Q?", ["first passage", "second passage"])
        self.assertEqual(len(msgs), 2)  # system + user; no few-shot needed
        self.assertEqual(msgs[0]["role"], "system")
        self.assertIn("[1]\nfirst passage", msgs[1]["content"])
        self.assertIn("[2]\nsecond passage", msgs[1]["content"])

    def test_build_judge_prompt(self):
        from ragbox.generate import build_judge_prompt
        msgs = build_judge_prompt("Q?", ["a passage"])
        self.assertEqual(len(msgs), 1)
        self.assertIn("YES or NO", msgs[0]["content"])
        self.assertIn("[1]\na passage", msgs[0]["content"])

    def test_layer1_refusal_never_calls_llm(self):
        from ragbox.pipeline import RAGPipeline
        pipe = RAGPipeline()
        pipe.index_documents(TINY_DOCS)
        stub = StubGenerator()
        ans = pipe.generate("xqzv blorpt wumpus zzzq", generator=stub)
        self.assertTrue(ans.refused)
        self.assertEqual(stub.judge_calls, 0)     # judge never invoked
        self.assertEqual(stub.generate_calls, 0)  # LLM never invoked

    def test_layer2_judge_no_refuses(self):
        from ragbox.pipeline import RAGPipeline
        pipe = RAGPipeline()
        pipe.index_documents(TINY_DOCS)
        stub = StubGenerator(judge_answer=False)
        ans = pipe.generate("What are cats?", generator=stub)
        self.assertTrue(ans.refused)
        self.assertEqual(stub.judge_calls, 1)
        self.assertEqual(stub.generate_calls, 0)  # no generation after NO

    def test_layer2_judge_yes_generates(self):
        from ragbox.pipeline import RAGPipeline
        pipe = RAGPipeline()
        pipe.index_documents(TINY_DOCS)
        stub = StubGenerator(text="Cats are mammals [1].", judge_answer=True)
        ans = pipe.generate("What are cats?", generator=stub)
        self.assertFalse(ans.refused)
        self.assertEqual(stub.judge_calls, 1)
        self.assertEqual(stub.generate_calls, 1)
        self.assertGreater(len(stub.last_passages), 0)
        self.assertEqual(ans.cited, [0])
        self.assertTrue(ans.sources)

    def test_judge_ablation(self):
        from ragbox.pipeline import RAGPipeline
        pipe = RAGPipeline()
        pipe.index_documents(TINY_DOCS)
        stub = StubGenerator(text="Cats are mammals [1].", judge_answer=False)
        ans = pipe.generate("What are cats?", generator=stub, judge=False)
        self.assertFalse(ans.refused)  # judge skipped: generates anyway
        self.assertEqual(stub.judge_calls, 0)

    def test_citations_fall_back_to_all_passages(self):
        from ragbox.pipeline import RAGPipeline
        pipe = RAGPipeline()
        pipe.index_documents(TINY_DOCS)
        stub = StubGenerator(text="Cats are mammals.", judge_answer=True)
        ans = pipe.generate("What are cats?", generator=stub)
        self.assertFalse(ans.refused)
        # model emitted no markers: cite every passage it was conditioned on
        self.assertEqual(ans.cited, list(range(len(stub.last_passages))))
        self.assertGreater(len(ans.cited), 0)


class TestEvalSet(unittest.TestCase):

    def test_eval_set_roundtrip(self):
        import tempfile
        from ragbox.eval import EvalQuestion, load_eval_set, save_eval_set
        qs = [EvalQuestion(id="q1", question="Rev?", answer="$1B",
                           type="factual", gold_chunks=["d#s0#c0"])]
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
            tmp = fh.name
        try:
            save_eval_set(qs, tmp, meta={"chunking": {"chunk_size": 600}})
            meta2, qs2 = load_eval_set(tmp)
            self.assertEqual(len(qs2), 1)
            self.assertEqual(qs2[0].answer, "$1B")
            self.assertFalse(qs2[0].verified)
            self.assertEqual(meta2["chunking"]["chunk_size"], 600)
        finally:
            Path(tmp).unlink()

if __name__ == "__main__":
    unittest.main()
