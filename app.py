"""RAG-in-a-Box web UI and CLI.

Web chat UI::

    python app.py                 # serve on http://127.0.0.1:5000

Command line::

    python app.py --ask "How much is a Mars trip?"
    python app.py --reindex       # rebuild the cached index
"""

from __future__ import annotations

import argparse
from pathlib import Path

from flask import Flask, jsonify, request

from ragbox.pipeline import RAGPipeline

BASE_DIR = Path(__file__).resolve().parent
CORPUS_DIR = BASE_DIR / "corpus"
INDEX_DIR = BASE_DIR / ".ragbox_index"

app = Flask(__name__)
_pipeline: RAGPipeline | None = None


def get_pipeline() -> RAGPipeline:
    """Build (or load the cached) index exactly once per process."""
    global _pipeline
    if _pipeline is None:
        if INDEX_DIR.exists():
            print(f"Loading cached index from {INDEX_DIR}")
            _pipeline = RAGPipeline.load(INDEX_DIR)
        else:
            print(f"Building index from {CORPUS_DIR} ...")
            _pipeline = RAGPipeline()
            n = _pipeline.index_directory(CORPUS_DIR)
            _pipeline.save(INDEX_DIR)
            print(f"Indexed {n} chunks; cached to {INDEX_DIR}")
    return _pipeline


HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>RAG-in-a-Box 📦</title>
<style>
  :root { --bg:#0f1420; --panel:#182030; --accent:#6ea8fe; --text:#e8ecf4; --muted:#8b94a7; }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--bg); color:var(--text);
         font-family:system-ui,-apple-system,"Segoe UI",sans-serif; }
  .wrap { max-width:760px; margin:0 auto; padding:32px 20px 120px; }
  h1 { font-size:1.6rem; } h1 small { color:var(--muted); font-weight:normal; }
  #log { display:flex; flex-direction:column; gap:14px; margin-top:24px; }
  .msg { background:var(--panel); border-radius:12px; padding:14px 16px; line-height:1.55; }
  .msg.q { align-self:flex-end; background:#24406e; max-width:85%; }
  .msg.a { align-self:flex-start; width:100%; }
  .msg.a pre { white-space:pre-wrap; font-family:inherit; margin:0; }
  .src { color:var(--muted); font-size:.82rem; margin-top:8px; }
  .meta { color:var(--muted); font-size:.75rem; margin-top:6px; }
  #bar { position:fixed; left:0; right:0; bottom:0; background:#0b0f18;
         border-top:1px solid #223; padding:14px; }
  #bar .inner { max-width:760px; margin:0 auto; display:flex; gap:10px; }
  input { flex:1; padding:12px 14px; border-radius:10px; border:1px solid #334;
          background:var(--panel); color:var(--text); font-size:1rem; }
  button { padding:12px 20px; border-radius:10px; border:0; background:var(--accent);
           color:#0b0f18; font-weight:700; font-size:1rem; cursor:pointer; }
  .hint { color:var(--muted); font-size:.9rem; }
  .hint code { background:var(--panel); padding:2px 6px; border-radius:6px; }
</style>
</head>
<body>
<div class="wrap">
  <h1>📦 RAG-in-a-Box <small>— retrieval chatbot with cited answers</small></h1>
  <p class="hint">Demo corpus: the <b>Acme Space Tours</b> passenger guide.
     Try: <code>How much does a Mars trip cost?</code>
     <code>Which ship flies to Europa?</code>
     <code>Can I bring my dog?</code></p>
  <div id="log"></div>
</div>
<div id="bar"><div class="inner">
  <input id="q" placeholder="Ask about space tours…" autocomplete="off">
  <button onclick="ask()">Ask</button>
</div></div>
<script>
const log = document.getElementById('log'), q = document.getElementById('q');
q.addEventListener('keydown', e => { if (e.key === 'Enter') ask(); });
async function ask() {
  const question = q.value.trim();
  if (!question) return;
  q.value = '';
  log.insertAdjacentHTML('beforeend', `<div class="msg q">${escapeHtml(question)}</div>`);
  const div = document.createElement('div');
  div.className = 'msg a'; div.innerHTML = '<pre>Thinking…</pre>';
  log.appendChild(div); div.scrollIntoView({behavior:'smooth'});
  try {
    const r = await fetch('/api/ask', {method:'POST',
      headers:{'Content-Type':'application/json'}, body:JSON.stringify({question})});
    const d = await r.json();
    let html = `<pre>${escapeHtml(d.answer)}</pre>`;
    html += `<div class="meta">answered in ${d.latency_ms} ms · ${d.citations.length} citation(s)</div>`;
    div.innerHTML = html;
  } catch (e) { div.innerHTML = '<pre>Something went wrong talking to the server.</pre>'; }
  div.scrollIntoView({behavior:'smooth'});
}
function escapeHtml(s){ return s.replace(/[&<>"']/g, c =>
  ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c])); }
</script>
</body>
</html>
"""


@app.get("/")
def index():
    return HTML


@app.get("/api/health")
def health():
    pipe = get_pipeline()
    return jsonify({"status": "ok", "chunks": len(pipe.store)})


@app.post("/api/ask")
def api_ask():
    data = request.get_json(force=True, silent=True) or {}
    question = (data.get("question") or "").strip()
    if not question:
        return jsonify({"error": "missing 'question'"}), 400
    return jsonify(get_pipeline().answer(question).to_dict())


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG-in-a-Box: retrieval chatbot")
    parser.add_argument("--ask", metavar="QUESTION", help="answer one question and exit")
    parser.add_argument("--reindex", action="store_true", help="rebuild the cached index")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()

    if args.reindex:
        import shutil
        shutil.rmtree(INDEX_DIR, ignore_errors=True)
        get_pipeline()
        print("Index rebuilt.")
        return

    if args.ask:
        pipe = get_pipeline()
        ans = pipe.answer(args.ask)
        print(f"Q: {args.ask}\n")
        print(ans.text)
        print(f"\n({ans.latency_ms:.0f} ms)")
        return

    print("Serving RAG-in-a-Box at http://%s:%d" % (args.host, args.port))
    app.run(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
