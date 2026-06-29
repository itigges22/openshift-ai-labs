#!/usr/bin/env python3
"""
AutoRAG playground — a tiny local web UI over the real pipeline.

Pure standard library (http.server). Nothing to install. It calls the exact same
retrieval + generation code as the CLI, against your live model backend, so the
answers you see in the browser are a real 1B model on real retrieved context.

  python3 serve.py            # http://localhost:7700
  PORT=8800 python3 serve.py

Backend is selected the same way as the CLI (env vars: BACKEND / CHAT_MODEL / ...).
"""

import json
import os
import re
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import model_client as mc
from rag.chunk import load_corpus, build_chunks
from rag.pipeline import build_store, retrieve, answer
from eval import score as S

HERE = os.path.dirname(os.path.abspath(__file__))
CORPUS_DIR = os.path.join(HERE, "corpus")
EVAL_PATH = os.path.join(HERE, "eval", "qa.jsonl")
BEST_PATH = os.path.join(HERE, "best_config.json")
PORT = int(os.environ.get("PORT", "7700"))

BASELINE = {"chunk_size": 250, "chunk_overlap": 0, "top_k": 8, "retriever": "dense"}

DOCS = load_corpus(CORPUS_DIR)
_store_cache = {}      # (size, overlap) -> VectorStore
_qvec_cache = {}       # text -> vector
_doc_cache = {}        # (hash(text), size, overlap) -> VectorStore  (user-imported docs)


def get_store(size, overlap):
    key = (size, overlap)
    if key not in _store_cache:
        chunks = build_chunks(DOCS, size, overlap)
        vecs = mc.embed([c["text"] for c in chunks])
        _store_cache[key] = build_store(chunks, vecs)
    return _store_cache[key]


def qvec(text):
    if text not in _qvec_cache:
        _qvec_cache[text] = mc.embed(text)[0]
    return _qvec_cache[text]


def load_eval():
    items = []
    with open(EVAL_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def winner_config():
    if os.path.exists(BEST_PATH):
        try:
            return json.load(open(BEST_PATH))["config"]
        except Exception:
            pass
    return {"chunk_size": 150, "chunk_overlap": 30, "top_k": 3, "retriever": "dense"}


def clean_config(c):
    return {
        "chunk_size": max(10, min(400, int(c.get("chunk_size", 150)))),
        "chunk_overlap": max(0, min(200, int(c.get("chunk_overlap", 30)))),
        "top_k": max(1, min(12, int(c.get("top_k", 3)))),
        "retriever": c.get("retriever") if c.get("retriever") in ("dense", "lexical", "hybrid") else "dense",
    }


# --------------------------------------------------------------------------- #
# Actions
# --------------------------------------------------------------------------- #
def do_ask(config, question):
    config = clean_config(config)
    store = get_store(config["chunk_size"], config["chunk_overlap"])
    t0 = time.time()
    text, retrieved = answer(store, question, qvec(question), config)
    dt = time.time() - t0
    return {
        "answer": text,
        "config": config,
        "latency_s": round(dt, 2),
        "ctx_words": sum(len(c["text"].split()) for c in retrieved),
        "retrieved": [
            {"id": c["id"], "doc": c["doc"], "text": c["text"][:400]} for c in retrieved
        ],
    }


# Numeric "facts" — dollar amounts, day/cycle windows, percentages, reason codes.
# These are exactly what collide across the policy docs ($0 vs $50, 60 vs 90 days),
# so they are the things a noisy context makes a small model get wrong.
_FACT_PATTERNS = [
    r"\$[\d,]+(?:\.\d+)?",
    r"\b\d+\s*business\s+days?\b",
    r"\b\d+\s*days?\b",
    r"\b\d+\s*(?:complete\s+)?(?:billing\s+)?cycles?\b",
    r"\b\d+\s*%",
    r"\b\d+\.\d+(?:\.\d+)?\b",
]


def extract_facts(text):
    t = text.lower()
    found = []
    for p in _FACT_PATTERNS:
        for m in re.findall(p, t):
            f = re.sub(r"\s+", " ", m).strip()
            if f not in found:
                found.append(f)
    return found


def find_eval(question):
    q = question.strip().lower()
    for it in load_eval():
        if it["q"].strip().lower() == q:
            return it
    return None


def run_one_store(store, config, question, gold=None, reference=None):
    """Analyze one config's answer on a given store. gold/reference are optional,
    so this serves both graded corpus questions and ad-hoc user documents."""
    t0 = time.time()
    text, retrieved = answer(store, question, qvec(question), config)
    dt = time.time() - t0

    gold_facts = extract_facts(reference) if reference else []
    answer_rank = 0
    chunks = []
    for i, c in enumerate(retrieved, 1):
        has_gold = bool(gold) and S._norm(gold) in S._norm(c["text"])
        if has_gold and not answer_rank:
            answer_rank = i
        cfacts = extract_facts(c["text"])
        chunks.append({
            "doc": c["doc"], "text": c["text"][:500], "words": len(c["text"].split()),
            "has_gold": has_gold,
            "distractors": [f for f in cfacts if f not in gold_facts],
        })

    ctx_facts = extract_facts(" ".join(c["text"] for c in retrieved))
    distractors = [f for f in ctx_facts if f not in gold_facts]
    ans_facts = extract_facts(text)
    has_gold_fact = bool(gold) and S._norm(gold) in S._norm(text)
    correct = (has_gold_fact or (bool(gold_facts) and all(g in ans_facts for g in gold_facts))) \
        if gold else None

    sc = None
    if reference:
        rv, cv = mc.embed([reference, text])
        sc = round(S.answer_score(reference, text, rv, cv), 3)

    return {
        "config": config, "answer": text, "latency_s": round(dt, 2),
        "ctx_words": sum(c["words"] for c in chunks), "n_chunks": len(chunks),
        "answer_rank": answer_rank, "chunks": chunks,
        "distractors": distractors, "n_distractors": len(distractors),
        "answer_facts": ans_facts, "gold_facts": gold_facts,
        "wrong_facts": [f for f in ans_facts if f not in gold_facts],
        "correct": correct, "score": sc,
    }


def run_one(config, question, item):
    config = clean_config(config)
    store = get_store(config["chunk_size"], config["chunk_overlap"])
    gold = item["gold"] if item else None
    reference = item["answer"] if item else None
    return run_one_store(store, config, question, gold, reference)


def _why(base, win, cut, gold=None, gold_label="the policy fact", gold_value=None):
    why = [
        f"Naive pulled {base['n_chunks']} chunks ({base['ctx_words']} words); "
        f"AutoRAG pulled {win['n_chunks']} ({win['ctx_words']} words) — {cut}% less "
        f"text for the model to read."
    ]
    if base["answer_rank"] and win["answer_rank"]:
        why.append(
            f"The passage that actually answers the question ranked "
            f"#{base['answer_rank']} of {base['n_chunks']} under naive, but "
            f"#{win['answer_rank']} of {win['n_chunks']} under AutoRAG — nearer the "
            f"top, where the model pays the most attention.")
    why.append(
        f"Naive's context carried {base['n_distractors']} competing figures"
        + (f" ({', '.join(base['distractors'][:5])}…)" if base["distractors"] else "")
        + f"; AutoRAG's carried {win['n_distractors']}. Colliding numbers are exactly "
        f"what make a 1B model answer with the wrong one.")
    if gold and base["correct"] is not None and base["correct"] != win["correct"]:
        if win["correct"] and not base["correct"]:
            wf = ", ".join(base["wrong_facts"][:3])
            why.append(
                "Result: naive answered incorrectly"
                + (f" (it stated {wf}, not {gold_label})" if wf else "")
                + f", while AutoRAG stated {gold_label}"
                + (f" (\"{gold_value}\")" if gold_value else "") + ".")
    elif gold and win["correct"] and base["correct"]:
        why.append("Both happened to land the right fact here — but AutoRAG did it on a "
                   "fraction of the context, i.e. cheaper and faster.")
    return why


def do_compare_doc(text, question, gold):
    text = (text or "").strip()
    if not text:
        return {"error": "Paste or upload a document first."}
    if not (question or "").strip():
        return {"error": "Ask a question about your document."}
    words = text.split()
    truncated = len(words) > 20000
    if truncated:
        text = " ".join(words[:20000])
    gold = (gold or "").strip() or None

    def docstore(cfg):
        key = (hash(text), cfg["chunk_size"], cfg["chunk_overlap"])
        if key not in _doc_cache:
            chunks = build_chunks([{"doc": "your-document", "text": text}],
                                  cfg["chunk_size"], cfg["chunk_overlap"])
            vecs = mc.embed([c["text"] for c in chunks])
            _doc_cache[key] = build_store(chunks, vecs)
        return _doc_cache[key]

    win = clean_config(winner_config())
    base_cfg = clean_config(BASELINE)
    base = run_one_store(docstore(base_cfg), base_cfg, question, gold, gold)
    wino = run_one_store(docstore(win), win, question, gold, gold)
    cut = round((1 - wino["ctx_words"] / (base["ctx_words"] or 1)) * 100)
    why = _why(base, wino, cut, gold=gold, gold_label="your key fact", gold_value=gold)
    return {"question": question, "gold": gold, "baseline": base, "winner": wino,
            "cut": cut, "why": why, "truncated": truncated}


def do_ask_doc(text, config, question):
    text = (text or "").strip()
    if not text:
        return {"error": "Paste or upload a document first."}
    if not (question or "").strip():
        return {"error": "Ask a question about your document."}
    words = text.split()
    truncated = len(words) > 20000
    if truncated:
        text = " ".join(words[:20000])
    config = clean_config(config)
    key = (hash(text), config["chunk_size"], config["chunk_overlap"])
    if key not in _doc_cache:
        chunks = build_chunks([{"doc": "your-document", "text": text}],
                              config["chunk_size"], config["chunk_overlap"])
        vecs = mc.embed([c["text"] for c in chunks])
        _doc_cache[key] = build_store(chunks, vecs)
    store = _doc_cache[key]
    t0 = time.time()
    ans, retrieved = answer(store, question, qvec(question), config)
    dt = time.time() - t0
    return {
        "answer": ans, "config": config, "latency_s": round(dt, 2),
        "ctx_words": sum(len(c["text"].split()) for c in retrieved),
        "n_chunks_total": len(store.chunks), "truncated": truncated,
        "retrieved": [{"id": c["id"], "doc": c["doc"], "text": c["text"][:400]} for c in retrieved],
    }


def do_compare(question):
    item = find_eval(question)
    base = run_one(BASELINE, question, item)
    win = run_one(winner_config(), question, item)

    cut = round((1 - win["ctx_words"] / (base["ctx_words"] or 1)) * 100)
    why = []
    why.append(
        f"Naive pulled {base['n_chunks']} chunks ({base['ctx_words']} words); "
        f"AutoRAG pulled {win['n_chunks']} ({win['ctx_words']} words) — {cut}% less "
        f"text for the model to read.")
    if base["answer_rank"] and win["answer_rank"]:
        why.append(
            f"The passage that actually answers the question ranked "
            f"#{base['answer_rank']} of {base['n_chunks']} under naive, but "
            f"#{win['answer_rank']} of {win['n_chunks']} under AutoRAG — nearer the "
            f"top, where the model pays the most attention.")
    why.append(
        f"Naive's context carried {base['n_distractors']} competing figures"
        + (f" ({', '.join(base['distractors'][:5])}…)" if base["distractors"] else "")
        + f"; AutoRAG's carried {win['n_distractors']}. Colliding numbers are exactly "
        f"what make a 1B model answer with the wrong one.")
    if item and base["correct"] != win["correct"]:
        if win["correct"] and not base["correct"]:
            wf = ", ".join(base["wrong_facts"][:3])
            why.append(
                f"Result: naive answered incorrectly"
                + (f" (it stated {wf}, which isn't the policy figure)" if wf else "")
                + f", while AutoRAG stated the correct fact "
                f"(\"{item['gold']}\").")
    elif item and win["correct"] and base["correct"]:
        why.append("Both happened to land the right fact here — but notice AutoRAG did "
                   "it on a quarter of the context, i.e. cheaper and faster.")

    return {
        "question": question, "item": item, "baseline": base, "winner": win,
        "cut": cut, "why": why,
    }


def do_sweep(full=False):
    space = (
        {"chunk_size": [50, 90, 150, 250], "chunk_overlap": [0, 20, 60],
         "top_k": [2, 3, 5, 8], "retriever": ["dense", "lexical", "hybrid"]}
        if full else
        {"chunk_size": [60, 150], "chunk_overlap": [0, 30],
         "top_k": [3, 5], "retriever": ["dense", "hybrid"]}
    )
    evalset = load_eval()
    qvecs = [qvec(it["q"]) for it in evalset]
    import itertools
    keys = list(space.keys())
    rows = []
    for combo in itertools.product(*(space[k] for k in keys)):
        c = dict(zip(keys, combo))
        store = get_store(c["chunk_size"], c["chunk_overlap"])
        ranks, ctxw = [], []
        for it, qv in zip(evalset, qvecs):
            r = retrieve(store, it["q"], qv, c)
            ranks.append(S.gold_in_chunks(it["gold"], r))
            ctxw.append(sum(len(x["text"].split()) for x in r))
        m = S.retrieval_metrics(ranks, ctxw)
        m["score"] = m["context_recall"] + 0.05 * m["mrr"] - 0.00002 * m["avg_ctx_words"]
        rows.append({"config": c, **m})
    rows.sort(key=lambda x: x["score"], reverse=True)

    # baseline row
    bstore = get_store(BASELINE["chunk_size"], BASELINE["chunk_overlap"])
    branks, bctx = [], []
    for it, qv in zip(evalset, qvecs):
        r = retrieve(bstore, it["q"], qv, BASELINE)
        branks.append(S.gold_in_chunks(it["gold"], r))
        bctx.append(sum(len(x["text"].split()) for x in r))
    bm = S.retrieval_metrics(branks, bctx)

    best = rows[0]
    with open(BEST_PATH, "w") as f:
        json.dump({"config": best["config"], "metrics": best, "baseline": BASELINE}, f, indent=2)
    return {"rows": rows, "baseline": {"config": BASELINE, **bm}, "n": len(evalset)}


# --------------------------------------------------------------------------- #
# HTTP
# --------------------------------------------------------------------------- #
class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype="application/json"):
        data = body if isinstance(body, bytes) else json.dumps(body).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/" or self.path.startswith("/?"):
            return self._send(200, PAGE.encode("utf-8"), "text/html; charset=utf-8")
        if self.path == "/logo.svg":
            try:
                with open(os.path.join(HERE, "assets", "redhat-logo.svg"), "rb") as f:
                    return self._send(200, f.read(), "image/svg+xml")
            except OSError:
                return self._send(404, {"error": "logo not found"})
        if self.path == "/api/state":
            ev = load_eval()
            return self._send(200, {
                "backend": mc.info(),
                "baseline": BASELINE,
                "winner": winner_config(),
                "samples": [it["q"] for it in ev[:8]],
                "questions": [it["q"] for it in ev],
                "corpus_docs": len(DOCS),
                "eval_n": len(ev),
            })
        return self._send(404, {"error": "not found"})

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        try:
            payload = json.loads(self.rfile.read(length) or b"{}")
        except Exception:
            payload = {}
        try:
            if self.path == "/api/ask":
                return self._send(200, do_ask(payload.get("config", {}), payload.get("question", "")))
            if self.path == "/api/ask_doc":
                return self._send(200, do_ask_doc(payload.get("text", ""), payload.get("config", {}), payload.get("question", "")))
            if self.path == "/api/compare_doc":
                return self._send(200, do_compare_doc(payload.get("text", ""), payload.get("question", ""), payload.get("gold", "")))
            if self.path == "/api/compare":
                return self._send(200, do_compare(payload.get("question", "")))
            if self.path == "/api/sweep":
                return self._send(200, do_sweep(full=bool(payload.get("full"))))
        except mc.BackendError as e:
            return self._send(503, {"error": str(e)})
        except Exception as e:
            return self._send(500, {"error": f"{type(e).__name__}: {e}"})
        return self._send(404, {"error": "not found"})


PAGE = r"""<!doctype html><html lang=en><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<title>Red Hat OpenShift AI · AutoRAG</title>
<style>
@import url('https://fonts.googleapis.com/css2?family=Red+Hat+Display:wght@400;500;700;900&family=Red+Hat+Mono:wght@400;500&display=swap');
:root{--red:#ee0000;--ink:#151515;--card:#1e1e1e;--bd:#2a2a2a;--mut:#a0a0a0;--txt:#e6e6e6;--grn:#3ea76a;
--rhd:'Red Hat Display',-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;
--rhm:'Red Hat Mono',ui-monospace,SFMono-Regular,Menlo,monospace}
*{box-sizing:border-box}body{margin:0;background:var(--ink);color:var(--txt);zoom:1.3;
font:15px/1.55 var(--rhd)}
header{display:flex;align-items:center;gap:14px;padding:14px 22px;border-bottom:1px solid var(--bd);background:#0e0e0e}
header .logo{height:27px;width:auto;display:block}
header b{font-weight:700}header span{color:var(--mut);font-size:13px}
.wrap{max-width:1080px;margin:0 auto;padding:22px}
.grid{display:grid;grid-template-columns:300px 1fr;gap:20px}
.card{background:var(--card);border:1px solid var(--bd);border-radius:12px;padding:16px}
h2{font-size:13px;text-transform:uppercase;letter-spacing:.06em;color:var(--mut);margin:0 0 12px}
label{display:block;font-size:12px;color:var(--mut);margin:12px 0 4px}
input[type=range]{width:100%}select,input[type=text],textarea{width:100%;background:#111;color:var(--txt);
border:1px solid var(--bd);border-radius:8px;padding:9px 10px;font-size:14px;font-family:var(--rhd)}
textarea{resize:vertical;line-height:1.5;margin-top:6px}
input[type=file]{font-size:12px;color:var(--mut);margin-top:6px;font-family:var(--rhd)}
input[type=file]::file-selector-button{background:#111;color:var(--txt);border:1px solid var(--bd);
border-radius:7px;padding:6px 10px;margin-right:10px;cursor:pointer;font-family:var(--rhd)}
.val{float:right;color:var(--txt);font-variant-numeric:tabular-nums}
.row{display:flex;gap:8px;flex-wrap:wrap;margin-top:8px}
button{background:var(--red);color:#fff;border:0;border-radius:8px;padding:10px 14px;font-size:14px;
font-weight:600;cursor:pointer;font-family:var(--rhd)}
button.ghost{background:#111;border:1px solid var(--bd);color:var(--txt);font-weight:500}
button.preset.active{border-color:var(--red);color:#fff;background:#1a0d0d;box-shadow:0 0 0 1px var(--red) inset}
button:disabled{opacity:.5;cursor:wait}
.chip{background:#111;border:1px solid var(--bd);color:var(--mut);border-radius:999px;padding:6px 11px;
font-size:12px;cursor:pointer}.chip:hover{border-color:var(--red);color:var(--txt)}
.preset{flex:1;text-align:center}
.ans{white-space:pre-wrap;background:#111;border:1px solid var(--bd);border-radius:8px;padding:14px;margin-top:6px}
.meta{font-size:12px;color:var(--mut);margin-top:6px}
.chunk{font-size:12px;color:var(--mut);background:#111;border:1px solid var(--bd);border-left:3px solid var(--red);
border-radius:6px;padding:8px 10px;margin-top:6px}.chunk b{color:var(--txt)}
table{width:100%;border-collapse:collapse;font-size:13px;margin-top:8px}
th,td{text-align:left;padding:6px 8px;border-bottom:1px solid var(--bd);font-variant-numeric:tabular-nums}
th{color:var(--mut);font-weight:500}tr.win td{color:var(--grn)}
.cmp{display:grid;grid-template-columns:1fr 1fr;gap:12px}
.tag{display:inline-block;font-size:11px;padding:2px 7px;border-radius:5px;margin-bottom:6px}
.tag.b{background:#3a1414;color:#ff8a8a}.tag.w{background:#143a22;color:#8affb0}
.tabs{display:flex;gap:6px;margin-bottom:14px}.tabs .chip.on{border-color:var(--red);color:#fff;background:#1a0d0d}
.hide{display:none}.spin{color:var(--mut);font-size:13px}
small.note{color:var(--mut)}
mark.good{background:#143a22;color:#8affb0;border-radius:4px;padding:0 3px}
mark.bad{background:#3a1414;color:#ff8a8a;border-radius:4px;padding:0 3px;text-decoration:line-through}
mark.dist{background:#3a2a14;color:#ffce8a;border-radius:4px;padding:0 3px}
.difftable{width:100%;border-collapse:collapse;font-size:13px;margin:6px 0 16px}
.difftable th,.difftable td{padding:8px 10px;border-bottom:1px solid var(--bd);text-align:left}
.difftable th{color:var(--mut);font-weight:500}
.difftable td.metric{color:var(--mut)}.difftable .b{color:#ff8a8a;font-variant-numeric:tabular-nums}
.difftable .w{color:#8affb0;font-variant-numeric:tabular-nums}.difftable .note2{color:var(--mut);font-size:12px}
.verdict{display:inline-block;font-size:11px;font-weight:700;padding:2px 8px;border-radius:5px}
.verdict.ok{background:#143a22;color:#8affb0}.verdict.no{background:#3a1414;color:#ff8a8a}
.why{background:#111;border:1px solid var(--bd);border-radius:8px;padding:6px 4px;margin-top:8px}
.why li{margin:8px 14px;color:#d4d4d4}.why li::marker{color:var(--red)}
.chunk.gold{border-left-color:var(--grn)}.chunk.noise{opacity:.6}
.badge{font-size:10px;font-weight:700;padding:1px 6px;border-radius:4px;margin-right:6px}
.badge.g{background:#143a22;color:#8affb0}.badge.d{background:#3a2a14;color:#ffce8a}
details summary{cursor:pointer;color:var(--mut);font-size:12px;margin-top:8px}
.qpick{display:flex;gap:8px;align-items:center}.qpick select{flex:1}
pre{position:relative;background:#0d0d0d;border:1px solid var(--bd);border-radius:8px;padding:12px;overflow:auto;
font:12.5px/1.5 var(--rhm);color:#d4d4d4;margin:4px 0 14px}
pre code{white-space:pre}
.copy{position:absolute;top:8px;right:8px;background:#222;border:1px solid var(--bd);color:var(--mut);
font-size:11px;padding:3px 8px;border-radius:6px;font-weight:500;cursor:pointer}
.cfgline{font-size:16px;margin:6px 0}.cfgline b{color:#8affb0}
.tag.n{background:#222;color:#cfcfcf}
ol.steps{margin:6px 0;padding-left:20px}ol.steps li{margin:8px 0;color:#d4d4d4}
ol.steps code,small code,.note code,li code{background:#0d0d0d;border:1px solid var(--bd);border-radius:4px;
padding:0 4px;font:12px var(--rhm)}
</style></head><body>
<header><img class=logo src="/logo.svg" alt="Red Hat"><span id=backend>OpenShift AI · connecting…</span></header>
<div class=wrap>
<div class=tabs>
<div class="chip on" data-tab=play>Play (one config)</div>
<div class=chip data-tab=cmp>Baseline vs AutoRAG</div>
<div class=chip data-tab=sweep>Run the sweep</div>
<div class=chip data-tab=doc>Import your own doc</div>
<div class=chip data-tab=adopt>Adopt in your stack</div>
</div>

<!-- PLAY -->
<section id=play>
<div class=grid>
<div class=card>
<h2>Retrieval config</h2>
<div class=row><button id=presetBase class="ghost preset" onclick="preset('baseline')">Naive baseline</button>
<button id=presetWin class="ghost preset" onclick="preset('winner')">AutoRAG winner</button></div>
<label>Chunk size <span class=val id=csv>150</span></label><input type=range id=cs min=20 max=300 step=10 value=150>
<label>Overlap <span class=val id=ovv>30</span></label><input type=range id=ov min=0 max=120 step=10 value=30>
<label>Top-k <span class=val id=tkv>3</span></label><input type=range id=tk min=1 max=10 step=1 value=3>
<label>Retriever</label><select id=rt><option value=dense>dense (semantic)</option>
<option value=lexical>lexical (BM25-lite)</option><option value=hybrid>hybrid (RRF)</option></select>
</div>
<div class=card>
<h2>Ask the 1B model</h2>
<input type=text id=q placeholder="Ask a card-disputes question…">
<div class=row id=samples></div>
<div class=row style=margin-top:12px><button onclick=ask()>Ask</button></div>
<div id=out></div>
</div>
</div>
</section>

<!-- DOC -->
<section id=doc class=hide>
<div class=grid>
<div class=card>
<h2>Your document</h2>
<input type=file id=docfile accept=".txt,.md,.markdown,.text,text/plain">
<small class=note>upload a .txt / .md file, or paste below — then ask questions answerable from it</small>
<textarea id=doctext rows=11 placeholder="Paste a policy, a README, meeting notes, a contract… any text."></textarea>
<h2 style=margin-top:14px>Retrieval config</h2>
<label>Chunk size <span class=val id=dcsv>150</span></label><input type=range id=dcs min=20 max=300 step=10 value=150>
<label>Top-k <span class=val id=dtkv>3</span></label><input type=range id=dtk min=1 max=10 step=1 value=3>
<label>Retriever</label><select id=drt><option value=dense>dense (semantic)</option>
<option value=lexical>lexical (BM25-lite)</option><option value=hybrid>hybrid (RRF)</option></select>
</div>
<div class=card>
<h2>Ask your document</h2>
<input type=text id=dq placeholder="Ask something answerable from your text…">
<input type=text id=dgold placeholder="Expected key fact (optional) — unlocks correctness + scoring" style=margin-top:8px>
<div class=row style=margin-top:12px><button onclick=askDoc()>Ask (your config)</button>
<button class=ghost onclick=askDocCompare()>Compare naive vs AutoRAG</button></div>
<small class=note>Runs on your local 1B model — your text is embedded in-memory, not stored. <b>Ask</b> uses the sliders; <b>Compare</b> pits the naive baseline against the AutoRAG winner config on your doc.</small>
<div id=dout></div>
</div>
</div>
</section>

<!-- COMPARE -->
<section id=cmp class=hide>
<div class=card>
<h2>Naive vs AutoRAG — what's different and why</h2>
<div class=qpick><select id=cqsel></select><button onclick=compare()>Compare</button></div>
<small class=note>Pick a graded question so we can show the ground-truth fact and which answer actually got it right.</small>
<div id=cout style=margin-top:16px></div>
</div>
</section>

<!-- SWEEP -->
<section id=sweep class=hide>
<div class=card>
<h2>Measure configs instead of guessing</h2>
<div class=row><button onclick="sweep(false)">Quick sweep (16)</button>
<button class=ghost onclick="sweep(true)">Full sweep (144)</button>
<small class=note>Scores every config on retrieval — no LLM calls — then sets the winner used elsewhere.</small></div>
<div id=sout style=margin-top:12px></div>
</div>
</section>

<!-- ADOPT -->
<section id=adopt class=hide></section>
</div>
<script>
const $=s=>document.querySelector(s), api=(p,b)=>fetch(p,{method:'POST',body:JSON.stringify(b||{})}).then(r=>r.json());
let STATE={};
function cfg(){return{chunk_size:+cs.value,chunk_overlap:+ov.value,top_k:+tk.value,retriever:rt.value}}
function sync(){csv.textContent=cs.value;ovv.textContent=ov.value;tkv.textContent=tk.value}
function clearPreset(){presetBase.classList.remove('active');presetWin.classList.remove('active')}
['input','change'].forEach(e=>['cs','ov','tk'].forEach(id=>$('#'+id).addEventListener(e,()=>{sync();clearPreset()})));
rt.addEventListener('change',clearPreset);
function preset(which){const c=STATE[which];if(!c)return;cs.value=c.chunk_size;ov.value=c.chunk_overlap;
tk.value=c.top_k;rt.value=c.retriever;sync();
presetBase.classList.toggle('active',which==='baseline');presetWin.classList.toggle('active',which==='winner')}
document.querySelectorAll('.tabs .chip').forEach(t=>t.onclick=()=>{
document.querySelectorAll('.tabs .chip').forEach(x=>x.classList.remove('on'));t.classList.add('on');
['play','doc','cmp','sweep','adopt'].forEach(id=>$('#'+id).classList.add('hide'));$('#'+t.dataset.tab).classList.remove('hide');
if(t.dataset.tab==='adopt')renderAdopt()});
function esc(s){return s.replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]))}
function chunks(r){return r.map(c=>`<div class=chunk><b>${c.doc}</b> · ${esc(c.text)}…</div>`).join('')}
function mk(t,f,cls){if(!f)return t;const re=new RegExp('('+f.replace(/[.*+?^${}()|[\]\\]/g,'\\$&')+')','gi');
return t.replace(re,'<mark class="'+cls+'">$1</mark>')}
function hl(text,good,bad){let t=esc(text.replace(/\*\*/g,''));
(good||[]).slice().sort((a,b)=>b.length-a.length).forEach(f=>t=mk(t,f,'good'));
(bad||[]).slice().sort((a,b)=>b.length-a.length).forEach(f=>t=mk(t,f,'bad'));return t}
function hlChunk(text,goldSub,dist){let t=esc(text.replace(/\*\*/g,''));t=mk(t,goldSub,'good');
(dist||[]).slice().sort((a,b)=>b.length-a.length).forEach(f=>t=mk(t,f,'dist'));return t}
function verdict(c){return c?'<span class="verdict ok">✓ correct</span>':'<span class="verdict no">✗ wrong</span>'}
function chunkList(x,goldSub){const items=x.chunks.map(c=>{
const cls=c.has_gold?'chunk gold':(c.distractors.length?'chunk':'chunk noise');
const badge=c.has_gold?'<span class="badge g">✓ ANSWER</span>':
(c.distractors.length?'<span class="badge d">'+c.distractors.length+' competing</span>':'<span class="badge d" style=opacity:.5>off-topic</span>');
return `<div class="${cls}">${badge}<b>${c.doc}</b> · ${hlChunk(c.text,goldSub,c.distractors)}…</div>`}).join('');
return `<details><summary>${x.n_chunks} retrieved chunks (${x.ctx_words} words)</summary>${items}</details>`}
async function ask(){const question=q.value.trim();if(!question)return;
out.innerHTML='<div class=spin>Generating on the 1B model…</div>';
const r=await api('/api/ask',{config:cfg(),question});if(r.error){out.innerHTML='<div class=ans>⚠ '+esc(r.error)+'</div>';return}
out.innerHTML=`<div class=ans>${hl(r.answer,[],[])}</div>
<div class=meta>chunk=${r.config.chunk_size} ov=${r.config.chunk_overlap} k=${r.config.top_k} ${r.config.retriever}
· ${r.ctx_words} context words · ${r.latency_s}s</div>
<h2 style=margin-top:14px>Retrieved context</h2>${chunks(r.retrieved)}`}
docfile.addEventListener('change',e=>{const f=e.target.files[0];if(!f)return;
const r=new FileReader();r.onload=()=>{doctext.value=r.result;dq.focus()};r.readAsText(f)});
['input','change'].forEach(e=>['dcs','dtk'].forEach(id=>$('#'+id).addEventListener(e,()=>{
dcsv.textContent=dcs.value;dtkv.textContent=dtk.value})));
async function askDoc(){const text=doctext.value.trim(),question=dq.value.trim();
if(!text){dout.innerHTML='<div class=ans>Paste or upload a document first.</div>';return}
if(!question){dout.innerHTML='<div class=ans>Ask a question about your document.</div>';return}
dout.innerHTML='<div class=spin>Indexing your document and generating on the 1B model…</div>';
const cfg={chunk_size:+dcs.value,chunk_overlap:Math.round(dcs.value*0.2),top_k:+dtk.value,retriever:drt.value};
const r=await api('/api/ask_doc',{text,question,config:cfg});
if(r.error){dout.innerHTML='<div class=ans>⚠ '+esc(r.error)+'</div>';return}
dout.innerHTML=`<div class=ans>${hl(r.answer,[],[])}</div>
<div class=meta>${r.n_chunks_total} chunks indexed · ${r.ctx_words} words retrieved · ${r.latency_s}s${r.truncated?' · doc truncated to 20k words':''}</div>
<h2 style=margin-top:14px>Retrieved from your document</h2>${chunks(r.retrieved)}`}
async function askDocCompare(){const text=doctext.value.trim(),question=dq.value.trim(),gold=dgold.value.trim();
if(!text){dout.innerHTML='<div class=ans>Paste or upload a document first.</div>';return}
if(!question){dout.innerHTML='<div class=ans>Ask a question about your document.</div>';return}
dout.innerHTML='<div class=spin>Running naive vs AutoRAG on your document…</div>';
const r=await api('/api/compare_doc',{text,question,gold});
if(r.error){dout.innerHTML='<div class=ans>⚠ '+esc(r.error)+'</div>';return}
renderDocCompare(r)}
function renderDocCompare(r){const b=r.baseline,w=r.winner,gold=r.gold;
const rank=x=>x.answer_rank?('#'+x.answer_rank+' of '+x.n_chunks):'—';
let rows='';
if(gold){rows+=`<tr><td class=metric>Got it right?</td><td>${verdict(b.correct)}</td><td>${verdict(w.correct)}</td><td class=note2>did the answer state your key fact</td></tr>`
+`<tr><td class=metric>Answer quality</td><td class=b>${b.score}</td><td class=w>${w.score}</td><td class=note2>0–1 vs your key fact</td></tr>`}
rows+=`<tr><td class=metric>Context fed to model</td><td class=b>${b.ctx_words}w · ${b.n_chunks} chunks</td><td class=w>${w.ctx_words}w · ${w.n_chunks} chunks</td><td class=note2>−${r.cut}% less to read</td></tr>`;
if(gold)rows+=`<tr><td class=metric>Answer-bearing chunk ranked</td><td class=b>${rank(b)}</td><td class=w>${rank(w)}</td><td class=note2>higher = model attends more</td></tr>`;
rows+=`<tr><td class=metric>Competing figures in context</td><td class=b>${b.n_distractors}</td><td class=w>${w.n_distractors}</td><td class=note2>colliding $/days/% that mislead</td></tr>`;
const col=(x,cls,name)=>`<div><span class="tag ${cls}">${name}</span> ${gold?verdict(x.correct):''}
<div class=ans>${hl(x.answer,x.gold_facts,x.wrong_facts)}</div>
<div class=meta>${x.ctx_words} ctx words · ${x.n_distractors} competing figures · ${x.latency_s}s</div>
${chunkList(x,gold||'')}</div>`;
dout.innerHTML=`<div class=meta style=margin-bottom:4px><b>Q:</b> ${esc(r.question)}</div>
${gold?'<div class=meta style=margin-bottom:8px><b>Your key fact:</b> <mark class=good>'+esc(gold)+'</mark></div>':'<div class=meta style=margin-bottom:8px>Add an “expected key fact” above to also score correctness.</div>'}
<table class=difftable><tr><th>What differs</th><th>Naive baseline</th><th>AutoRAG winner</th><th></th></tr>${rows}</table>
<h2 style=margin-top:4px>Why they differ</h2><ul class=why>${r.why.map(s=>'<li>'+esc(s)+'</li>').join('')}</ul>
<h2 style=margin-top:14px>The two answers${gold?' <small class=note>· <mark class=good>green</mark> = your fact, <mark class=bad>red</mark> = figure not in it</small>':''}</h2>
<div class=cmp>${col(b,'b','Naive baseline')}${col(w,'w','AutoRAG winner')}</div>`}
async function compare(){const question=cqsel.value;if(!question)return;
cout.innerHTML='<div class=spin>Running both configs on the 1B model…</div>';
const r=await api('/api/compare',{question});if(r.error){cout.innerHTML='<div class=ans>⚠ '+esc(r.error)+'</div>';return}
const b=r.baseline,w=r.winner,it=r.item,gold=it?it.gold:'';
const rank=x=>x.answer_rank?('#'+x.answer_rank+' of '+x.n_chunks):'not retrieved';
const ref=it?`<div class=meta style=margin-bottom:8px><b>Ground-truth fact:</b> <mark class=good>${esc(it.gold)}</mark></div>`:'';
const table=`<table class=difftable>
<tr><th>What differs</th><th>Naive baseline</th><th>AutoRAG winner</th><th></th></tr>
<tr><td class=metric>Got it right?</td><td>${verdict(b.correct)}</td><td>${verdict(w.correct)}</td><td class=note2>did the answer state the policy fact</td></tr>
<tr><td class=metric>Answer quality</td><td class=b>${b.score}</td><td class=w>${w.score}</td><td class=note2>0–1 vs reference answer</td></tr>
<tr><td class=metric>Context fed to model</td><td class=b>${b.ctx_words}w · ${b.n_chunks} chunks</td><td class=w>${w.ctx_words}w · ${w.n_chunks} chunks</td><td class=note2>−${r.cut}% less to read</td></tr>
<tr><td class=metric>Answer-bearing chunk ranked</td><td class=b>${rank(b)}</td><td class=w>${rank(w)}</td><td class=note2>higher = model attends to it more</td></tr>
<tr><td class=metric>Competing figures in context</td><td class=b>${b.n_distractors}</td><td class=w>${w.n_distractors}</td><td class=note2>colliding $/days/% that mislead</td></tr>
</table>`;
const col=(x,cls,name)=>`<div><span class="tag ${cls}">${name}</span> ${verdict(x.correct)}
<div class=ans>${hl(x.answer,x.gold_facts,x.wrong_facts)}</div>
<div class=meta>${x.ctx_words} ctx words · ${x.n_distractors} competing figures · ${x.latency_s}s</div>
${chunkList(x,gold)}</div>`;
const why=`<h2 style=margin-top:4px>Why they differ</h2><ul class=why>${r.why.map(s=>'<li>'+esc(s)+'</li>').join('')}</ul>`;
cout.innerHTML=`<div class=meta style=margin-bottom:4px><b>Q:</b> ${esc(r.question)}</div>${ref}${table}${why}
<h2 style=margin-top:14px>The two answers <small class=note>· <mark class=good>green</mark> = correct policy fact, <mark class=bad>red</mark> = figure not in policy</small></h2>
<div class=cmp>${col(b,'b','Naive baseline')}${col(w,'w','AutoRAG winner')}</div>`}
async function sweep(full){sout.innerHTML='<div class=spin>Scoring configs…</div>';
const r=await api('/api/sweep',{full});if(r.error){sout.innerHTML='<div class=ans>⚠ '+esc(r.error)+'</div>';return}
const pct=x=>(x*100).toFixed(1)+'%';
const rows=r.rows.slice(0,8).map((x,i)=>`<tr class="${i==0?'win':''}"><td>${pct(x.context_recall)}</td>
<td>${x.mrr.toFixed(3)}</td><td>${Math.round(x.avg_ctx_words)}w</td>
<td>chunk=${x.config.chunk_size} ov=${x.config.chunk_overlap} k=${x.config.top_k} ${x.config.retriever}${i==0?' ←':''}</td></tr>`).join('');
const b=r.baseline;
sout.innerHTML=`<table><tr><th>recall</th><th>mrr</th><th>ctx</th><th>config</th></tr>${rows}</table>
<div class=meta style=margin-top:10px>Naive baseline: recall ${pct(b.context_recall)} · mrr ${b.mrr.toFixed(3)} · ${Math.round(b.avg_ctx_words)}w —
winner uses <b style=color:var(--grn)>${Math.round((1-r.rows[0].avg_ctx_words/b.avg_ctx_words)*100)}% less context</b> at equal recall.
Winner is now the “AutoRAG winner” preset on the Play tab.</div>`;
STATE.winner=r.rows[0].config;renderAdopt()}
function pre(code){return '<pre><code>'+esc(code)+'</code><button class=copy onclick="cp(this)">copy</button></pre>'}
function cp(b){navigator.clipboard.writeText(b.previousElementSibling.textContent);b.textContent='copied';setTimeout(()=>b.textContent='copy',1200)}
function renderAdopt(){const c=STATE.winner,W=c.chunk_size,O=c.chunk_overlap,K=c.top_k,R=c.retriever;
const raw=(doctext.value||'').trim();const hasdoc=raw.length>0;
const TQ=String.fromCharCode(34).repeat(3);
const docex=(hasdoc?raw.slice(0,700):'Paste your document text here — or import one in the "Import your own doc" tab.').replace(new RegExp(TQ,'g'),'\\"\\"\\"');
const q=((dq.value||'').trim()||'What are the key terms?').replace(/"/g,'\\"');
const hyb=R==='hybrid';
const lab='# End-to-end on YOUR document, with the AutoRAG-winning config.\n'
+'# This IS the playground\'s code: model_client.py + rag/ — pure stdlib + Ollama, ~200 lines you own.\n'
+'import model_client as mc                 # Ollama | vLLM | OpenAI-compatible | stub (one interface)\n'
+'from rag.chunk import build_chunks\n'
+'from rag.pipeline import build_store, answer\n\n'
+'document = '+TQ+docex+TQ+'\n\n'
+'cfg = {"chunk_size": '+W+', "chunk_overlap": '+O+', "top_k": '+K+', "retriever": "'+R+'"}  # <- sweep winner\n\n'
+'chunks = build_chunks([{"doc": "your-doc", "text": document}], cfg["chunk_size"], cfg["chunk_overlap"])\n'
+'store  = build_store(chunks, mc.embed([c["text"] for c in chunks]))   # embed once, reuse\n\n'
+'q = "'+q+'"\n'
+'ans, hits = answer(store, q, mc.embed(q)[0], cfg)   # retrieve ('+R+', k='+K+') -> generate\n'
+'print(ans)\n'
+'print("from:", [h["id"] for h in hits])';
const lc='# Same winner config wired into LangChain (FAISS + local Ollama).\n'
+'from langchain_text_splitters import RecursiveCharacterTextSplitter\n'
+'from langchain_community.vectorstores import FAISS\n'
+'from langchain_ollama import OllamaEmbeddings, ChatOllama\n'
+(hyb?'from langchain.retrievers import EnsembleRetriever\nfrom langchain_community.retrievers import BM25Retriever\n':'')
+'from langchain.chains import RetrievalQA\n\n'
+'document = '+TQ+docex+TQ+'\n\n'
+'# AutoRAG chunk size is in WORDS; LangChain counts characters, so x5 (or use a token splitter).\n'
+'docs = RecursiveCharacterTextSplitter(chunk_size='+(W*5)+', chunk_overlap='+(O*5)+').create_documents([document])\n'
+'vs   = FAISS.from_documents(docs, OllamaEmbeddings(model="nomic-embed-text"))\n'
+(hyb
  ?'dense = vs.as_retriever(search_kwargs={"k": '+K+'})\nbm25  = BM25Retriever.from_documents(docs); bm25.k = '+K+'\nretriever = EnsembleRetriever(retrievers=[bm25, dense], weights=[0.5, 0.5])  # hybrid (RRF-like)\n'
  :'retriever = vs.as_retriever(search_kwargs={"k": '+K+'})\n')
+'qa = RetrievalQA.from_chain_type(llm=ChatOllama(model="llama3.2:1b", temperature=0), retriever=retriever)\n'
+'print(qa.invoke("'+q+'")["result"])';
const li='# LlamaIndex equivalent (chunk size in tokens ~ words here).\n'
+'from llama_index.core.node_parser import SentenceSplitter\n'
+'splitter = SentenceSplitter(chunk_size='+W+', chunk_overlap='+O+')\n'
+'query_engine = index.as_query_engine(similarity_top_k='+K+')\n'
+'print(query_engine.query("'+q+'"))';
const env='# Scale out: keep the code, point the model at a vLLM endpoint on Red Hat OpenShift AI.\n'
+'# (1) the lab client — one env switch:\n'
+'export BACKEND=openai OPENAI_BASE=https://<vllm-route>/v1 CHAT_MODEL=<served-model>\n\n'
+'# (2) or in LangChain, swap the two local clients for OpenAI-compatible ones:\n'
+'from langchain_openai import ChatOpenAI, OpenAIEmbeddings\n'
+'llm = ChatOpenAI(base_url="https://<vllm-route>/v1", api_key="EMPTY", model="<served-model>")\n'
+'emb = OpenAIEmbeddings(base_url="https://<vllm-route>/v1", api_key="EMPTY", model="<embed-model>")';
$('#adopt').innerHTML=`<div class=card>
<h2>Your winning config</h2>
<div class=cfgline>chunk=<b>${W}</b> words · overlap=<b>${O}</b> · top-k=<b>${K}</b> · retriever=<b>${R}</b></div>
<small class=note>${hasdoc?'Scripts below are filled in with the document + question from the “Import your own doc” tab.':'Tip: import a document and type a question in the “Import your own doc” tab, then reopen this tab — these scripts auto-fill with your text.'}</small>
<h2 style=margin-top:18px>Run it on your document — the lab's own code (no framework)</h2>
<small class=note>Copy <code>model_client.py</code> + <code>rag/</code> into your project and this runs as-is.</small>
${pre(lab)}
<h2 style=margin-top:8px>Wire the same config into LangChain</h2>${pre(lc)}
<h2 style=margin-top:8px>LlamaIndex</h2>${pre(li)}
<h2 style=margin-top:8px>Scale to vLLM / OpenShift AI — same code</h2>${pre(env)}
<h2 style=margin-top:8px>Different vector DB? Keep the sweep, implement two functions</h2>
${pre('def index(chunks) -> store                # embed + insert (FAISS / Chroma / pgvector)\ndef retrieve(store, query, k) -> chunks    # dense / lexical / hybrid')}
<small class=note>They're called in <code>rag/pipeline.py</code>; the reference store is <code>rag/store.py</code>.</small>
<h2 style=margin-top:18px>Tune on your data (2 swaps)</h2>
<ol class=steps>
<li><b>Your corpus</b> — replace <code>corpus/*.md</code> (or import a doc in the doc tab).</li>
<li><b>Your eval</b> — replace <code>eval/qa.jsonl</code> with ~15–20 real questions, each with a <code>gold</code> phrase that must be retrieved, then <code>python3 autorag.py sweep</code> to re-find the winner above.</li>
</ol>
<small class=note>Full guide: <code>autorag/ADOPT.md</code>. AutoRAG is a technique, not a platform lock-in.</small>
</div>`}
(async()=>{STATE=await fetch('/api/state').then(r=>r.json());
$('#backend').textContent=STATE.backend+'  ·  '+STATE.corpus_docs+' docs · '+STATE.eval_n+' eval Qs';
samples.innerHTML=STATE.samples.map(s=>`<span class=chip onclick="q.value=this.textContent;ask()">${esc(s)}</span>`).join('');
cqsel.innerHTML=STATE.questions.map(s=>`<option>${esc(s)}</option>`).join('');
cqsel.value=STATE.questions.find(q=>q.includes('lost debit card'))||STATE.questions[0];
preset('winner');renderAdopt()})();
</script></body></html>"""


def main():
    print(f"\n  AutoRAG playground  ·  {mc.info()}")
    print(f"  → http://localhost:{PORT}\n")
    if mc.BACKEND != "stub":
        try:
            mc.embed("warmup")
        except mc.BackendError as e:
            print(f"  ⚠ backend not reachable yet: {e}\n")
    ThreadingHTTPServer(("", PORT), Handler).serve_forever()


if __name__ == "__main__":
    main()
