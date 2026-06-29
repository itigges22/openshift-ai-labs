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


def do_compare(question):
    base = do_ask(BASELINE, question)
    win = do_ask(winner_config(), question)
    return {"baseline": base, "winner": win}


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
        if self.path == "/api/state":
            ev = load_eval()
            return self._send(200, {
                "backend": mc.info(),
                "baseline": BASELINE,
                "winner": winner_config(),
                "samples": [it["q"] for it in ev[:8]],
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
<title>AutoRAG Playground · Red Hat OpenShift AI</title>
<style>
:root{--red:#ee0000;--ink:#151515;--card:#1e1e1e;--bd:#2a2a2a;--mut:#a0a0a0;--txt:#e6e6e6;--grn:#3ea76a}
*{box-sizing:border-box}body{margin:0;background:var(--ink);color:var(--txt);
font:15px/1.5 -apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif}
header{display:flex;align-items:center;gap:12px;padding:14px 22px;border-bottom:1px solid var(--bd);background:#0e0e0e}
header .dot{width:11px;height:11px;border-radius:50%;background:var(--red)}
header b{font-weight:700}header span{color:var(--mut);font-size:13px}
.wrap{max-width:1080px;margin:0 auto;padding:22px}
.grid{display:grid;grid-template-columns:300px 1fr;gap:20px}
.card{background:var(--card);border:1px solid var(--bd);border-radius:12px;padding:16px}
h2{font-size:13px;text-transform:uppercase;letter-spacing:.06em;color:var(--mut);margin:0 0 12px}
label{display:block;font-size:12px;color:var(--mut);margin:12px 0 4px}
input[type=range]{width:100%}select,input[type=text]{width:100%;background:#111;color:var(--txt);
border:1px solid var(--bd);border-radius:8px;padding:9px 10px;font-size:14px}
.val{float:right;color:var(--txt);font-variant-numeric:tabular-nums}
.row{display:flex;gap:8px;flex-wrap:wrap;margin-top:8px}
button{background:var(--red);color:#fff;border:0;border-radius:8px;padding:10px 14px;font-size:14px;
font-weight:600;cursor:pointer}button.ghost{background:#111;border:1px solid var(--bd);color:var(--txt);font-weight:500}
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
</style></head><body>
<header><i class=dot></i><b>AutoRAG Playground</b><span id=backend>Red Hat OpenShift AI · connecting…</span></header>
<div class=wrap>
<div class=tabs>
<div class="chip on" data-tab=play>Play (one config)</div>
<div class=chip data-tab=cmp>Baseline vs AutoRAG</div>
<div class=chip data-tab=sweep>Run the sweep</div>
</div>

<!-- PLAY -->
<section id=play>
<div class=grid>
<div class=card>
<h2>Retrieval config</h2>
<div class=row><button class="ghost preset" onclick="preset('baseline')">Naive baseline</button>
<button class="ghost preset" onclick="preset('winner')">AutoRAG winner</button></div>
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

<!-- COMPARE -->
<section id=cmp class=hide>
<div class=card>
<h2>Same question, both configs</h2>
<input type=text id=cq placeholder="Ask a card-disputes question…">
<div class=row id=csamples></div>
<div class=row style=margin-top:12px><button onclick=compare()>Compare</button>
<small class=note>Naive baseline (big chunks, top-k 8) vs the swept winner — watch noise change the answer.</small></div>
<div id=cout style=margin-top:14px></div>
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
</div>
<script>
const $=s=>document.querySelector(s), api=(p,b)=>fetch(p,{method:'POST',body:JSON.stringify(b||{})}).then(r=>r.json());
let STATE={};
function cfg(){return{chunk_size:+cs.value,chunk_overlap:+ov.value,top_k:+tk.value,retriever:rt.value}}
function sync(){csv.textContent=cs.value;ovv.textContent=ov.value;tkv.textContent=tk.value}
['input','change'].forEach(e=>['cs','ov','tk'].forEach(id=>$('#'+id).addEventListener(e,sync)));
function preset(which){const c=STATE[which];if(!c)return;cs.value=c.chunk_size;ov.value=c.chunk_overlap;
tk.value=c.top_k;rt.value=c.retriever;sync()}
document.querySelectorAll('.tabs .chip').forEach(t=>t.onclick=()=>{
document.querySelectorAll('.tabs .chip').forEach(x=>x.classList.remove('on'));t.classList.add('on');
['play','cmp','sweep'].forEach(id=>$('#'+id).classList.add('hide'));$('#'+t.dataset.tab).classList.remove('hide')});
function esc(s){return s.replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]))}
function chunks(r){return r.map(c=>`<div class=chunk><b>${c.doc}</b> · ${esc(c.text)}…</div>`).join('')}
async function ask(){const question=q.value.trim();if(!question)return;
out.innerHTML='<div class=spin>Generating on the 1B model…</div>';
const r=await api('/api/ask',{config:cfg(),question});if(r.error){out.innerHTML='<div class=ans>⚠ '+esc(r.error)+'</div>';return}
out.innerHTML=`<div class=ans>${esc(r.answer)}</div>
<div class=meta>chunk=${r.config.chunk_size} ov=${r.config.chunk_overlap} k=${r.config.top_k} ${r.config.retriever}
· ${r.ctx_words} context words · ${r.latency_s}s</div>
<h2 style=margin-top:14px>Retrieved context</h2>${chunks(r.retrieved)}`}
async function compare(){const question=cq.value.trim();if(!question)return;
cout.innerHTML='<div class=spin>Running both configs on the 1B model…</div>';
const r=await api('/api/compare',{question});if(r.error){cout.innerHTML='<div class=ans>⚠ '+esc(r.error)+'</div>';return}
const col=(x,cls,name)=>`<div><span class="tag ${cls}">${name}</span>
<div class=ans>${esc(x.answer)}</div><div class=meta>${x.ctx_words} ctx words · k=${x.config.top_k} · ${x.latency_s}s</div></div>`;
cout.innerHTML=`<div class=cmp>${col(r.baseline,'b','Naive baseline')}${col(r.winner,'w','AutoRAG winner')}</div>`}
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
STATE.winner=r.rows[0].config}
(async()=>{STATE=await fetch('/api/state').then(r=>r.json());
$('#backend').textContent=STATE.backend+'  ·  '+STATE.corpus_docs+' docs · '+STATE.eval_n+' eval Qs';
samples.innerHTML=STATE.samples.map(s=>`<span class=chip onclick="q.value=this.textContent;ask()">${esc(s.slice(0,42))}…</span>`).join('');
csamples.innerHTML=STATE.samples.map(s=>`<span class=chip onclick="cq.value=this.textContent;compare()">${esc(s.slice(0,42))}…</span>`).join('');
preset('winner')})();
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
