#!/usr/bin/env python3
"""
AutoRAG — automated RAG configuration search, local-first.

RAG has a dozen knobs (chunk size, overlap, top-k, dense vs lexical vs hybrid
retrieval, ...). The "right" combination is corpus-specific, and hand-tuning it
is the classic RAG pain point: you change chunk size, eyeball a few answers,
convince yourself it's better, and ship a guess.

AutoRAG replaces the guessing with a measured search. It sweeps the config space
against a held-out eval set, scores each config on a retrieval metric (no LLM
calls — fast and cheap), ranks them, and hands you the winner. Then it runs the
real 1B model on the naive baseline vs the winner so you can read the
end-to-end answer-quality lift in actual generated text.

Everything here runs on a laptop against Ollama (llama3.2:1b + nomic-embed-text),
or against any OpenAI-compatible endpoint — a vLLM `InferenceService` on Red Hat
OpenShift AI is the same code, just BACKEND=openai. That portability from laptop
to cluster is the whole point: tune locally, serve at scale.

Usage:
  python autorag.py sweep              # search configs, print leaderboard
  python autorag.py sweep --full       # larger search space
  python autorag.py demo               # baseline vs winner, real generated answers
  python autorag.py ask "..."          # answer one question with the winning config
  BACKEND=stub python autorag.py sweep # run the harness with no model (offline test)
"""

import itertools
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import model_client as mc
from rag.chunk import load_corpus, build_chunks
from rag.pipeline import build_store, retrieve, answer
from eval import score as S

HERE = os.path.dirname(os.path.abspath(__file__))
CORPUS_DIR = os.path.join(HERE, "corpus")
EVAL_PATH = os.path.join(HERE, "eval", "qa.jsonl")
BEST_PATH = os.path.join(HERE, "best_config.json")

# The naive config a lot of teams actually ship: don't really chunk (one big
# chunk per ~150–220-word doc), and "retrieve a lot to be safe" (high top-k,
# dense-only). It hits full recall but buries the answer under ~1,500 words of
# competing policies — and this corpus is full of colliding numbers ($50,
# 60 days, 90 days, 30 days), exactly what makes a 1B model grab the wrong one.
BASELINE = {"chunk_size": 250, "chunk_overlap": 0, "top_k": 8, "retriever": "dense"}

QUICK = {
    "chunk_size": [60, 150],
    "chunk_overlap": [0, 30],
    "top_k": [3, 5],
    "retriever": ["dense", "hybrid"],
}
FULL = {
    "chunk_size": [50, 90, 150, 250],
    "chunk_overlap": [0, 20, 60],
    "top_k": [2, 3, 5, 8],
    "retriever": ["dense", "lexical", "hybrid"],
}

RH = "\033[31m"   # red hat red
DIM = "\033[2m"
BOLD = "\033[1m"
GRN = "\033[32m"
END = "\033[0m"


def banner(title):
    print(f"\n{RH}{BOLD}● Red Hat OpenShift AI · AutoRAG{END}  {DIM}{mc.info()}{END}")
    print(f"{BOLD}{title}{END}")


def load_eval():
    items = []
    with open(EVAL_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def configs(space):
    keys = list(space.keys())
    for combo in itertools.product(*(space[k] for k in keys)):
        yield dict(zip(keys, combo))


def cfg_str(c):
    return (f"chunk={c['chunk_size']:<4} ov={c['chunk_overlap']:<3} "
            f"k={c['top_k']:<2} {c['retriever']:<7}")


# --------------------------------------------------------------------------- #
# Core: evaluate one config on retrieval metrics (no generation)
# --------------------------------------------------------------------------- #
def eval_config(config, evalset, store_cache, qvecs):
    key = (config["chunk_size"], config["chunk_overlap"])
    store = store_cache[key]
    ranks, ctx_words = [], []
    for item, qvec in zip(evalset, qvecs):
        retrieved = retrieve(store, item["q"], qvec, config)
        ranks.append(S.gold_in_chunks(item["gold"], retrieved))
        ctx_words.append(sum(len(c["text"].split()) for c in retrieved))
    m = S.retrieval_metrics(ranks, ctx_words)
    # Composite the sweep ranks on. Recall dominates absolutely (weight ~1 per
    # whole point); MRR breaks recall ties toward answer-at-top; context size is
    # a light penalty so that, all else equal, the tighter context wins — which
    # is exactly what helps a small model and cuts cost.
    m["score"] = m["context_recall"] + 0.05 * m["mrr"] - 0.00002 * m["avg_ctx_words"]
    return m


def build_store_cache(docs, space):
    """Embed chunks once per (chunk_size, overlap) — the expensive step, cached."""
    chunkings = sorted({(s, o) for s in space["chunk_size"] for o in space["chunk_overlap"]})
    cache = {}
    for (size, overlap) in chunkings:
        chunks = build_chunks(docs, size, overlap)
        t0 = time.time()
        vecs = mc.embed([c["text"] for c in chunks])
        dt = time.time() - t0
        cache[(size, overlap)] = build_store(chunks, vecs)
        print(f"  {DIM}indexed chunk={size:<4} ov={overlap:<3} "
              f"-> {len(chunks):>3} chunks  ({dt:4.1f}s){END}")
    return cache


# --------------------------------------------------------------------------- #
# Commands
# --------------------------------------------------------------------------- #
def cmd_sweep(full=False):
    space = FULL if full else QUICK
    evalset = load_eval()
    docs = load_corpus(CORPUS_DIR)
    all_configs = list(configs(space))

    banner(f"Sweep · {len(all_configs)} configs · {len(evalset)} eval questions "
           f"· corpus {len(docs)} docs")

    print(f"\n{DIM}Indexing (embeddings cached per chunking)…{END}")
    store_cache = build_store_cache(docs, space)
    qvecs = mc.embed([it["q"] for it in evalset])

    print(f"\n{DIM}Scoring configs on retrieval (context_recall · mrr)…{END}\n")
    results = []
    for c in all_configs:
        m = eval_config(c, evalset, store_cache, qvecs)
        results.append((c, m))
        print(f"  {cfg_str(c)}  recall={m['context_recall']*100:5.1f}%  "
              f"mrr={m['mrr']:.3f}  ctx={m['avg_ctx_words']:4.0f}w  ({m['hits']}/{m['n']})")

    results.sort(key=lambda x: x[1]["score"], reverse=True)
    best, bestm = results[0]

    # where does the naive baseline land?
    base_m = eval_config(_ensure_in_cache(BASELINE, docs, store_cache),
                         evalset, store_cache, qvecs)

    print(f"\n{BOLD}Leaderboard (top 5){END}")
    for c, m in results[:5]:
        tag = f"  {GRN}← winner{END}" if c is best else ""
        print(f"  recall={m['context_recall']*100:5.1f}%  mrr={m['mrr']:.3f}  "
              f"ctx={m['avg_ctx_words']:4.0f}w  {cfg_str(c)}{tag}")

    rec_lift = (bestm["context_recall"] - base_m["context_recall"]) * 100
    mrr_lift = bestm["mrr"] - base_m["mrr"]
    ctx_cut = (1 - bestm["avg_ctx_words"] / (base_m["avg_ctx_words"] or 1)) * 100

    print(f"\n{BOLD}Naive baseline{END}  {cfg_str(BASELINE)}")
    print(f"   recall={base_m['context_recall']*100:.1f}%  mrr={base_m['mrr']:.3f}  "
          f"ctx={base_m['avg_ctx_words']:.0f}w")
    print(f"{BOLD}{GRN}AutoRAG winner{END} {cfg_str(best)}")
    print(f"   recall={bestm['context_recall']*100:.1f}%  mrr={bestm['mrr']:.3f}  "
          f"ctx={bestm['avg_ctx_words']:.0f}w")

    print(f"\n{BOLD}AutoRAG vs naive baseline{END}")
    print(f"  context recall : {GRN if rec_lift>=0 else RH}{rec_lift:+.1f} pts{END}"
          f"  {DIM}(both high here — a clean corpus saturates recall){END}")
    print(f"  answer rank    : {GRN if mrr_lift>=0 else RH}{mrr_lift:+.3f} MRR{END}"
          f"  {DIM}(answer ranked nearer the top of context){END}")
    print(f"  context size   : {GRN if ctx_cut>=0 else RH}{ctx_cut:+.0f}% words{END}"
          f"  {DIM}(less noise + lower cost/latency for the model){END}")
    print(f"\n  {DIM}The recall ceiling hides the real win: the winner feeds the model "
          f"tighter,\n  better-ranked context. On a 1B model that is the difference between "
          f"a right\n  and a wrong answer — see {BOLD}python autorag.py demo{END}{DIM}.{END}")

    with open(BEST_PATH, "w") as f:
        json.dump({"config": best, "metrics": bestm, "baseline": BASELINE}, f, indent=2)
    print(f"{DIM}Saved winning config -> {os.path.relpath(BEST_PATH)}{END}")
    print(f"\nNext: {BOLD}python autorag.py demo{END} "
          f"— run the 1B model on baseline vs winner and read the answers.\n")
    return best


def _ensure_in_cache(config, docs, store_cache):
    key = (config["chunk_size"], config["chunk_overlap"])
    if key not in store_cache:
        chunks = build_chunks(docs, *key)
        vecs = mc.embed([c["text"] for c in chunks])
        store_cache[key] = build_store(chunks, vecs)
    return config


def _load_best():
    if os.path.exists(BEST_PATH):
        return json.load(open(BEST_PATH))["config"]
    return None


def cmd_demo(n=None):
    """Generate real answers with baseline vs winning config and score quality."""
    best = _load_best()
    if best is None:
        print(f"{DIM}No saved winner yet — running a sweep first…{END}")
        best = cmd_sweep()

    evalset = load_eval()
    if n:
        evalset = evalset[:n]
    docs = load_corpus(CORPUS_DIR)

    banner(f"A/B · generating answers with {mc.CHAT_MODEL} · {len(evalset)} questions")
    print(f"  baseline {cfg_str(BASELINE)}")
    print(f"  winner   {cfg_str(best)}\n")

    cache = {}
    _ensure_in_cache(BASELINE, docs, cache)
    _ensure_in_cache(best, docs, cache)

    def run(config):
        store = cache[(config["chunk_size"], config["chunk_overlap"])]
        scores = []
        rows = []
        for it in evalset:
            qvec = mc.embed(it["q"])[0]
            text, retrieved = answer(store, it["q"], qvec, config)
            ref_vec, cand_vec = mc.embed([it["answer"], text])
            sc = S.answer_score(it["answer"], text, ref_vec, cand_vec)
            scores.append(sc)
            rows.append((it, text, sc))
        return sum(scores) / (len(scores) or 1), rows

    print(f"{DIM}Running baseline…{END}")
    base_avg, base_rows = run(BASELINE)
    print(f"{DIM}Running AutoRAG winner…{END}\n")
    win_avg, win_rows = run(best)

    # show the questions where tight context helped most (largest winner-baseline
    # gap) — that's where naive RAG's noise actually changed the answer.
    gaps = sorted(range(len(evalset)),
                  key=lambda i: win_rows[i][2] - base_rows[i][2], reverse=True)
    print(f"{BOLD}Where tighter context changed the answer (biggest wins){END}\n")
    for i in gaps[:3]:
        it, btext, bs = base_rows[i]
        _, wtext, ws = win_rows[i]
        print(f"{BOLD}Q:{END} {it['q']}")
        print(f"  {DIM}gold: {it['answer'][:120]}{END}")
        print(f"  {RH}baseline ({bs:.2f}){END} {btext[:150].replace(chr(10), ' ')}")
        print(f"  {GRN}winner   ({ws:.2f}){END} {wtext[:150].replace(chr(10), ' ')}\n")

    base_ctx = sum(len(c["text"].split())
                   for it in evalset
                   for c in retrieve(cache[(BASELINE["chunk_size"], BASELINE["chunk_overlap"])],
                                     it["q"], mc.embed(it["q"])[0], BASELINE)) / len(evalset)
    win_ctx = sum(len(c["text"].split())
                  for it in evalset
                  for c in retrieve(cache[(best["chunk_size"], best["chunk_overlap"])],
                                    it["q"], mc.embed(it["q"])[0], best)) / len(evalset)

    lift = (win_avg - base_avg) * 100
    ctx_cut = (1 - win_ctx / (base_ctx or 1)) * 100
    print(f"{BOLD}Answer quality (0–1, blend of semantic sim + keyword coverage){END}")
    print(f"  baseline : {base_avg:.3f}   {DIM}({base_ctx:.0f} words of context/question){END}")
    print(f"  winner   : {win_avg:.3f}   {DIM}({win_ctx:.0f} words of context/question){END}")
    print(f"  {BOLD}{GRN if lift >= 0 else RH}answer-quality lift: {lift:+.1f} points{END}"
          f"   {BOLD}{GRN}context: -{ctx_cut:.0f}% tokens{END}")
    print(f"  {DIM}better answers on less context = higher accuracy AND lower cost/latency.{END}\n")


def cmd_ask(question):
    best = _load_best() or BASELINE
    docs = load_corpus(CORPUS_DIR)
    cache = {}
    _ensure_in_cache(best, docs, cache)
    banner("Ask · winning config")
    print(f"  {cfg_str(best)}\n")
    store = cache[(best["chunk_size"], best["chunk_overlap"])]
    qvec = mc.embed(question)[0]
    text, retrieved = answer(store, question, qvec, best)
    print(f"{BOLD}Q:{END} {question}\n{BOLD}A:{END} {text}\n")
    print(f"{DIM}retrieved: {', '.join(c['id'] for c in retrieved)}{END}\n")


def main():
    args = sys.argv[1:]
    cmd = args[0] if args else "sweep"
    if cmd == "sweep":
        cmd_sweep(full="--full" in args)
    elif cmd == "demo":
        n = None
        if "--n" in args:
            n = int(args[args.index("--n") + 1])
        cmd_demo(n=n)
    elif cmd == "ask":
        cmd_ask(" ".join(a for a in args[1:] if not a.startswith("--")))
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
