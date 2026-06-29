# AutoRAG — stop hand-tuning RAG, measure it

A **runnable** lab. Not a slideshow — it indexes a real corpus, retrieves with a
real embedding model, generates with a real **1B** LLM on your laptop, and shows
you, in real numbers, why the RAG config you guessed is costing you accuracy.

```
┌──────────────────────────────────────────────────────────────────────┐
│  Red Hat OpenShift AI · AutoRAG                                        │
│                                                                        │
│  Naive baseline   chunk=250 ov=0  k=8  dense   recall=100%  ctx=1347w  │
│  AutoRAG winner   chunk=150 ov=30 k=3  dense   recall=100%  ctx= 358w  │
│                                                                        │
│  answer-quality lift: +2.5 points     context: -73% tokens            │
└──────────────────────────────────────────────────────────────────────┘
```

> Your numbers will vary a little run-to-run — it's a real model, not a fixture.

**This is a technique, not a platform lock-in.** AutoRAG = "search your RAG config
against an eval set instead of guessing." The whole optimizer is a few hundred
lines of pure Python you own (~320 in `autorag.py`, ~750 across the whole lab). Run it on your laptop, point the winning config at your
LangChain / LlamaIndex / vector-DB stack, and scale the *same code* to a vLLM
endpoint on Red Hat OpenShift AI by changing one env var. **→ See [`ADOPT.md`](ADOPT.md)**
or the **"Adopt in your stack"** tab in the playground.

---

## Scope: a minimal, honest demo

This runs the **real** AutoRAG loop — define a config space, evaluate every config
against an eval set, score it, pick the winner — with real embeddings, real
retrieval, and a real local 1B model generating the answers. Nothing is scripted
or hardcoded (the only fake path is the opt-in `BACKEND=stub` for offline CI).

It is **not** the full open-source [AutoRAG](https://github.com/Marker-Inc-Korea/AutoRAG)
project or a production platform. To keep it laptop-sized and readable, this demo:

- sweeps **4 knobs** (chunk size, overlap, top-k, retriever) — full tooling also
  sweeps embedding models, rerankers, query expansion/HyDE, and prompt templates;
- does an **exhaustive grid search** rather than greedy node-wise optimization;
- scores on a **deterministic gold-phrase retrieval metric** (fast, no LLM in the
  loop) rather than an LLM-judge for faithfulness/answer-correctness;
- uses an **in-memory vector store** rather than a managed vector DB.

So: the method is genuine and running; the search space and metrics are a
deliberately small slice. The code is structured so you can widen the grid or swap
the metric/store without touching the loop.

## The pain point this fixes

RAG has a dozen knobs — chunk size, overlap, top-k, dense vs. lexical vs. hybrid
retrieval, rerankers — and the right combination is **corpus-specific**. The
default workflow is:

1. Pick chunk size = "whatever the tutorial used."
2. Set top-k high "to be safe."
3. Eyeball five answers, decide it looks good, ship it.

That ships a **guess**. And on a small, cheap, locally-hosted model the guess
hurts twice as much: a 1B model handed 1,300 words of loosely-related policy text
will confidently pull the **wrong** number out of it. This lab demonstrates that
failure live — and then fixes it by *measuring* instead of guessing.

**AutoRAG** = sweep the config space against a held-out eval set, score each
config, keep the winner. That's it. The intelligence is in scoring the right
thing, cheaply.

## What it actually does

- **Indexes** a 16-document bank card-disputes knowledge base.
- **Sweeps** 16 (quick) or 144 (`--full`) RAG configs.
- Scores each config on **retrieval** metrics with *no LLM calls* — fast and
  free — because retrieval is what RAG tuning actually changes:
  - `context_recall` — did we retrieve the chunk containing the answer?
  - `mrr` — how near the top did it rank?
  - `avg_ctx_words` — how much text did we force into the model? (noise + cost)
- **Then** runs the real 1B model on the naive baseline vs. the winning config so
  you can read the end-to-end answer-quality lift in actual generated text.

Why score retrieval, not generation, during the sweep? Because generation is the
expensive part. Scoring 144 configs on retrieval takes seconds; scoring them on
generation would take an hour. You spend the LLM budget once, on the final A/B.

## Run it

### Playground (browser) — easiest way to poke at it

```bash
python3 serve.py            # → http://localhost:7700  (PORT=8800 to change)
```

A tiny local web UI (pure stdlib, nothing to install) over the *same* pipeline:

- **Play** — drag the chunk-size / overlap / top-k sliders, pick a retriever, ask
  a question, and read the 1B model's answer plus the exact chunks it retrieved.
- **Baseline vs AutoRAG** — same question, both configs, side by side: watch the
  naive baseline's noisy context produce a worse (or wrong) answer.
- **Run the sweep** — score every config in the browser and set the winner.

### Option A — Ollama (laptop-first, zero Python deps)

```bash
# 1. install + start Ollama:  https://ollama.com
ollama pull llama3.2:1b
ollama pull nomic-embed-text

# 2. run the lab (uses Ollama by default)
python3 autorag.py sweep      # search configs, print the leaderboard
python3 autorag.py demo       # baseline vs winner on the real 1B model
python3 autorag.py ask "How long do I have to dispute a duplicate charge?"
```

Fits in **well under 8 GB** — `llama3.2:1b` is ~1.3 GB and `nomic-embed-text` is
~0.3 GB. No `pip install` anything; the client is pure standard library.

### Option B — any OpenAI-compatible endpoint (vLLM, llm-d, …)

The same code, pointed at an OpenAI-style `/v1` server — for example a **vLLM
`InferenceService` on Red Hat OpenShift AI**:

```bash
export BACKEND=openai
export OPENAI_BASE=https://my-model.apps.cluster.example.com/v1
export OPENAI_KEY=...           # token, or EMPTY for a local vLLM
export CHAT_MODEL=...           # served model name
python3 autorag.py demo
```

### Option C — no model at all (offline / CI)

```bash
BACKEND=stub python3 autorag.py sweep
```

A deterministic hashed-embedding stub so the harness — and your CI — runs with no
model present.

## Reading the output

- **Recall saturates** on a clean corpus: naive and tuned both find the answer.
  That's expected and honest. The win is *elsewhere*.
- **MRR + context size** are where AutoRAG separates configs: same recall, but the
  winner ranks the answer at the top and feeds the model **~¼ the context**.
- **The `demo` is the payoff**: less, better-ranked context is the difference
  between the 1B model answering "I don't know" / the wrong dollar amount and
  answering correctly — shown side by side on the questions where it mattered most.

## How this maps to Red Hat OpenShift AI

This lab is deliberately laptop-sized, but every piece has a production analogue —
you tune locally and scale the *same* contract:

| Laptop (this lab)            | Red Hat OpenShift AI                                  |
| ---------------------------- | ---------------------------------------------------- |
| `llama3.2:1b` via Ollama     | the same model served by **vLLM** on **KServe**      |
| one process, one GPU         | **llm-d** distributed inference for throughput/scale |
| in-memory vector store       | a managed vector DB / the platform's RAG stack       |
| this eval set + `score.py`   | **Eval Hub** collections run on every candidate      |
| `BACKEND=openai`             | a vLLM `InferenceService` endpoint — no code change  |

The portability *is* the point: the OpenAI-compatible contract means the harness
you debugged on a 1B model on your laptop is the harness that runs against a
70B model on the cluster.

## Composability (where this lab plugs into the others)

AutoRAG cleans up the **retrieval** half of "connecting a model to data." It sits
next to the other labs in this repo:

- **Eval Hub** — the scoring here is a tiny eval harness; the real one is a
  reusable collection you run on every model + RAG candidate.
- **An AutoML / XGBoost verifier** — a cheap classifier that scores whether a
  generated answer is grounded, used as a *judge* so you can sweep without an LLM
  in the loop, and as a runtime guard.
- **Inference-Time Scaling** — once retrieval is clean, ITS (best-of-N, etc.)
  squeezes more accuracy out of the same 1B model. Clean data in → ITS has
  something good to work with.

## Files

```
autorag/
  autorag.py        the sweep + A/B CLI (start here)
  model_client.py   Ollama / OpenAI-compatible / stub — pure stdlib, swappable
  rag/
    chunk.py        load + chunk the corpus (the first knobs to sweep)
    store.py        pure-Python vector store: dense, lexical (BM25-lite), hybrid
    pipeline.py     index -> retrieve -> generate
  eval/
    qa.jsonl        21 graded questions with gold facts
    score.py        retrieval metrics + answer-quality scoring
  corpus/           16 bank card-dispute policy docs
```
