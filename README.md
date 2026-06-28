# OpenShift AI Labs

Small, **actually-runnable** developer labs for connecting a model to your data —
the unglamorous, high-leverage work of getting clean inputs and outputs around an
LLM. Each lab runs **locally first**, on a laptop, against a **1B-class model**
(≤ 8 GB), using Ollama or any OpenAI-compatible endpoint. No cloud account, no
slideware — `git clone` and run.

The premise: you shouldn't have to provision a cluster to *learn* how a technique
works or whether it helps your data. Prove it on a 1B model on your laptop, then
scale the **exact same contract** to a real model on **Red Hat OpenShift AI** —
vLLM on KServe for serving, llm-d for distributed inference, Eval Hub for
evaluation. Laptop and cluster run the same code; only `BACKEND` changes.

> These labs are about **model customization and clean data I/O** — preparing a
> model and its data plumbing so inputs and outputs are trustworthy. They are not
> an agent framework; they're the layer underneath one.

## The labs

| Lab           | What it proves on a 1B model                                            | Status      |
| ------------- | ----------------------------------------------------------------------- | ----------- |
| **`autorag/`**| Hand-tuned RAG ships a guess; an automated config sweep beats it — and tighter context is the difference between a 1B model answering right or wrong. | ✅ runnable |
| `evalhub/`    | A reusable eval collection you run on every model + config candidate.   | planned     |
| `sdghub/`     | Generate synthetic training/eval data to cover gaps your real data misses. | planned  |
| `redteam/`    | Probe a small model for jailbreaks/safety failures before you ship it.  | planned     |

They're **composable**: AutoRAG cleans retrieval, Eval Hub scores every change,
SDG Hub manufactures the data the others need, an AutoML/XGBoost classifier acts
as a cheap verifier/judge, and Inference-Time Scaling squeezes more out of the
same small model once the data going in is clean.

## Requirements

- **Python 3.9+** (standard library only — the labs install nothing via pip).
- **A model backend**, either:
  - [**Ollama**](https://ollama.com) (recommended, laptop-first), or
  - any **OpenAI-compatible** `/v1` endpoint (e.g. a **vLLM** server, including a
    Red Hat OpenShift AI `InferenceService`).

```bash
# laptop-first setup
ollama pull llama3.2:1b
ollama pull nomic-embed-text
```

## Quickstart

```bash
cd autorag
python3 autorag.py sweep    # find the best RAG config by measuring, not guessing
python3 autorag.py demo     # watch the 1B model answer better on tighter context
```

Each lab has its own README with the full story, the pain point it targets, and
how it maps to Red Hat OpenShift AI.

## Why local-first

A 1B model is an honest forcing function. Tricks that only work because a frontier
model is quietly covering for sloppy data plumbing fall apart on a 1B — so if your
RAG, your eval, and your guardrails make a 1B answer correctly, they'll hold up
when you scale to a bigger model on the cluster. Build it small, prove it, scale
the same contract.
