"""
Tiny, zero-dependency client for local LLM + embedding backends.

Two backends, same interface, picked by env vars (no code change needed):

  * Ollama (default)        — laptop-first. `ollama serve` on :11434.
  * OpenAI-compatible        — vLLM / llm-d / any OpenAI-style /v1 endpoint.

This is deliberately built on urllib from the standard library so the whole
lab is `git clone && python autorag.py` with nothing to install. In a Red Hat
OpenShift AI cluster you'd point BACKEND=openai at a vLLM `InferenceService`
and the exact same code runs at scale — that portability is the point.

A `stub` backend is included so the harness is testable without any model
running (deterministic, offline).

Env:
  BACKEND        ollama | openai | stub          (default: ollama)
  CHAT_MODEL     default: llama3.2:1b
  EMBED_MODEL    default: nomic-embed-text
  OLLAMA_HOST    default: http://localhost:11434
  OPENAI_BASE    default: http://localhost:8000/v1
  OPENAI_KEY     default: EMPTY
"""

import json
import os
import urllib.request
import urllib.error

BACKEND = os.environ.get("BACKEND", "ollama").lower()
CHAT_MODEL = os.environ.get("CHAT_MODEL", "llama3.2:1b")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "nomic-embed-text")
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
OPENAI_BASE = os.environ.get("OPENAI_BASE", "http://localhost:8000/v1").rstrip("/")
OPENAI_KEY = os.environ.get("OPENAI_KEY", "EMPTY")


class BackendError(RuntimeError):
    pass


def _post(url, payload, headers=None, timeout=120):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    for k, v in (headers or {}).items():
        req.add_header(k, v)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as e:
        raise BackendError(
            f"Could not reach {url}: {e}. "
            f"Is the backend running? (BACKEND={BACKEND})"
        ) from e


# --------------------------------------------------------------------------- #
# Chat
# --------------------------------------------------------------------------- #
def chat(prompt, system=None, temperature=0.0, max_tokens=512):
    """Single-turn completion. Returns the model's text."""
    if BACKEND == "stub":
        return _stub_chat(prompt, system)

    if BACKEND == "ollama":
        msgs = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.append({"role": "user", "content": prompt})
        out = _post(
            f"{OLLAMA_HOST}/api/chat",
            {
                "model": CHAT_MODEL,
                "messages": msgs,
                "stream": False,
                "options": {"temperature": temperature, "num_predict": max_tokens},
            },
        )
        return out["message"]["content"].strip()

    if BACKEND == "openai":
        msgs = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.append({"role": "user", "content": prompt})
        out = _post(
            f"{OPENAI_BASE}/chat/completions",
            {
                "model": CHAT_MODEL,
                "messages": msgs,
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
            headers={"Authorization": f"Bearer {OPENAI_KEY}"},
        )
        return out["choices"][0]["message"]["content"].strip()

    raise BackendError(f"Unknown BACKEND={BACKEND!r}")


# --------------------------------------------------------------------------- #
# Embeddings
# --------------------------------------------------------------------------- #
def embed(texts):
    """Embed a list of strings -> list of float vectors."""
    if isinstance(texts, str):
        texts = [texts]

    if BACKEND == "stub":
        return [_stub_embed(t) for t in texts]

    if BACKEND == "ollama":
        vecs = []
        for t in texts:
            out = _post(
                f"{OLLAMA_HOST}/api/embeddings",
                {"model": EMBED_MODEL, "prompt": t},
            )
            vecs.append(out["embedding"])
        return vecs

    if BACKEND == "openai":
        out = _post(
            f"{OPENAI_BASE}/embeddings",
            {"model": EMBED_MODEL, "input": texts},
            headers={"Authorization": f"Bearer {OPENAI_KEY}"},
        )
        return [d["embedding"] for d in out["data"]]

    raise BackendError(f"Unknown BACKEND={BACKEND!r}")


# --------------------------------------------------------------------------- #
# Stub backend — deterministic, offline. Lets the harness be tested with no
# model. A 64-dim hashed bag-of-words embedding: good enough that lexically
# similar texts land near each other, so retrieval still "works" in tests.
# --------------------------------------------------------------------------- #
def _tokens(text):
    return [w for w in "".join(c.lower() if c.isalnum() else " " for c in text).split() if w]


def _stub_embed(text, dim=64):
    vec = [0.0] * dim
    for tok in _tokens(text):
        vec[hash(tok) % dim] += 1.0
    norm = sum(v * v for v in vec) ** 0.5 or 1.0
    return [v / norm for v in vec]


def _stub_chat(prompt, system=None):
    # Echo the most relevant-looking sentence from the supplied context so the
    # end-to-end flow produces plausible text without a real model.
    ctx = prompt
    if "Context:" in prompt:
        ctx = prompt.split("Context:", 1)[1]
    line = next((l.strip() for l in ctx.splitlines() if len(l.strip()) > 30), "")
    return f"[stub answer] {line[:200]}"


def info():
    if BACKEND == "ollama":
        where = OLLAMA_HOST
    elif BACKEND == "openai":
        where = OPENAI_BASE
    else:
        where = "offline"
    return f"backend={BACKEND} chat={CHAT_MODEL} embed={EMBED_MODEL} @ {where}"
