"""
The RAG pipeline itself: index -> retrieve -> generate.

A 'config' is a dict of knobs:
  chunk_size, chunk_overlap, top_k, retriever ('dense'|'lexical'|'hybrid')

Indexing embeds chunks. Embeddings are the expensive part, so the harness
caches them per (chunk_size, chunk_overlap) — see autorag.py.
"""

import model_client as mc
from rag.store import VectorStore


SYSTEM = (
    "You answer strictly from the provided context. Give the answer directly in "
    "one or two sentences, and quote the specific figure (number of days, dollar "
    "amount, reason code, name) when the context provides one. "
    "If the context does not contain the answer, reply with exactly: I don't know. "
    "Do not restate the question. Do not add disclaimers, caveats about the user's "
    "specific situation, or suggestions to contact anyone unless the context itself "
    "says to. No preamble — start with the answer. Reply in plain prose: no "
    "bullet points, no markdown, no bold."
)


def build_store(chunks, embeddings):
    store = VectorStore()
    for ch, vec in zip(chunks, embeddings):
        store.add(ch["id"], ch["text"], ch["doc"], vec)
    return store


def retrieve(store, query, query_vec, config):
    r = config["retriever"]
    k = config["top_k"]
    if r == "dense":
        hits = store.dense(query_vec, k)
    elif r == "lexical":
        hits = store.lexical(query, k)
    else:
        hits = store.hybrid(query_vec, query, k)
    return [c for _, c in hits]


def build_prompt(query, retrieved):
    context = "\n\n".join(f"[{c['doc']}] {c['text']}" for c in retrieved)
    return f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"


def answer(store, query, query_vec, config):
    retrieved = retrieve(store, query, query_vec, config)
    prompt = build_prompt(query, retrieved)
    text = mc.chat(prompt, system=SYSTEM, temperature=0.0)
    return text, retrieved
