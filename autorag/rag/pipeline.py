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
    "You are a bank's card-disputes support assistant. Answer ONLY from the "
    "provided context. If the context does not contain the answer, say you don't "
    "know. Be concise and cite policy terms when relevant."
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
