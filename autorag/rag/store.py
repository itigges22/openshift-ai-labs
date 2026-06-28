"""
A pure-Python in-memory vector store with dense + lexical retrieval.

No FAISS, no chromadb, no numpy — just enough to be honest about what a
retriever does and fast enough for a laptop corpus. Swap this for a managed
vector DB in production; the AutoRAG sweep logic above it doesn't change.
"""

import math
from collections import Counter


def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(y * y for y in b)) or 1.0
    return dot / (na * nb)


def _tokens(text):
    return [w for w in "".join(c.lower() if c.isalnum() else " " for c in text).split() if w]


class VectorStore:
    """Holds chunks + their embeddings; supports dense, lexical, hybrid search."""

    def __init__(self):
        self.chunks = []        # list[dict]: {id, text, doc, vec, toks}
        self._df = Counter()    # document frequency for idf
        self._n = 0

    def add(self, chunk_id, text, doc, vec):
        toks = _tokens(text)
        self.chunks.append(
            {"id": chunk_id, "text": text, "doc": doc, "vec": vec, "toks": toks}
        )
        for t in set(toks):
            self._df[t] += 1
        self._n += 1

    # -- dense (semantic) ------------------------------------------------- #
    def dense(self, query_vec, top_k):
        scored = [(cosine(query_vec, c["vec"]), c) for c in self.chunks]
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[:top_k]

    # -- lexical (BM25-lite) --------------------------------------------- #
    def lexical(self, query_text, top_k):
        q = _tokens(query_text)
        scores = []
        for c in self.chunks:
            tf = Counter(c["toks"])
            s = 0.0
            for term in q:
                if term in tf:
                    idf = math.log(1 + (self._n - self._df[term] + 0.5) / (self._df[term] + 0.5))
                    s += idf * (tf[term] / (len(c["toks"]) + 1))
            scores.append((s, c))
        scores.sort(key=lambda x: x[0], reverse=True)
        return scores[:top_k]

    # -- hybrid (reciprocal rank fusion of dense + lexical) -------------- #
    def hybrid(self, query_vec, query_text, top_k, k=60):
        dense_rank = {c["id"]: i for i, (_, c) in enumerate(self.dense(query_vec, len(self.chunks)))}
        lex_rank = {c["id"]: i for i, (_, c) in enumerate(self.lexical(query_text, len(self.chunks)))}
        fused = {}
        for c in self.chunks:
            r = 0.0
            if c["id"] in dense_rank:
                r += 1.0 / (k + dense_rank[c["id"]])
            if c["id"] in lex_rank:
                r += 1.0 / (k + lex_rank[c["id"]])
            fused[c["id"]] = (r, c)
        ranked = sorted(fused.values(), key=lambda x: x[0], reverse=True)
        return ranked[:top_k]
