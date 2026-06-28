"""
Scoring for the AutoRAG sweep.

Two layers:

  Retrieval metrics (no LLM, fast) — these drive the sweep because retrieval is
  what AutoRAG actually tunes, and you can score thousands of configs without
  paying for a single generation:
    * context_recall : fraction of questions whose gold fact was retrieved
    * mrr            : mean reciprocal rank of the first chunk with the gold fact

  Answer metric (needs the LLM) — used only for the final baseline-vs-winner
  A/B so you can see the end-to-end quality lift in real generated text:
    * answer_score   : blend of embedding similarity to the reference answer
                       and keyword coverage of the reference
"""

import math


def _norm(text):
    # normalize markdown emphasis and the dash family (— – -) so gold facts
    # match regardless of how they were typed.
    for d in ("—", "–", "−"):
        text = text.replace(d, "-")
    return " ".join(text.replace("*", "").lower().split())


def _tokens(text):
    return [w for w in "".join(c.lower() if c.isalnum() else " " for c in text).split() if w]


def gold_in_chunks(gold, chunks):
    """Return 1-based rank of first chunk containing the gold fact, else 0."""
    g = _norm(gold)
    for i, c in enumerate(chunks, start=1):
        if g in _norm(c["text"]):
            return i
    return 0


def retrieval_metrics(per_query_ranks, ctx_words=None):
    """per_query_ranks: list[int] (0 = miss). ctx_words: words retrieved/query.

    context_recall : did we retrieve the answer at all (the must-have).
    mrr            : how high the answer ranked (cleaner top-of-context).
    avg_ctx_words  : how much text we forced into the model (noise + cost).
                     Two configs with equal recall are NOT equal — the one that
                     answers with less context is cheaper and easier for a small
                     model to get right.
    """
    n = len(per_query_ranks) or 1
    hits = sum(1 for r in per_query_ranks if r > 0)
    recall = hits / n
    mrr = sum((1.0 / r) for r in per_query_ranks if r > 0) / n
    avg_ctx = (sum(ctx_words) / len(ctx_words)) if ctx_words else 0.0
    return {
        "context_recall": recall, "mrr": mrr, "avg_ctx_words": avg_ctx,
        "hits": hits, "n": len(per_query_ranks),
    }


def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(y * y for y in b)) or 1.0
    return dot / (na * nb)


def keyword_coverage(reference, candidate):
    ref = set(_tokens(reference))
    if not ref:
        return 0.0
    cand = set(_tokens(candidate))
    stop = {"the", "a", "an", "of", "to", "is", "are", "and", "or", "for", "in",
            "on", "within", "be", "must", "it", "that", "this", "with", "by"}
    ref -= stop
    if not ref:
        return 0.0
    return len(ref & cand) / len(ref)


def answer_score(reference, candidate, ref_vec, cand_vec):
    """Blend semantic similarity with keyword coverage -> 0..1."""
    sim = (cosine(ref_vec, cand_vec) + 1) / 2  # map [-1,1] -> [0,1]
    cov = keyword_coverage(reference, candidate)
    return 0.6 * sim + 0.4 * cov
