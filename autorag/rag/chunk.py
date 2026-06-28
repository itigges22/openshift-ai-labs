"""
Document loading + chunking. Chunk size / overlap are the first knobs AutoRAG
sweeps — they change retrieval quality more than almost anything else, and the
'right' value is corpus-specific, which is exactly why you tune it instead of
guessing.
"""

import glob
import os


def load_corpus(corpus_dir):
    """Return list[dict]: {doc, text}. One entry per markdown file."""
    docs = []
    for path in sorted(glob.glob(os.path.join(corpus_dir, "*.md"))):
        with open(path, encoding="utf-8") as f:
            docs.append({"doc": os.path.basename(path), "text": f.read()})
    return docs


def chunk_text(text, size, overlap):
    """Word-based sliding window. size/overlap are in words (cheap + portable)."""
    words = text.split()
    if not words:
        return []
    if size <= 0:
        return [text]
    step = max(1, size - overlap)
    out = []
    for start in range(0, len(words), step):
        piece = words[start:start + size]
        if piece:
            out.append(" ".join(piece))
        if start + size >= len(words):
            break
    return out


def build_chunks(docs, size, overlap):
    """Flatten a corpus into id'd chunks for a given chunking config."""
    chunks = []
    for d in docs:
        for i, piece in enumerate(chunk_text(d["text"], size, overlap)):
            chunks.append({"id": f"{d['doc']}#{i}", "doc": d["doc"], "text": piece})
    return chunks
