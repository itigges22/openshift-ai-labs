# Use AutoRAG on your own RAG

AutoRAG isn't a product you buy or a platform feature you have to be on. It's a
**technique** — *search your RAG config space against an eval set instead of
guessing* — and this lab is a complete, dependency-free implementation of it. The
whole optimizer is ~200 lines you already have: `autorag.py`, `rag/`, `eval/`.

This page is how you put it on **your** retrieval stack.

## The value, in one sentence

You stop shipping a guessed RAG config and start shipping a measured one — which,
as the `demo`/Compare views show, is the difference between a small model answering
correctly on tight context and answering wrong on a pile of noise. Same recall, ~¼
the context: **cheaper, faster, and more accurate**, on any model.

## It's portable by construction

Nothing here is tied to a vendor. The only integration point with the outside
world is a 60-line `model_client.py` with three backends behind one interface:

| You have…                          | Set                                        |
| ---------------------------------- | ------------------------------------------ |
| Ollama on your laptop              | *(default)* `BACKEND=ollama`               |
| Any OpenAI-compatible server (vLLM)| `BACKEND=openai OPENAI_BASE=… CHAT_MODEL=…` |
| Nothing / CI                       | `BACKEND=stub`                             |

The same sweep you run on a 1B model on your laptop runs against a 70B model on a
**vLLM `InferenceService` on Red Hat OpenShift AI** — you change an env var, not
code. Tune locally, scale the contract.

## Adopt it in 3 steps

### 1. Bring your own corpus
Drop your documents into `corpus/` (or point `CORPUS_DIR` at them). Markdown,
text, whatever — `rag/chunk.py` just reads files.

### 2. Bring your own eval set
Replace `eval/qa.jsonl`. ~15–20 real questions is plenty. Each line:

```json
{"q": "your question", "answer": "the ideal answer", "gold": "a distinctive phrase that must be retrieved"}
```

`gold` is the one thing that makes this work without a human in the loop: a short,
verbatim phrase from the source doc that *proves* the right passage was retrieved.
That's how the sweep scores 144 configs with zero LLM calls.

### 3. Run it
```bash
python3 autorag.py sweep          # find the config that wins on YOUR data
python3 autorag.py demo           # see the answer-quality lift on your model
```

## Plug it into your existing framework

The sweep is framework-agnostic — it optimizes the four knobs that every RAG stack
exposes. Take the winning config and set these:

### LangChain
```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

# AutoRAG reports chunk size in WORDS; LangChain's splitter counts characters,
# so multiply by ~5 (or use a token splitter at ~chunk_size tokens).
splitter = RecursiveCharacterTextSplitter(
    chunk_size=WINNER_CHUNK * 5,
    chunk_overlap=WINNER_OVERLAP * 5,
)
retriever = vectordb.as_retriever(search_kwargs={"k": WINNER_TOPK})
# retriever="dense"  -> similarity search (above)
# retriever="hybrid" -> EnsembleRetriever([BM25Retriever(...), vectordb.as_retriever(...)])
```

### LlamaIndex
```python
from llama_index.core.node_parser import SentenceSplitter

splitter = SentenceSplitter(chunk_size=WINNER_CHUNK, chunk_overlap=WINNER_OVERLAP)
query_engine = index.as_query_engine(similarity_top_k=WINNER_TOPK)
```

### Your own / raw vector DB (FAISS, Chroma, pgvector, …)
The lab's `rag/store.py` is an in-memory reference store. To use your DB instead,
implement the same tiny surface and the sweep doesn't change:

```python
def index(chunks) -> store        # embed + insert
def retrieve(store, query, k) -> list[chunk]   # dense / lexical / hybrid
```

`rag/pipeline.py` shows exactly where those two functions are called. Swap the body,
keep the sweep.

## Where this goes next (composability)

Same data-quality theme, same local-first labs:

- **Eval Hub** — the `gold`-phrase scorer here is a baby eval. Graduate to a reusable
  collection you run on every model + RAG candidate, locally and on the cluster.
- **An AutoML / XGBoost verifier** — train a cheap classifier to judge "is this
  answer grounded?" and use it both as a sweep metric and a runtime guard.
- **Inference-Time Scaling** — once retrieval is clean, ITS (best-of-N, etc.) squeezes
  more accuracy from the same small model. Clean inputs first, then scale inference.

You don't need any of those to get value today. Swap your corpus and eval, run the
sweep, ship the winning config. The rest is upside.
