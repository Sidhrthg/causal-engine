# HippoRAG Integration (graph-based retrieval)

**Graph-based RAG** using [HippoRAG](https://github.com/OSU-NLP-Group/HippoRAG) is integrated and **used by default** when the index exists.

---

## When it’s used

- **Just RAG tab:** Search **uses HippoRAG** whenever the HippoRAG index is built (`data/documents/hipporag_index/`). Build the index once via **“Build HippoRAG index”** or `python scripts/index_hipporag.py`. Check **“Use classic keyword/semantic search only”** to disable HippoRAG.
- **Validate with RAG:** Uses HippoRAG retrieval by default when the package is installed and the index exists. Set **`USE_HIPPORAG=0`** (or `false`/`no`) to force classic retrieval.

Your **domain Knowledge Graph** (supply chain, causal relations) is unchanged; HippoRAG builds a **separate document graph** from the corpus for retrieval.

---

## Setup

1. **Use a virtual environment with Python 3.10–3.12** (not 3.14). On macOS, Homebrew’s `python3` is often 3.14, so use 3.12 explicitly:
   ```bash
   brew install python@3.12   # if needed
   cd /path/to/Causal-engine
   python3.12 -m venv .venv
   source .venv/bin/activate
   pip install -e ".[ui]"     # project + Gradio
   pip install hipporag      # optional; if this hits dependency conflicts, skip it — classic RAG still works
   ```
   HippoRAG pins strict deps (e.g. `torch==2.5.1`). If `pip install hipporag` fails with resolution errors, you can skip HippoRAG; the app uses classic keyword/semantic retrieval and works without it.

   **NumPy conflict:** HippoRAG/vllm declare `numpy<2`, while DoWhy and cvxpy (used by the causal-engine) need `numpy>=2`. We recommend keeping **numpy 2.x** (e.g. `pip install "numpy>=2.0,<2.4"`) so the main app and DoWhy work. vllm often runs fine with NumPy 2.x despite the constraint; if HippoRAG or vllm fails at runtime, use **classic RAG** (uncheck “Use HippoRAG” / set `USE_HIPPORAG=0`) or use a **separate env** only for HippoRAG indexing.

2. **Indexing** needs an LLM and an embedding model (for entity/relation extraction and embeddings):
   - **OpenAI:** set `OPENAI_API_KEY`. Defaults: `gpt-4o-mini`, `nvidia/NV-Embed-v2`.
   - **vLLM:** run a local vLLM server and set `VLLM_BASE_URL` (e.g. `http://localhost:8000/v1`); HippoRAG will use it for indexing.

3. **Build the index** once (reads `data/documents/`):
   - **In Gradio:** Just RAG tab → **“Build HippoRAG index”**.
   - **CLI:** `python scripts/index_hipporag.py [--docs-dir data/documents] [--save-dir ...]`.

4. **Use it:**
   - **Just RAG:** Check “Use HippoRAG …” and search.
   - **Validate with RAG:** `USE_HIPPORAG=1 python -m scripts.validate_with_real_rag --run-dir <path>` (or run from Gradio with the env set).

---

## Files

- **`src/minerals/hipporag_retrieval.py`** – Optional wrapper: `HippoRAGRetriever`, `get_retriever()`, `hipporag_available()`.
- **`scripts/index_hipporag.py`** – CLI to build the HippoRAG index.
- **App:** Just RAG has “Build HippoRAG index” and “Use HippoRAG (graph-based retrieval)”.
- **Validator:** Uses `get_retriever(use_hipporag=...)` when `USE_HIPPORAG` is set.

If `hipporag` is not installed, the app and validator fall back to the default retriever (keyword/semantic); no error.
