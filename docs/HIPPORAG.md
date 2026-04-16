# HippoRAG Integration (graph-based retrieval)

**Graph-based RAG** using [HippoRAG](https://github.com/OSU-NLP-Group/HippoRAG) is integrated and **used by default** when the index exists.

---

## How to get HippoRAG (install the Python package)

You install the **`hipporag`** library from **PyPI** inside this project’s virtualenv. You do **not** need to clone the upstream HippoRAG repo for normal use.

1. **Use Python 3.10–3.12** (this repo does not support 3.14). Activate your venv:
   ```bash
   cd /path/to/Causal-engine
   source .venv/bin/activate
   ```

2. **Install via this repo’s optional extra** (pins `hipporag` in `pyproject.toml`):
   ```bash
   pip install -e ".[hipporag]"
   ```
   **Or** install only the package:
   ```bash
   pip install hipporag
   ```

3. **Verify** the import:
   ```bash
   python -c "from hipporag import HippoRAG; print('hipporag OK')"
   ```

4. **If `pip` fails** (torch / numpy / vllm conflicts): skip HippoRAG — the app falls back to classic RAG. Options: resolve conflicts in a **separate venv** used only for indexing, or use **classic search** in the UI / `USE_HIPPORAG=0`. See **Setup** below for NumPy notes.

5. **After the package works**, you still need a **local index** under `data/documents/hipporag_index/` (not in git — see **Setup**). Build with `python scripts/index_hipporag.py` or Gradio **“Build HippoRAG index”**.

**Upstream source / paper:** [OSU-NLP-Group/HippoRAG](https://github.com/OSU-NLP-Group/HippoRAG)

---

## When it’s used

- **Just RAG tab:** Search **uses HippoRAG** whenever the HippoRAG index is built (`data/documents/hipporag_index/`). Build the index once via **“Build HippoRAG index”** or `python scripts/index_hipporag.py`. Check **“Use classic keyword/semantic search only”** to disable HippoRAG.
- **Validate with RAG:** Uses HippoRAG retrieval by default when the package is installed and the index exists. Set **`USE_HIPPORAG=0`** (or `false`/`no`) to force classic retrieval.

Your **domain Knowledge Graph** (supply chain, causal relations) is unchanged; HippoRAG builds a **separate document graph** from the corpus for retrieval.

---

## Setup

1. **Install the `hipporag` package** (see **How to get HippoRAG** above). Typical one-liner with Gradio:
   ```bash
   brew install python@3.12   # if needed
   cd /path/to/Causal-engine
   python3.12 -m venv .venv
   source .venv/bin/activate
   pip install -e ".[ui,hipporag]"    # Gradio + HippoRAG; drop hipporag if pip conflicts
   ```
   HippoRAG pins strict deps (e.g. `torch==2.5.1`). If install fails, skip the extra — classic RAG still works.

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
