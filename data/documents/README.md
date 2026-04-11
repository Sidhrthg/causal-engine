# RAG document corpus

Documents here are indexed for **RAG (Retrieval-Augmented Generation)** and used by:

- **Gradio app** → “RAG – Search Documents” tab  
- **Validate with RAG** → historical context for validation reports  

The corpus is **topic-agnostic**: you can add documents for **non-mineral scenarios** (e.g. energy, agriculture, policy). Upload or drop `.txt`/`.md` into `data/documents/` (or use the app upload), then rebuild the index.

## How to add documents

1. **Add files**  
   Put **`.txt`** or **`.md`** files anywhere under `data/documents/`.  
   Subfolders are fine (e.g. `usgs/`, `policy/`, `trade_reports/`, `uploaded/`).  
   **Or** use the Gradio app: **RAG – Search Documents** → Upload files → **Save to corpus** → **Rebuild search index**.

2. **Rebuild the index** (from the project root):
   ```bash
   python scripts/index_rag_documents.py
   ```
   This updates `data/documents/index.json` and (if possible) `data/documents/embeddings.pkl`.

3. **Use in the app**  
   Open the Gradio app and use the **“RAG – Search Documents”** tab to search, or run **“Validate with RAG”** on a run directory.

## PDFs

The RAG indexer only reads **.txt** and **.md**. To use PDFs:

- Convert them to text (e.g. copy-paste, or `pdftotext`), or  
- Use `scripts/ingest_docs.py --path data/documents --out artifacts/knowledge/index.jsonl` for the separate TF-IDF index (used by some other scripts, not the Gradio RAG tab).
