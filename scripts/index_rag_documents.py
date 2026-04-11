#!/usr/bin/env python3
"""
Build the RAG document index used by the Gradio app and validate_with_real_rag.

Place .txt and .md files in data/documents/ (subfolders OK), then run:
  python scripts/index_rag_documents.py

Creates/updates:
  - data/documents/index.json   (chunks + keyword index)
  - data/documents/embeddings.pkl (optional; for semantic search)
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.minerals.rag_retrieval import SimpleRAGRetriever


def main():
    docs_dir = PROJECT_ROOT / "data" / "documents"
    index_path = docs_dir / "index.json"

    if not docs_dir.exists():
        print(f"❌ Documents directory not found: {docs_dir}")
        print("   Create it and add .txt or .md files, then run this script again.")
        return 1

    retriever = SimpleRAGRetriever(
        documents_dir=str(docs_dir),
        index_path=str(index_path),
    )

    # Build embeddings only if ANTHROPIC_API_KEY or VOYAGE_API_KEY / sentence-transformers available
    build_embeddings = True
    retriever.ingest_documents(force_reindex=True, build_embeddings=build_embeddings)

    if len(retriever.chunks) == 0:
        print("⚠️  No .txt or .md files found in data/documents/")
        return 1

    print(f"\n✅ Done. Indexed {len(retriever.chunks)} chunks.")
    print(f"   Index: {index_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
