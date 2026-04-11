"""
Re-index RAG system with real USGS (and other) documents.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.minerals.rag_retrieval import SimpleRAGRetriever


def reindex_with_real_docs():
    """
    Re-index RAG with documents in data/documents (e.g. USGS .txt from extract_pdf_text).
    """
    print("🔄 Re-indexing RAG with documents in data/documents...")

    docs_dir = PROJECT_ROOT / "data" / "documents"
    index_path = docs_dir / "index.json"

    retriever = SimpleRAGRetriever(
        documents_dir=str(docs_dir),
        index_path=str(index_path),
    )

    retriever.ingest_documents(force_reindex=True, build_embeddings=True)

    if len(retriever.chunks) == 0:
        print("⚠️  No .txt or .md files found in data/documents/")
        return

    print(f"\n✅ Indexed {len(retriever.chunks)} chunks from documents")

    # Test retrieval
    print("\n🧪 Testing retrieval...")
    results = retriever.retrieve("graphite price volatility supply", top_k=3)

    print(f"\n📄 Retrieved {len(results)} chunks:")
    for i, chunk in enumerate(results, 1):
        source = chunk.get("metadata", {}).get("source_file", "?")
        text_preview = (chunk.get("text") or "")[:100]
        print(f"\n{i}. {source}")
        print(f"   {text_preview}...")

    print("\n✅ RAG system ready with real documents!")


if __name__ == "__main__":
    reindex_with_real_docs()
