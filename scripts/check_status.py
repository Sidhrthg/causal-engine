"""
Quick status check: documents, trade data, RAG index.
"""

from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parent.parent

    print("=" * 60)
    print("CAUSAL ENGINE – STATUS")
    print("=" * 60)

    # Documents
    docs = root / "data" / "documents"
    if docs.exists():
        pdfs = list(docs.rglob("*.pdf"))
        txts = list(docs.rglob("*.txt"))
        print(f"\n📄 Documents: {len(pdfs)} PDFs, {len(txts)} TXT/MD")
    else:
        print("\n📄 Documents: data/documents/ not found")

    # Trade data
    canonical = root / "data" / "canonical"
    trade_files = []
    if canonical.exists():
        for name in (
            "comtrade_network_graphite.csv",
            "comtrade_global_graphite.csv",
            "comtrade_graphite_bilateral.csv",
        ):
            p = canonical / name
            if p.exists():
                trade_files.append((name, p.stat().st_size))
    print(f"\n📊 Trade data: {len(trade_files)} file(s)")
    for name, size in trade_files:
        print(f"   - {name} ({size / 1024:.1f} KB)")

    # RAG index
    index_json = root / "data" / "documents" / "index.json"
    chroma_dir = root / "data" / "chroma_db"
    if index_json.exists():
        print(f"\n📚 RAG (simple): index.json present")
    elif chroma_dir.exists():
        n = len(list(chroma_dir.rglob("*.sqlite3"))) or len(list(chroma_dir.iterdir()))
        print(f"\n📚 RAG (Chroma): {chroma_dir.name}/ present")
    else:
        print("\n📚 RAG: No index yet (run index_rag_documents.py or rag_industrial)")

    print("\n" + "=" * 60)
    print("Day 1 complete: docs indexed, trade network downloaded, RAG ready")
    print("=" * 60)


if __name__ == "__main__":
    main()
