#!/usr/bin/env python3
"""CLI: build HippoRAG index under ``<docs-dir>/hipporag_index/``. See ``docs/HIPPORAG.md``."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    from src.minerals.hipporag_retrieval import HippoRAGRetriever, hipporag_available
except ImportError as e:
    print("❌ hipporag_retrieval failed:", e)
    print("   Install: python3 -m pip install hipporag  or  pip install -e \".[hipporag]\"")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Build HippoRAG index from documents")
    parser.add_argument(
        "--docs-dir",
        default="data/documents",
        help="Documents directory (.txt, .md)",
    )
    parser.add_argument(
        "--save-dir",
        default="",
        help="HippoRAG save dir (default: <docs-dir>/hipporag_index)",
    )
    args = parser.parse_args()

    if not hipporag_available():
        print("❌ hipporag not installed. Run: python3 -m pip install hipporag  or  pip install -e \".[hipporag]\"")
        sys.exit(1)

    docs_dir = Path(args.docs_dir)
    if not docs_dir.exists():
        print(f"❌ Documents dir not found: {docs_dir}")
        sys.exit(1)

    save_dir = args.save_dir or str(docs_dir / "hipporag_index")
    retriever = HippoRAGRetriever(documents_dir=str(docs_dir), save_dir=save_dir)
    msg = retriever.index()
    print(msg)
    if "❌" in msg:
        sys.exit(1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
