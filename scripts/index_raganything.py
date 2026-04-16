#!/usr/bin/env python3
"""
CLI: build RAGAnything multimodal index under ``<docs-dir>/raganything_index/``.

Processes PDFs (with charts/tables/images), Word docs, plain text, and images
into a unified multimodal knowledge graph via LightRAG + MinerU.

Requires OPENAI_API_KEY (for LLM entity extraction + embeddings).

Usage
-----
    # Index all PDFs in the default documents directory:
    python scripts/index_raganything.py

    # Index only specific files:
    python scripts/index_raganything.py --files data/documents/usgs_books/mcs_2024_complete.pdf

    # Index a specific subdirectory, PDFs only:
    python scripts/index_raganything.py --docs-dir data/documents/usgs --ext .pdf

    # Disable image/vision processing (faster, text+tables only):
    python scripts/index_raganything.py --no-vision

    # Use a different working directory:
    python scripts/index_raganything.py --working-dir /tmp/my_index
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()

try:
    from src.minerals.raganything_retrieval import RAGAnythingRetriever, raganything_available
except ImportError as e:
    print("❌ raganything_retrieval import failed:", e)
    sys.exit(1)

SUPPORTED_EXTS = {
    ".pdf", ".txt", ".md",
    ".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".gif", ".webp",
    ".doc", ".docx", ".ppt", ".pptx", ".xls", ".xlsx",
}


def collect_files(docs_dir: Path, exts: set, recursive: bool) -> list[Path]:
    pattern = "**/*" if recursive else "*"
    files = [
        f for f in docs_dir.glob(pattern)
        if f.is_file()
        and f.suffix.lower() in exts
        and "raganything_index" not in str(f)
        and "hipporag_index" not in str(f)
    ]
    return sorted(files)


def main():
    parser = argparse.ArgumentParser(
        description="Build RAGAnything multimodal index from documents"
    )
    parser.add_argument(
        "--docs-dir",
        default="data/documents",
        help="Root documents directory to index (default: data/documents)",
    )
    parser.add_argument(
        "--working-dir",
        default="",
        help="RAGAnything working dir (default: <docs-dir>/raganything_index)",
    )
    parser.add_argument(
        "--files",
        nargs="+",
        metavar="FILE",
        help="Index specific files instead of a directory",
    )
    parser.add_argument(
        "--ext",
        nargs="+",
        default=None,
        metavar="EXT",
        help="Only index files with these extensions, e.g. --ext .pdf .txt",
    )
    parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="Do not recurse into subdirectories",
    )
    parser.add_argument(
        "--no-vision",
        action="store_true",
        help="Disable vision/image processing (faster, text+tables only)",
    )
    parser.add_argument(
        "--llm-model",
        default="gpt-4o-mini",
        help="OpenAI LLM model for entity extraction (default: gpt-4o-mini)",
    )
    parser.add_argument(
        "--embed-model",
        default="text-embedding-3-large",
        help="OpenAI embedding model (default: text-embedding-3-large)",
    )
    args = parser.parse_args()

    if not raganything_available():
        print("❌ raganything not installed. Run: pip install raganything")
        sys.exit(1)

    docs_dir = Path(args.docs_dir)
    working_dir = Path(args.working_dir) if args.working_dir else docs_dir / "raganything_index"

    # Collect files
    if args.files:
        files = [Path(f) for f in args.files]
        missing = [f for f in files if not f.exists()]
        if missing:
            print(f"❌ Files not found: {missing}")
            sys.exit(1)
    else:
        if not docs_dir.exists():
            print(f"❌ Documents dir not found: {docs_dir}")
            sys.exit(1)
        exts = set(args.ext) if args.ext else SUPPORTED_EXTS
        files = collect_files(docs_dir, exts, recursive=not args.no_recursive)
        if not files:
            print(f"⚠  No supported files found in {docs_dir}")
            sys.exit(0)

    print(f"📂 Documents : {docs_dir if not args.files else 'explicit list'}")
    print(f"📁 Index dir : {working_dir}")
    print(f"🤖 LLM       : {args.llm_model}")
    print(f"🔢 Embed     : {args.embed_model}")
    print(f"👁  Vision    : {'disabled' if args.no_vision else 'enabled (gpt-4o-mini)'}")
    print(f"📄 Files     : {len(files)}")
    print()

    # Show file list (truncated)
    for f in files[:10]:
        print(f"   {f}")
    if len(files) > 10:
        print(f"   ... and {len(files) - 10} more")
    print()

    # Build retriever
    try:
        ret = RAGAnythingRetriever(
            working_dir=str(working_dir),
            llm_model_name=args.llm_model,
            embedding_model_name=args.embed_model,
            vision_model_name=None if args.no_vision else "gpt-4o-mini",
        )
    except Exception as e:
        print(f"❌ Failed to initialise RAGAnythingRetriever: {e}")
        sys.exit(1)

    # Index
    print("⏳ Indexing (this calls OpenAI — may take a while for large corpora)…\n")
    status = ret.index([str(f) for f in files])
    print(f"\n✅ {status}")
    print(f"\nGraph file: {working_dir}/graph_chunk_entity_relation.graphml")
    print("\nTo use in code:")
    print(f"  ret = RAGAnythingRetriever(working_dir='{working_dir}')")
    print("  chunks = ret.retrieve('your query here')")
    print("  ret.bridge_to_causal_kg(kg)  # merge into CausalKnowledgeGraph")


if __name__ == "__main__":
    main()
