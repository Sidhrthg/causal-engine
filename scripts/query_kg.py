#!/usr/bin/env python3
"""CLI interface for KG-grounded causal queries.

Usage:
  python scripts/query_kg.py query "What are the causal pathways from china export controls to EV batteries?"
  python scripts/query_kg.py audit graphite
  python scripts/query_kg.py entities --type commodity
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _get_retriever_with_auto_index(
    documents_dir: str = "data/documents",
):
    """Get a retriever, preferring HippoRAG when index exists, else SimpleRAGRetriever."""
    from src.minerals.hipporag_retrieval import get_retriever, hipporag_available

    docs_dir = Path(documents_dir)
    hipporag_index = docs_dir / "hipporag_index" / "doc_meta.json"

    # Prefer HippoRAG if available and indexed
    if hipporag_available() and hipporag_index.exists():
        print("Using HippoRAG retriever (graph-based multi-hop).")
        return get_retriever(use_hipporag=True, documents_dir=str(docs_dir))

    # Fallback to SimpleRAGRetriever with auto-reindex
    print("HippoRAG index not found — falling back to SimpleRAGRetriever.")
    from src.minerals.rag_retrieval import SimpleRAGRetriever

    index_path = docs_dir / "index.json"
    retriever = SimpleRAGRetriever(
        documents_dir=str(docs_dir),
        index_path=str(index_path),
    )

    needs_reindex = False
    if not index_path.exists():
        needs_reindex = True
    else:
        index_mtime = index_path.stat().st_mtime
        text_files = list(docs_dir.rglob("*.txt")) + list(docs_dir.rglob("*.md"))
        for f in text_files:
            if f.stat().st_mtime > index_mtime:
                needs_reindex = True
                break

    if needs_reindex:
        print("Index is stale or missing — reindexing documents...")
        retriever.ingest_documents(force_reindex=True, build_embeddings=True)
    elif not retriever.chunks:
        retriever.ingest_documents(force_reindex=True, build_embeddings=True)

    return retriever


def main():
    parser = argparse.ArgumentParser(
        description="Query the causal knowledge graph with natural language."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # query subcommand
    qp = subparsers.add_parser("query", help="Ask a causal question")
    qp.add_argument("question", help="Natural language question")
    qp.add_argument("--json", action="store_true", help="Output raw JSON")
    qp.add_argument(
        "--no-rag", action="store_true", help="Skip RAG (KG paths only)"
    )
    qp.add_argument(
        "--max-depth", type=int, default=5, help="Max path depth (default: 5)"
    )

    # audit subcommand
    ap = subparsers.add_parser("audit", help="Audit mechanism completeness")
    ap.add_argument("entity_id", help="Entity ID to audit (e.g. graphite)")
    ap.add_argument("--json", action="store_true", help="Output raw JSON")

    # entities subcommand
    ep = subparsers.add_parser("entities", help="List KG entities")
    ep.add_argument("--type", help="Filter by entity type (e.g. commodity, policy)")

    args = parser.parse_args()

    from src.minerals.knowledge_graph import build_critical_minerals_kg
    from src.minerals.kg_query import KGQueryEngine

    kg = build_critical_minerals_kg()

    # Set up retriever (optional — graceful if no docs indexed)
    retriever = None
    if not getattr(args, "no_rag", False):
        try:
            retriever = _get_retriever_with_auto_index()
        except Exception as e:
            print(f"Note: RAG retriever unavailable ({e}). Using KG paths only.\n")

    engine = KGQueryEngine(
        kg=kg,
        retriever=retriever,
        max_path_depth=getattr(args, "max_depth", 5),
    )

    if args.command == "query":
        result = engine.query(args.question)
        if getattr(args, "json", False):
            print(json.dumps(result, indent=2, default=str))
        else:
            _print_query(result)

    elif args.command == "audit":
        result = engine.audit_mechanisms(args.entity_id)
        if getattr(args, "json", False):
            print(json.dumps(result, indent=2, default=str))
        else:
            _print_audit(result)

    elif args.command == "entities":
        _print_entities(kg, getattr(args, "type", None))


def _print_query(r: dict) -> None:
    print("=" * 70)
    print(f"Question:  {r['raw_question']}")
    print(f"Entities:  {r['entity_ids_found']}")
    print(f"Confidence: {r['confidence']:.2f}")
    print("=" * 70)
    print(f"\n{r['answer']}")

    if r["kg_paths"]:
        print(f"\n--- Causal Paths ({len(r['kg_paths'])}) ---")
        for i, p in enumerate(r["kg_paths"][:5]):
            print(f"\nPath {i + 1} (confidence: {p['confidence']:.2f}):")
            print(f"  {p['linearized']}")

    if r["supporting_docs"]:
        print(f"\n--- Supporting Documents ({len(r['supporting_docs'])}) ---")
        for d in r["supporting_docs"][:3]:
            src = d["metadata"].get(
                "source_file", d["metadata"].get("source", "unknown")
            )
            print(f"  [{src}] {d['text'][:120]}...")


def _print_audit(r: dict) -> None:
    print("=" * 70)
    print(f"Mechanism Audit: {r['entity_id']}")
    print(f"Coverage Score:  {r['coverage_score']:.2f}")
    print("=" * 70)

    print(f"\nExisting mechanisms ({len(r['existing_mechanisms'])}):")
    for m in r["existing_mechanisms"]:
        print(f"  {m['source']} -> {m['target']}: {m['mechanism']}")

    if r["suggested_new_mechanisms"]:
        print(f"\nSuggested new mechanisms ({len(r['suggested_new_mechanisms'])}):")
        for m in r["suggested_new_mechanisms"]:
            print(f"  {m.get('source', '?')} -> {m.get('target', '?')}: {m['mechanism']}")
            if m.get("evidence_source"):
                print(f"    Evidence: {m['evidence_source']}")
    else:
        print("\nNo new mechanisms suggested.")

    print(f"\nSummary: {r['audit_summary']}")


def _print_entities(kg, type_filter: str = None) -> None:
    from src.minerals.knowledge_graph import EntityType

    for entity in sorted(
        kg._entities.values(), key=lambda e: (e.entity_type.value, e.id)
    ):
        if type_filter and entity.entity_type.value != type_filter:
            continue
        print(f"  [{entity.entity_type.value:20s}] {entity.id}")


if __name__ == "__main__":
    main()
