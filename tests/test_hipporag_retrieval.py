"""Tests for HippoRAG retrieval (hipporag_retrieval module).

Run with: pytest tests/test_hipporag_retrieval.py -v
With hipporag installed: pytest tests/test_hipporag_retrieval.py -v
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.minerals.hipporag_retrieval import (
    HIPPORAG_AVAILABLE,
    _collect_docs_from_dir,
    get_retriever,
    hipporag_available,
)


def test_hipporag_available_returns_bool():
    """hipporag_available() is a boolean (True if package installed)."""
    assert isinstance(hipporag_available(), bool)


def test_collect_docs_from_dir_empty_dir(tmp_path: Path):
    """Empty dir or no .txt/.md yields empty docs and meta."""
    docs, meta = _collect_docs_from_dir(tmp_path)
    assert docs == []
    assert meta == []


def test_collect_docs_from_dir_single_file(tmp_path: Path):
    """Single .txt file is chunked by paragraph; short paragraphs skipped."""
    (tmp_path / "foo.txt").write_text(
        "First paragraph here with enough text to pass the length filter.\n\n"
        "Second paragraph also long enough to be included in the index.\n\n"
        "Short.\n\n"
        "Another good paragraph with sufficient content for indexing.",
        encoding="utf-8",
    )
    docs, meta = _collect_docs_from_dir(tmp_path)
    assert len(docs) >= 2
    assert len(meta) == len(docs)
    assert all("source_file" in m for m in meta)
    assert all(m["source_file"] == "foo.txt" for m in meta)
    # "Short." should be skipped (len < 30)
    assert not any(d.strip() == "Short." for d in docs)


def test_collect_docs_from_dir_respects_cap_per_file(tmp_path: Path):
    """At most 50 paragraphs per file are indexed."""
    many_paras = "\n\n".join([f"Paragraph number {i} with enough text to be indexed." for i in range(60)])
    (tmp_path / "many.txt").write_text(many_paras, encoding="utf-8")
    docs, meta = _collect_docs_from_dir(tmp_path)
    assert len(docs) <= 50
    assert len(meta) == len(docs)


def test_get_retriever_use_hipporag_false():
    """get_retriever(use_hipporag=False) returns SimpleRAGRetriever (no hipporag needed)."""
    retriever = get_retriever(use_hipporag=False, documents_dir="data/documents")
    assert hasattr(retriever, "retrieve")
    assert hasattr(retriever, "chunks")
    assert callable(retriever.retrieve)


def test_hipporag_retriever_raises_when_not_installed():
    """HippoRAGRetriever constructor raises if hipporag is not installed."""
    from src.minerals.hipporag_retrieval import HippoRAGRetriever

    if HIPPORAG_AVAILABLE:
        pytest.skip("hipporag is installed; cannot test RuntimeError path")
    with pytest.raises(RuntimeError, match="hipporag not installed"):
        HippoRAGRetriever(documents_dir="data/documents", save_dir="/tmp/never")


def test_get_retriever_fallback_when_no_index(tmp_path: Path):
    """When use_hipporag=True but no hipporag index exists, get_retriever falls back to SimpleRAGRetriever."""
    retriever = get_retriever(
        use_hipporag=True,
        documents_dir=str(tmp_path),
    )
    assert hasattr(retriever, "retrieve")
    assert hasattr(retriever, "chunks")
    assert callable(retriever.retrieve)
