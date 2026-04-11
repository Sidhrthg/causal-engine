"""
Optional HippoRAG-based retriever for graph-based, multi-hop retrieval.

Uses the same interface as SimpleRAGRetriever (retrieve returns list of
{text, metadata}) so it can be swapped in for Just RAG and Validate with RAG.

Requires: hipporag package (python3 -m pip install hipporag or pip install -e ".[hipporag]") and OPENAI_API_KEY or vLLM for indexing.
Set USE_HIPPORAG=1 and build the index (script or Gradio) to use this backend.
"""

from __future__ import annotations

import json
import multiprocessing
import os
import platform
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

load_dotenv()

# HippoRAG uses multiprocessing.Manager() at import time, which crashes on
# macOS with Python 3.12+ (default start method is "spawn"). Force "fork".
if platform.system() == "Darwin":
    try:
        multiprocessing.set_start_method("fork", force=True)
    except RuntimeError:
        pass  # already set

try:
    from hipporag import HippoRAG
    HIPPORAG_AVAILABLE = True
except (ImportError, RuntimeError):
    # Package missing or failed to load native deps — callers use SimpleRAGRetriever instead.
    HippoRAG = None  # type: ignore
    HIPPORAG_AVAILABLE = False


# Where HippoRAG stores its graph + vectors; also holds our doc_meta.json (passage ↔ file map).
DEFAULT_HIPPORAG_SAVE_DIR = "data/documents/hipporag_index"


def _collect_docs_from_dir(documents_dir: Path) -> tuple[List[str], List[Dict[str, Any]]]:
    """
    Walk ``documents_dir`` for ``.txt`` / ``.md``, split into paragraphs, return parallel lists.

    Skips paragraphs under 30 chars; keeps at most 50 paragraphs per file to cap index size.
    Tries several encodings per file, then falls back to UTF-8 with replacement.
    """
    docs: List[str] = []
    meta: List[Dict[str, Any]] = []
    text_files = list(documents_dir.rglob("*.txt")) + list(documents_dir.rglob("*.md"))
    for file_path in sorted(text_files):
        for enc in ("utf-8", "latin-1", "cp1252"):
            try:
                with open(file_path, "r", encoding=enc) as f:
                    content = f.read()
                break
            except (UnicodeDecodeError, OSError):
                continue
        else:
            try:
                with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                    content = f.read()
            except OSError:
                continue
        try:
            rel = file_path.relative_to(documents_dir)
        except ValueError:
            rel = file_path
        source_file = str(rel)
        # Chunk by paragraph (HippoRAG prefers coherent passages)
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", content) if p.strip()]
        if not paragraphs:
            paragraphs = [content[:2000]] if content.strip() else []
        for i, para in enumerate(paragraphs[:50]):  # cap per file to avoid huge index
            if len(para) < 30:
                continue
            docs.append(para)
            meta.append({"source_file": source_file, "chunk_index": i})
    return docs, meta


class HippoRAGRetriever:
    """
    Graph-based retriever using HippoRAG (optional).

    Same contract as ``SimpleRAGRetriever``: ``retrieve(query, top_k)`` returns
    ``[{"text": str, "metadata": dict}, ...]``. Use ``index()`` once to build
    the graph from ``documents_dir``; query-time needs the same LLM/embed setup
    as indexing (e.g. ``OPENAI_API_KEY`` or ``VLLM_BASE_URL``).

    ``llm_model_name`` / ``embedding_model_name`` should match what was used when
    the index under ``save_dir`` was built, or retrieval can misbehave.
    """

    def __init__(
        self,
        documents_dir: str = "data/documents",
        save_dir: Optional[str] = None,
        llm_model_name: str = "gpt-4o-mini",
        embedding_model_name: str = "text-embedding-3-large",
        llm_base_url: Optional[str] = None,
        embedding_base_url: Optional[str] = None,
        openai_api_key: Optional[str] = None,
    ):
        if not HIPPORAG_AVAILABLE:
            raise RuntimeError("hipporag not installed. Run: python3 -m pip install hipporag or pip install -e \".[hipporag]\"")
        self.documents_dir = Path(documents_dir)
        self.save_dir = Path(save_dir or DEFAULT_HIPPORAG_SAVE_DIR)
        self._docs: List[str] = []
        self._meta: List[Dict[str, Any]] = []
        self._hipporag: Optional[Any] = None
        self._llm_model_name = llm_model_name
        self._embedding_model_name = embedding_model_name
        self._llm_base_url = llm_base_url
        self._embedding_base_url = embedding_base_url
        key = openai_api_key or os.getenv("OPENAI_API_KEY")
        # HippoRAG reads the key from the environment internally.
        if key and not os.getenv("OPENAI_API_KEY"):
            os.environ["OPENAI_API_KEY"] = key
        self._load_or_init()

    def _read_doc_meta(self) -> None:
        """Load ``doc_meta.json`` into ``_docs`` / ``_meta`` if present; no-op if missing."""
        mapping_path = self.save_dir / "doc_meta.json"
        if not mapping_path.exists():
            return
        with open(mapping_path, "r") as f:
            data = json.load(f)
        self._docs = data.get("docs", [])
        self._meta = data.get("meta", [])

    def _load_or_init(self) -> None:
        """Reload passage list from disk, then construct the HippoRAG client for this save_dir."""
        self._read_doc_meta()
        self._hipporag = HippoRAG(
            save_dir=str(self.save_dir),
            llm_model_name=self._llm_model_name,
            embedding_model_name=self._embedding_model_name,
            llm_base_url=self._llm_base_url or os.getenv("VLLM_BASE_URL"),
            embedding_base_url=self._embedding_base_url,
        )

    @property
    def chunks(self) -> List[Any]:
        """
        Same shape as ``SimpleRAGRetriever.chunks``: lightweight objects with
        ``text`` and ``metadata`` (used by eval and question generation).
        """
        class PseudoChunk:
            def __init__(self, text: str, metadata: dict):
                self.text = text
                self.metadata = metadata
        return [PseudoChunk(d, m) for d, m in zip(self._docs, self._meta)]

    def _find_doc_metadata(self, doc_text: str) -> Dict[str, Any]:
        """
        Map a HippoRAG-returned passage back to ``source_file`` / ``chunk_index``.

        HippoRAG may return text that exactly matches a stored chunk or a longer
        span; we try exact match first, then whether any indexed chunk is a substring.
        """
        # Exact match first
        for i, d in enumerate(self._docs):
            if d == doc_text:
                return dict(self._meta[i])
        # Substring match: check if any indexed chunk is contained in the retrieved passage
        for i, d in enumerate(self._docs):
            if d in doc_text:
                return dict(self._meta[i])
        return {"source_file": "?"}

    def index(self) -> str:
        """
        Scan ``documents_dir``, write ``doc_meta.json``, and run HippoRAG's ``index``.

        Returns a short human-readable status (including emoji prefixes used by the UI).
        """
        if not self.documents_dir.exists():
            return f"❌ Documents dir not found: {self.documents_dir}"
        docs, meta = _collect_docs_from_dir(self.documents_dir)
        if not docs:
            return "⚠️ No .txt/.md documents found to index."
        self._docs = docs
        self._meta = meta
        self.save_dir.mkdir(parents=True, exist_ok=True)
        with open(self.save_dir / "doc_meta.json", "w") as f:
            json.dump({"docs": docs, "meta": meta}, f, indent=2)
        try:
            self._hipporag.index(docs=docs)
        except Exception as e:
            return f"❌ HippoRAG index failed: {e}"
        return f"✅ HippoRAG indexed {len(docs)} passages. Save dir: {self.save_dir}"

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[Dict] = None,  # unused; kept for SimpleRAGRetriever API parity
    ) -> List[Dict[str, Any]]:
        """
        Run graph-based retrieval for ``query``.

        On failure, returns a single-item list whose ``text`` is an error message
        (so callers don't crash). Caps HippoRAG at 20 hits per query internally.
        """
        if not self._docs:
            self._read_doc_meta()
            if not self._docs:
                return []
        try:
            retrieval_results = self._hipporag.retrieve(
                queries=[query],
                num_to_retrieve=min(top_k, 20),
            )
        except Exception as e:
            return [{"text": f"[HippoRAG retrieve error: {e}]", "metadata": {"source_file": "?"}}]
        # HippoRAG returns List[QuerySolution]; normalize to list of {text, metadata}
        out: List[Dict[str, Any]] = []
        if isinstance(retrieval_results, list) and len(retrieval_results) > 0:
            first_result = retrieval_results[0]
            # HippoRAG >=2.0 returns QuerySolution objects with .docs and .doc_scores
            if hasattr(first_result, "docs") and isinstance(first_result.docs, list):
                scores = getattr(first_result, "doc_scores", None)
                for idx_r, doc_text in enumerate(first_result.docs[:top_k]):
                    score = float(scores[idx_r]) if scores is not None and idx_r < len(scores) else 0.0
                    # Try to find source metadata by matching against indexed docs
                    meta = self._find_doc_metadata(doc_text)
                    meta["score"] = score
                    out.append({"text": doc_text, "metadata": meta})
            elif isinstance(first_result, list):
                # Legacy list-of-strings format
                for item in first_result[:top_k]:
                    if isinstance(item, str):
                        meta = self._find_doc_metadata(item)
                        out.append({"text": item, "metadata": meta})
                    elif isinstance(item, dict):
                        text = item.get("text", item.get("passage", str(item)))
                        out.append({"text": text, "metadata": item.get("metadata", {"source_file": "?"})})
        return out


def hipporag_available() -> bool:
    """True if the ``hipporag`` package imported successfully."""
    return HIPPORAG_AVAILABLE


def get_retriever(
    use_hipporag: bool = True,
    documents_dir: Optional[str] = None,
    index_path: Optional[str] = None,
    api_key: Optional[str] = None,
):
    """
    Pick a retriever: HippoRAG when the package is installed and
    ``<documents_dir>/hipporag_index/doc_meta.json`` exists; otherwise
    ``SimpleRAGRetriever``.

    Args:
        use_hipporag: If False, always use ``SimpleRAGRetriever``.
        documents_dir: Root folder for corpus files (default ``data/documents``).
        index_path: Path to ``index.json`` for the simple retriever.
        api_key: Optional OpenAI key for the simple retriever's embeddings.

    Returns:
        An object with ``retrieve(query, top_k=...)`` and ``chunks``.
    """
    if use_hipporag and HIPPORAG_AVAILABLE:
        docs_dir = documents_dir or "data/documents"
        doc_meta = Path(docs_dir) / "hipporag_index" / "doc_meta.json"
        if doc_meta.exists():
            return HippoRAGRetriever(
                documents_dir=docs_dir,
                save_dir=str(doc_meta.parent),
            )
    from src.minerals.rag_retrieval import SimpleRAGRetriever
    return SimpleRAGRetriever(
        documents_dir=documents_dir or "data/documents",
        index_path=index_path or "data/documents/index.json",
        api_key=api_key,
    )
