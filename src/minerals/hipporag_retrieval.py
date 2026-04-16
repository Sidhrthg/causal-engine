"""
Optional HippoRAG retriever; same ``retrieve`` shape as SimpleRAGRetriever.

Needs ``hipporag``, OPENAI_API_KEY or vLLM, and a built index under
``data/documents/hipporag_index/`` (see ``docs/HIPPORAG.md``). That directory is
gitignored (large parquets); build locally after clone.
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
    HippoRAG = None  # type: ignore
    HIPPORAG_AVAILABLE = False


DEFAULT_HIPPORAG_SAVE_DIR = "data/documents/hipporag_index"


def _collect_docs_from_dir(documents_dir: Path) -> tuple[List[str], List[Dict[str, Any]]]:
    """Load .txt/.md under documents_dir; chunk by paragraph (max 50/file, min 30 chars)."""
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
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", content) if p.strip()]
        if not paragraphs:
            paragraphs = [content[:2000]] if content.strip() else []
        for i, para in enumerate(paragraphs[:50]):
            if len(para) < 30:
                continue
            docs.append(para)
            meta.append({"source_file": source_file, "chunk_index": i})
    return docs, meta


class HippoRAGRetriever:
    """HippoRAG backend; ``retrieve`` matches SimpleRAGRetriever. Match LLM/embed names to the built index."""

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
        if key and not os.getenv("OPENAI_API_KEY"):
            os.environ["OPENAI_API_KEY"] = key
        self._load_or_init()

    def _read_doc_meta(self) -> None:
        mapping_path = self.save_dir / "doc_meta.json"
        if not mapping_path.exists():
            return
        with open(mapping_path, "r") as f:
            data = json.load(f)
        self._docs = data.get("docs", [])
        self._meta = data.get("meta", [])

    def _load_or_init(self) -> None:
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
        """Chunk-like objects with ``text`` and ``metadata`` (RAG eval)."""
        class PseudoChunk:
            def __init__(self, text: str, metadata: dict):
                self.text = text
                self.metadata = metadata
        return [PseudoChunk(d, m) for d, m in zip(self._docs, self._meta)]

    def _find_doc_metadata(self, doc_text: str) -> Dict[str, Any]:
        """Match returned text to indexed chunk for source_file metadata."""
        for i, d in enumerate(self._docs):
            if d == doc_text:
                return dict(self._meta[i])
        for i, d in enumerate(self._docs):
            if d in doc_text:
                return dict(self._meta[i])
        return {"source_file": "?"}

    def index(self) -> str:
        """Build index from ``documents_dir``; returns status string for UI."""
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
        filters: Optional[Dict] = None,
    ) -> List[Dict[str, Any]]:
        """Graph retrieval; ``filters`` ignored (API parity with SimpleRAGRetriever)."""
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
        out: List[Dict[str, Any]] = []
        if isinstance(retrieval_results, list) and len(retrieval_results) > 0:
            first_result = retrieval_results[0]
            if hasattr(first_result, "docs") and isinstance(first_result.docs, list):
                scores = getattr(first_result, "doc_scores", None)
                for idx_r, doc_text in enumerate(first_result.docs[:top_k]):
                    score = float(scores[idx_r]) if scores is not None and idx_r < len(scores) else 0.0
                    meta = self._find_doc_metadata(doc_text)
                    meta["score"] = score
                    out.append({"text": doc_text, "metadata": meta})
            elif isinstance(first_result, list):
                for item in first_result[:top_k]:
                    if isinstance(item, str):
                        meta = self._find_doc_metadata(item)
                        out.append({"text": item, "metadata": meta})
                    elif isinstance(item, dict):
                        text = item.get("text", item.get("passage", str(item)))
                        out.append({"text": text, "metadata": item.get("metadata", {"source_file": "?"})})
        return out


def hipporag_available() -> bool:
    return HIPPORAG_AVAILABLE


def get_retriever(
    use_hipporag: bool = True,
    documents_dir: Optional[str] = None,
    index_path: Optional[str] = None,
    api_key: Optional[str] = None,
):
    """HippoRAG if installed and ``hipporag_index/doc_meta.json`` exists; else SimpleRAGRetriever."""
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
