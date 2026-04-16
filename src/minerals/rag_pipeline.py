"""
Standalone RAG pipeline for the minerals corpus.

Usage (independent of causal engine)::

    from src.minerals.rag_pipeline import RAGPipeline
    rag = RAGPipeline()           # auto-selects best available backend
    answer = rag.ask("What happened to graphite prices in 2008?")
    print(answer["answer"])
    for src in answer["sources"]:
        print(src["source"], src["similarity"])

The backend is chosen automatically: HippoRAG (if installed + indexed)
> IndustrialRAG (ChromaDB, if installed) > SimpleRAGRetriever (always available).
Override with ``backend="simple"`` / ``"industrial"`` / ``"hipporag"``.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.llm.chat import chat_completion, is_chat_available
from src.llm.memory import EpisodicMemory
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

# Default prompt injected before retrieved context.
_DEFAULT_SYSTEM = (
    "You are a minerals supply chain analyst. "
    "Answer the question using ONLY the provided source documents. "
    "Cite sources by their [number] when using information from them. "
    "If the documents do not contain enough information, say so explicitly."
)


def _normalize_chunk(raw: Any) -> Dict:
    """Normalize retriever output to ``{text, metadata, similarity}``."""
    if isinstance(raw, dict):
        meta = raw.get("metadata", {})
        if not isinstance(meta, dict):
            meta = {"source_file": str(meta)}
        sim = (
            raw.get("similarity")
            or raw.get("hybrid_score")
            or raw.get("score")
            or (meta.get("score") if isinstance(meta, dict) else None)
            or 0.0
        )
        return {
            "text": str(raw.get("text", raw.get("passage", ""))),
            "metadata": meta,
            "similarity": float(sim),
        }
    text = getattr(raw, "text", str(raw))
    meta = getattr(raw, "metadata", {})
    if not isinstance(meta, dict):
        meta = {"source_file": str(meta)}
    return {"text": text, "metadata": meta, "similarity": 0.0}


def _build_context(chunks: List[Dict], max_chars: int = 12_000) -> str:
    """Numbered context block for the LLM, capped at ``max_chars``."""
    lines: List[str] = []
    total = 0
    for i, chunk in enumerate(chunks, 1):
        source = chunk["metadata"].get("source_file", chunk["metadata"].get("source", "?"))
        sim = chunk.get("similarity", 0.0)
        header = f"[{i}] {source}" + (f" (score={sim:.3f})" if sim else "")
        body = chunk["text"].strip()
        entry = f"{header}\n{body}\n"
        if total + len(entry) > max_chars:
            remaining = max_chars - total - len(header) - 10
            if remaining > 100:
                entry = f"{header}\n{body[:remaining]}...\n"
                lines.append(entry)
            break
        lines.append(entry)
        total += len(entry)
    return "\n".join(lines)


class RAGPipeline:
    """
    Unified RAG pipeline — retrieve from any backend, generate with any LLM.

    Backends (in auto-selection order):
    1. ``hipporag``  — graph-based, multi-hop retrieval (needs hipporag + index)
    2. ``industrial`` — ChromaDB + sentence-transformers (needs chromadb)
    3. ``simple``    — keyword + embedding search, always available

    Args:
        backend: ``"auto"`` | ``"hipporag"`` | ``"industrial"`` | ``"simple"``
        documents_dir: Root directory for documents (all backends).
        chroma_dir: ChromaDB persistence dir (industrial backend).
        collection_name: ChromaDB collection name (industrial backend).
        embedding_model: Sentence-transformers model (industrial backend).
        memory_dir: Directory for episodic memory. Pass ``None`` to disable.
        openai_model: HippoRAG LLM name (default ``OPENAI_MODEL`` or ``gpt-4o-mini``).
    """

    def __init__(
        self,
        backend: str = "auto",
        documents_dir: str = "data/documents",
        chroma_dir: str = "data/chroma_db",
        collection_name: str = "minerals_corpus",
        embedding_model: str = "all-mpnet-base-v2",
        memory_dir: Optional[str] = "data/memory",
        openai_model: Optional[str] = None,
    ):
        self.documents_dir = documents_dir
        self._retriever = None
        self.backend_name: str = "none"
        self._openai_model = openai_model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self._init_retriever(backend, documents_dir, chroma_dir, collection_name, embedding_model)
        self.memory: Optional[EpisodicMemory] = (
            EpisodicMemory(memory_dir=memory_dir) if memory_dir else None
        )

    def _init_retriever(
        self,
        backend: str,
        documents_dir: str,
        chroma_dir: str,
        collection_name: str,
        embedding_model: str,
    ) -> None:
        order = (
            ["raganything", "hipporag", "industrial", "simple"] if backend == "auto" else [backend]
        )
        for name in order:
            try:
                retriever = self._load_backend(
                    name, documents_dir, chroma_dir, collection_name, embedding_model, self._openai_model
                )
                self._retriever = retriever
                self.backend_name = name
                logger.info(f"RAGPipeline: using backend '{name}'")
                return
            except Exception as exc:
                logger.debug(f"RAGPipeline: backend '{name}' not available: {exc}")
        tried = ", ".join(order)
        raise RuntimeError(
            f"No RAG backend available (tried: {tried}). "
            "For 'industrial' install chromadb; for 'hipporag' install hipporag and build the index. "
            "Use backend='simple' or backend='auto' to fall back automatically."
        )

    def _load_backend(
        self,
        name: str,
        documents_dir: str,
        chroma_dir: str,
        collection_name: str,
        embedding_model: str,
        openai_model: str = "gpt-4o-mini",
    ):
        if name == "raganything":
            from src.minerals.raganything_retrieval import raganything_available, RAGAnythingRetriever

            if not raganything_available():
                raise ImportError("raganything not installed")
            from src.minerals.raganything_retrieval import DEFAULT_WORKING_DIR
            graphml = Path(documents_dir) / "raganything_index" / "graph_chunk_entity_relation.graphml"
            if not graphml.exists():
                raise FileNotFoundError(f"RAGAnything index not found at {graphml}")
            return RAGAnythingRetriever(
                working_dir=str(Path(documents_dir) / "raganything_index"),
                llm_model_name=openai_model,
            )

        if name == "hipporag":
            from src.minerals.hipporag_retrieval import hipporag_available, HippoRAGRetriever

            if not hipporag_available():
                raise ImportError("hipporag not installed")
            index_path = Path(documents_dir) / "hipporag_index" / "doc_meta.json"
            if not index_path.exists():
                raise FileNotFoundError(f"HippoRAG index not found at {index_path}")
            return HippoRAGRetriever(
                documents_dir=documents_dir,
                save_dir=str(Path(documents_dir) / "hipporag_index"),
                llm_model_name=openai_model,
            )

        if name == "industrial":
            from src.minerals.rag_industrial import IndustrialRAG

            rag = IndustrialRAG(
                collection_name=collection_name,
                persist_directory=chroma_dir,
                embedding_model=embedding_model,
            )
            if rag.collection.count() == 0:
                raise RuntimeError("IndustrialRAG collection is empty; index documents first")
            return rag

        if name == "simple":
            from src.minerals.rag_retrieval import SimpleRAGRetriever

            return SimpleRAGRetriever(
                documents_dir=documents_dir,
                index_path=f"{documents_dir}/index.json",
            )

        raise ValueError(f"Unknown backend: {name!r}. Choose auto/hipporag/industrial/simple.")

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        filters: Optional[Dict] = None,
        use_hybrid: bool = True,
    ) -> List[Dict]:
        """
        Retrieve the top-k most relevant document chunks for *query*.

        Returns a list of dicts::

            {"text": str, "metadata": dict, "similarity": float}

        When episodic memory is enabled, chunks that appeared in past high-quality
        answers receive a small score boost and are re-ranked accordingly.

        Args:
            query: Natural language query.
            top_k: Number of chunks to return.
            filters: Backend-specific metadata filters (e.g. ``{"mineral": "graphite"}``).
            use_hybrid: Use hybrid semantic+keyword scoring when available (industrial backend).
        """
        if self._retriever is None:
            return []

        fetch_k = top_k * 2 if self.memory else top_k
        try:
            from src.minerals.rag_industrial import IndustrialRAG

            if isinstance(self._retriever, IndustrialRAG) and use_hybrid:
                raw = self._retriever.hybrid_search(query, top_k=fetch_k, filters=filters)
            else:
                raw = self._retriever.retrieve(query, top_k=fetch_k, filters=filters)
        except Exception as exc:
            logger.error(f"Retrieval failed: {exc}")
            return []

        chunks = [_normalize_chunk(r) for r in raw]

        if self.memory:
            boosts = self.memory.boosted_chunks()
            if boosts:
                for chunk in chunks:
                    preview = chunk["text"][:200]
                    chunk_id = hashlib.md5(preview.encode()).hexdigest()[:16]
                    if chunk_id in boosts:
                        chunk["similarity"] += boosts[chunk_id] * 0.15
                        chunk["boosted"] = True
                chunks.sort(key=lambda c: c["similarity"], reverse=True)

        return chunks[:top_k]

    def ask(
        self,
        question: str,
        top_k: int = 8,
        filters: Optional[Dict] = None,
        system_prompt: Optional[str] = None,
        max_context_chars: int = 12_000,
        max_tokens: int = 2_048,
        use_memory: bool = True,
    ) -> Dict[str, Any]:
        """
        Full RAG: retrieve relevant chunks, then generate an answer grounded in them.

        When episodic memory is enabled (default), the prompt is augmented with
        similar past Q&A pairs as few-shot examples, and the result is stored in
        memory for future queries. Call ``feedback()`` with the returned
        ``episode_id`` to record whether the answer was correct.

        Returns::

            {
                "answer": str,          # LLM response with inline citations
                "sources": [            # chunks used, ordered by relevance
                    {"source": str, "text_preview": str, "similarity": float},
                    ...
                ],
                "question": str,
                "episode_id": str | None,   # pass to feedback() to rate this answer
                "n_retrieved": int,
                "backend": str,
                "llm_available": bool,
            }

        Args:
            question: Natural language question.
            top_k: Number of document chunks to retrieve.
            filters: Optional metadata filters passed to the retriever.
            system_prompt: Override default system prompt.
            max_context_chars: Hard cap on total context characters passed to LLM.
            max_tokens: Max tokens for LLM response.
            use_memory: Inject few-shot context from past episodes (default True).
        """
        chunks = self.retrieve(question, top_k=top_k, filters=filters)
        llm_ok = is_chat_available()

        sources = [
            {
                "source": c["metadata"].get("source_file", c["metadata"].get("source", "?")),
                "text_preview": c["text"][:200] + ("..." if len(c["text"]) > 200 else ""),
                "similarity": c["similarity"],
            }
            for c in chunks
        ]

        if not chunks:
            return {
                "answer": "No relevant documents found in the corpus.",
                "sources": [],
                "question": question,
                "episode_id": None,
                "n_retrieved": 0,
                "backend": self.backend_name,
                "llm_available": llm_ok,
            }

        few_shot = ""
        if use_memory and self.memory:
            few_shot = self.memory.build_few_shot_context(question)

        doc_budget = max_context_chars - len(few_shot)
        context = _build_context(chunks, max_chars=max(doc_budget, 2_000))
        sys_msg = system_prompt or _DEFAULT_SYSTEM

        user_msg = (
            f"{few_shot}"
            f"Source documents:\n\n{context}\n\n"
            f"---\n\nQuestion: {question}"
        )

        if llm_ok:
            try:
                answer = chat_completion(
                    messages=[
                        {"role": "user", "content": f"{sys_msg}\n\n{user_msg}"},
                    ],
                    max_tokens=max_tokens,
                )
            except Exception as exc:
                logger.error(f"LLM generation failed: {exc}")
                answer = f"[LLM error: {exc}]\n\nRetrieved context:\n{context}"
        else:
            answer = (
                "[LLM not configured — set ANTHROPIC_API_KEY or LLM_BACKEND]\n\n"
                f"Retrieved context:\n{context}"
            )

        episode_id: Optional[str] = None
        if self.memory:
            episode_id = self.memory.store(
                query=question,
                answer=answer,
                sources=sources,
                backend=self.backend_name,
            )

        return {
            "answer": answer,
            "sources": sources,
            "question": question,
            "episode_id": episode_id,
            "n_retrieved": len(chunks),
            "backend": self.backend_name,
            "llm_available": llm_ok,
        }

    def feedback(
        self,
        episode_id: str,
        rating: int,
        correction: Optional[str] = None,
    ) -> bool:
        """
        Record user feedback for a past answer.

        Args:
            episode_id: The ``episode_id`` from a previous ``ask()`` call.
            rating: 1 = accurate, -1 = wrong/hallucinated, 0 = neutral.
            correction: Correct answer when rating=-1 (stored for analysis,
                excluded from future few-shot injection).

        Returns:
            True if the episode was found and updated.
        """
        if self.memory is None:
            logger.warning("Memory not enabled; feedback ignored.")
            return False
        return self.memory.rate(episode_id, rating, correction)

    def index_documents(self, force_reindex: bool = False) -> str:
        """
        (Re-)index the document corpus for the active backend.

        Returns a status message. Only applies to ``simple`` and ``industrial``
        backends; HippoRAG indexing must be done via ``scripts/index_hipporag.py``.
        """
        from src.minerals.rag_retrieval import SimpleRAGRetriever
        from src.minerals.rag_industrial import IndustrialRAG

        if isinstance(self._retriever, SimpleRAGRetriever):
            self._retriever.ingest_documents(force_reindex=force_reindex)
            n = len(self._retriever.chunks)
            return f"SimpleRAGRetriever: indexed {n} chunks"

        if isinstance(self._retriever, IndustrialRAG):
            n = self._retriever.index_corpus(
                doc_directory=self.documents_dir,
                force_reindex=force_reindex,
            )
            return f"IndustrialRAG: indexed {n} chunks into ChromaDB"

        return "Indexing not supported for this backend (HippoRAG: use scripts/index_hipporag.py)"

    def stats(self) -> Dict[str, Any]:
        """Corpus stats and optional memory summary."""
        from src.minerals.rag_industrial import IndustrialRAG

        if isinstance(self._retriever, IndustrialRAG):
            base = self._retriever.get_statistics()
        else:
            chunks = getattr(self._retriever, "chunks", [])
            base = {"backend": self.backend_name, "total_chunks": len(chunks)}

        base["memory"] = self.memory.summary() if self.memory else None
        return base
