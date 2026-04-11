"""
Episodic memory for the RAG pipeline.

Stores past Q&A episodes, retrieves similar ones for few-shot prompt injection,
tracks knowledge gaps (queries the corpus couldn't answer), and maintains a
map of validated chunk boost scores derived from high-quality episodes.

Storage layout (all under memory_dir):
  episodes.json      — list of Episode dicts
  embeddings.npy     — parallel array of query embeddings (optional)
  knowledge_gaps.json — queries the system failed to answer well

Similarity search uses sentence-transformers when available, falls back to
Jaccard token overlap (no extra dependencies needed).

Usage::

    from src.llm.memory import EpisodicMemory

    mem = EpisodicMemory()

    # Store an episode after calling rag.ask()
    eid = mem.store(query="What drove graphite prices in 2008?",
                    answer="Prices rose due to ...",
                    sources=[...], faithfulness=0.85, relevance=0.9)

    # Rate it (from user feedback)
    mem.rate(eid, rating=1)

    # Build few-shot context for a new query
    ctx = mem.build_few_shot_context("Why did graphite get expensive in 2008?")
    # → "Relevant past Q&A: ..."

    # See what the system doesn't know
    gaps = mem.knowledge_gaps()

    # Stats
    print(mem.summary())
"""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

DEFAULT_MEMORY_DIR = "data/memory"


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class Episode:
    """One Q&A interaction stored in episodic memory."""
    episode_id: str
    query: str
    answer: str
    sources: List[Dict]          # [{source, text_preview, similarity}, ...]
    faithfulness: float = -1.0   # 0-1 from LLM judge; -1 = not graded
    relevance: float = -1.0      # 0-1 from LLM judge; -1 = not graded
    user_rating: int = 0         # -1 = bad, 0 = neutral, 1 = good
    correction: str = ""         # user-provided correct answer when rating=-1
    timestamp: str = ""
    backend: str = ""
    tags: List[str] = field(default_factory=list)

    def quality_score(self) -> float:
        """
        Composite quality in [0, 1].
        Averages any available signals: faithfulness, relevance, user rating.
        Returns 0.5 when nothing has been graded yet.
        """
        components: List[float] = []
        if self.faithfulness >= 0:
            components.append(self.faithfulness)
        if self.relevance >= 0:
            components.append(self.relevance)
        if self.user_rating != 0:
            components.append((self.user_rating + 1) / 2.0)  # -1→0, 0→0.5, 1→1
        return float(np.mean(components)) if components else 0.5


# ---------------------------------------------------------------------------
# Memory
# ---------------------------------------------------------------------------

class EpisodicMemory:
    """
    Persistent Q&A memory for the RAG pipeline.

    Args:
        memory_dir: Directory for episodes.json, embeddings.npy, knowledge_gaps.json.
        embedding_model: Sentence-transformers model for query similarity.
            Pass ``None`` to use token-overlap fallback (no extra dependencies).
        min_quality_for_fewshot: Minimum episode quality score before an episode
            is offered as a few-shot example.
    """

    def __init__(
        self,
        memory_dir: str = DEFAULT_MEMORY_DIR,
        embedding_model: Optional[str] = "all-MiniLM-L6-v2",
        min_quality_for_fewshot: float = 0.6,
    ):
        self._dir = Path(memory_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._episodes_path = self._dir / "episodes.json"
        self._embeddings_path = self._dir / "embeddings.npy"
        self._gaps_path = self._dir / "knowledge_gaps.json"

        self._episodes: List[Episode] = []
        self._embeddings: Optional[np.ndarray] = None   # (N, dim)
        self._encoder = None
        self._min_quality = min_quality_for_fewshot

        self._load()
        self._init_encoder(embedding_model)

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def _init_encoder(self, model_name: Optional[str]) -> None:
        if model_name is None:
            return
        try:
            from sentence_transformers import SentenceTransformer
            self._encoder = SentenceTransformer(model_name)
            logger.debug(f"EpisodicMemory: loaded encoder '{model_name}'")
        except ImportError:
            logger.info(
                "sentence-transformers not installed; "
                "EpisodicMemory will use token-overlap similarity."
            )

    def _encode(self, text: str) -> Optional[np.ndarray]:
        if self._encoder is not None:
            return self._encoder.encode(text, show_progress_bar=False)
        return None

    def _sim(self, query: str, idx: int) -> float:
        """Cosine similarity if encoder available, else Jaccard over tokens."""
        if (
            self._encoder is not None
            and self._embeddings is not None
            and idx < len(self._embeddings)
        ):
            q_emb = self._encode(query)
            e_emb = self._embeddings[idx]
            nq, ne = np.linalg.norm(q_emb), np.linalg.norm(e_emb)
            if nq == 0 or ne == 0:
                return 0.0
            return float(np.dot(q_emb, e_emb) / (nq * ne))
        # Token-overlap fallback
        q_tok = set(query.lower().split())
        e_tok = set(self._episodes[idx].query.lower().split())
        if not q_tok or not e_tok:
            return 0.0
        return len(q_tok & e_tok) / len(q_tok | e_tok)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self) -> None:
        if self._episodes_path.exists():
            try:
                raw = json.loads(self._episodes_path.read_text())
                self._episodes = [Episode(**ep) for ep in raw]
            except Exception as exc:
                logger.warning(f"Could not load episodes: {exc}")
                self._episodes = []
        if self._embeddings_path.exists():
            try:
                self._embeddings = np.load(str(self._embeddings_path))
            except Exception:
                self._embeddings = None
        logger.debug(f"EpisodicMemory: loaded {len(self._episodes)} episodes.")

    def _save(self) -> None:
        self._episodes_path.write_text(
            json.dumps([asdict(ep) for ep in self._episodes], indent=2)
        )
        if self._embeddings is not None:
            np.save(str(self._embeddings_path), self._embeddings)

    def _append_embedding(self, emb: Optional[np.ndarray]) -> None:
        if emb is None:
            return
        emb = emb.reshape(1, -1)
        self._embeddings = (
            emb if self._embeddings is None else np.vstack([self._embeddings, emb])
        )

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def store(
        self,
        query: str,
        answer: str,
        sources: List[Dict],
        faithfulness: float = -1.0,
        relevance: float = -1.0,
        backend: str = "",
        tags: Optional[List[str]] = None,
    ) -> str:
        """
        Store a Q&A episode.

        Returns:
            episode_id — pass to ``rate()`` to record user feedback.
        """
        episode_id = str(uuid.uuid4())[:8]
        ep = Episode(
            episode_id=episode_id,
            query=query,
            answer=answer,
            sources=sources,
            faithfulness=faithfulness,
            relevance=relevance,
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
            backend=backend,
            tags=tags or [],
        )
        self._episodes.append(ep)
        self._append_embedding(self._encode(query))
        self._save()
        logger.debug(f"Stored episode {episode_id}: '{query[:60]}...'")
        return episode_id

    def rate(
        self,
        episode_id: str,
        rating: int,
        correction: Optional[str] = None,
    ) -> bool:
        """
        Record user feedback for a stored episode.

        Args:
            episode_id: Returned by ``store()``.
            rating: 1 = accurate, -1 = wrong/hallucinated, 0 = neutral.
            correction: Correct answer text when rating=-1. Stored for analysis
                and excluded from future few-shot injection.

        Returns:
            True if the episode was found and updated.
        """
        for ep in self._episodes:
            if ep.episode_id == episode_id:
                ep.user_rating = rating
                if correction:
                    ep.correction = correction
                self._save()
                logger.info(f"Episode {episode_id} rated {rating:+.2f}")
                return True
        logger.warning(f"Episode {episode_id} not found in memory.")
        return False

    # ------------------------------------------------------------------
    # Few-shot retrieval
    # ------------------------------------------------------------------

    def recall_similar(
        self,
        query: str,
        top_k: int = 3,
        min_quality: Optional[float] = None,
        exclude_negative: bool = True,
    ) -> List[Tuple[Episode, float]]:
        """
        Return the top-k past episodes most similar to *query*.

        Only returns episodes meeting the quality threshold and not explicitly
        rated as wrong by the user (unless ``exclude_negative=False``).

        Returns:
            List of (Episode, similarity_score) sorted by descending similarity.
        """
        if not self._episodes:
            return []
        threshold = min_quality if min_quality is not None else self._min_quality
        scored: List[Tuple[int, float]] = []
        for i, ep in enumerate(self._episodes):
            if exclude_negative and ep.user_rating == -1:
                continue
            if ep.quality_score() < threshold:
                continue
            scored.append((i, self._sim(query, i)))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [(self._episodes[i], sim) for i, sim in scored[:top_k]]

    def build_few_shot_context(
        self,
        query: str,
        top_k: int = 3,
        min_similarity: float = 0.15,
    ) -> str:
        """
        Build a few-shot prompt block from high-quality past episodes similar to *query*.

        Returns an empty string when no useful examples exist — callers can safely
        prepend this to any prompt without checking first.

        The block is designed to be injected *before* the retrieved documents section
        so the LLM sees "how we answered similar questions before" as additional context.
        """
        similar = self.recall_similar(query, top_k=top_k)
        useful = [(ep, sim) for ep, sim in similar if sim >= min_similarity]
        if not useful:
            return ""

        lines = [
            "## Relevant past answers (for context — prioritise the source documents below)\n"
        ]
        for ep, sim in useful:
            lines.append(f"**Past question:** {ep.query}")
            preview = ep.answer[:400] + ("..." if len(ep.answer) > 400 else "")
            lines.append(f"**Past answer:** {preview}")
            if ep.sources:
                srcs = ", ".join(
                    s.get("source", "?") if isinstance(s, dict) else str(s)
                    for s in ep.sources[:3]
                )
                lines.append(f"**Sources:** {srcs}")
            lines.append("")
        return "\n".join(lines) + "---\n\n"

    # ------------------------------------------------------------------
    # Retrieval boosting
    # ------------------------------------------------------------------

    def boosted_chunks(self, min_quality: float = 0.7) -> Dict[str, float]:
        """
        Return a mapping of chunk content hashes → boost score.

        Chunks that appeared in high-quality episodes get a positive boost
        so the retriever up-ranks them for similar future queries.
        The boost value is the maximum quality score seen for that chunk.
        """
        boosts: Dict[str, float] = {}
        for ep in self._episodes:
            q = ep.quality_score()
            if q < min_quality:
                continue
            for src in ep.sources:
                preview = src.get("text_preview", "") if isinstance(src, dict) else str(src)
                chunk_id = hashlib.md5(preview.encode()).hexdigest()[:16]
                boosts[chunk_id] = max(boosts.get(chunk_id, 0.0), q)
        return boosts

    # ------------------------------------------------------------------
    # Knowledge gaps
    # ------------------------------------------------------------------

    def mark_gap(
        self, query: str, source_file: str = "", notes: str = ""
    ) -> None:
        """Record a query the corpus could not confidently answer."""
        gaps = self._load_gaps()
        gap_id = hashlib.md5(query.encode()).hexdigest()[:8]
        gaps[gap_id] = {
            "query": query,
            "source_file": source_file,
            "notes": notes,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        self._gaps_path.write_text(json.dumps(gaps, indent=2))

    def knowledge_gaps(self) -> List[Dict]:
        """Return all recorded knowledge gaps, newest first."""
        gaps = list(self._load_gaps().values())
        gaps.sort(key=lambda g: g.get("timestamp", ""), reverse=True)
        return gaps

    def _load_gaps(self) -> Dict:
        if self._gaps_path.exists():
            try:
                return json.loads(self._gaps_path.read_text())
            except Exception:
                pass
        return {}

    # ------------------------------------------------------------------
    # Learning from eval reports
    # ------------------------------------------------------------------

    def learn_from_eval(
        self,
        retrieval_report: Dict,
        answer_report: Optional[Dict] = None,
        min_faithfulness: float = 0.7,
        min_relevance: float = 0.7,
    ) -> Dict[str, int]:
        """
        Ingest ``RAGEvaluator`` output to auto-populate memory.

        - Episodes from ``answer_report`` with high faithfulness + relevance
          are stored as high-quality few-shot examples.
        - Failed retrievals from ``retrieval_report`` are marked as knowledge gaps.

        Args:
            retrieval_report: Output of ``RAGEvaluator.evaluate_retrieval()``.
            answer_report: Output of ``RAGEvaluator.evaluate_answers()`` (optional).
            min_faithfulness / min_relevance: Thresholds for storing as example.

        Returns:
            ``{"stored": N, "gaps": M}``
        """
        stored = 0
        gaps = 0

        if answer_report:
            for result in answer_report.get("results", []):
                faith = getattr(result, "faithfulness", 0.0)
                rel = getattr(result, "relevance", 0.0)
                if faith >= min_faithfulness and rel >= min_relevance:
                    sq = getattr(result, "question", None)
                    if sq is not None:
                        self.store(
                            query=sq.question,
                            answer=result.answer,
                            sources=result.sources_used,
                            faithfulness=faith,
                            relevance=rel,
                            tags=["eval_generated"],
                        )
                        stored += 1

        for result in retrieval_report.get("results", []):
            if not getattr(result, "hit", True):
                sq = getattr(result, "question", None)
                if sq is not None:
                    self.mark_gap(
                        query=sq.question,
                        source_file=sq.source_file,
                        notes="Failed retrieval in eval loop",
                    )
                    gaps += 1

        logger.info(f"learn_from_eval: stored {stored} episodes, marked {gaps} gaps.")
        return {"stored": stored, "gaps": gaps}

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def summary(self) -> Dict:
        """Return memory statistics."""
        n = len(self._episodes)
        if n == 0:
            return {"n_episodes": 0, "n_gaps": len(self.knowledge_gaps()), "memory_dir": str(self._dir)}

        graded = [ep for ep in self._episodes if ep.faithfulness >= 0 or ep.relevance >= 0]
        rated_pos = [ep for ep in self._episodes if ep.user_rating == 1]
        rated_neg = [ep for ep in self._episodes if ep.user_rating == -1]
        faith_scores = [ep.faithfulness for ep in self._episodes if ep.faithfulness >= 0]

        return {
            "n_episodes": n,
            "n_graded": len(graded),
            "n_positive_ratings": len(rated_pos),
            "n_negative_ratings": len(rated_neg),
            "avg_faithfulness": round(float(np.mean(faith_scores)), 3) if faith_scores else -1.0,
            "avg_quality": round(float(np.mean([ep.quality_score() for ep in self._episodes])), 3),
            "n_gaps": len(self.knowledge_gaps()),
            "n_boosted_chunks": len(self.boosted_chunks()),
            "memory_dir": str(self._dir),
        }
