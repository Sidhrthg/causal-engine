"""
RAG evaluation loop for the minerals corpus.

Evaluates retrieval quality WITHOUT manual labels by generating synthetic
questions from corpus chunks (the chunk is the ground-truth answer source),
then measuring whether retrieval finds the originating chunk.

Pipeline
--------
1. Sample N chunks from the corpus
2. For each chunk: LLM generates K questions that can only be answered from it
3. For each (question, chunk) pair: retrieve top-k candidates
4. Score: Hit@k, MRR@k (retrieval), Faithfulness, Answer-Relevance (answer quality)

Usage
-----
    from src.minerals.rag_pipeline import RAGPipeline
    from src.minerals.rag_eval import RAGEvaluator

    pipeline = RAGPipeline()
    evaluator = RAGEvaluator(pipeline)

    # Generate 30 synthetic test questions from the corpus
    questions = evaluator.generate_questions(n_chunks=30, questions_per_chunk=2)

    # Evaluate retrieval
    report = evaluator.evaluate_retrieval(questions, top_k=5)
    print(f"Hit@5:  {report['hit_at_k']:.2%}")
    print(f"MRR@5:  {report['mrr']:.3f}")

    # Optionally grade answer quality (costs LLM calls)
    report = evaluator.evaluate_answers(questions[:10], top_k=5)
    print(f"Faithfulness:     {report['faithfulness_mean']:.2%}")
    print(f"Answer relevance: {report['relevance_mean']:.2%}")
"""

from __future__ import annotations

import json
import random
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.llm.chat import chat_completion, is_chat_available
from src.minerals.rag_pipeline import RAGPipeline
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class SyntheticQuestion:
    """A question generated from a specific corpus chunk."""
    question: str
    source_text: str          # The chunk text the question was generated from
    source_file: str          # Source document
    chunk_index: int          # Position in corpus (for identity check)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievalResult:
    """Result of retrieving for one question."""
    question: SyntheticQuestion
    retrieved: List[Dict]     # top-k retrieved chunks (normalized dicts)
    hit: bool                 # True if source chunk was in top-k
    rank: Optional[int]       # 1-based rank of source chunk (None if not found)
    reciprocal_rank: float    # 1/rank if found, else 0.0


@dataclass
class AnswerResult:
    """Result of a full RAG answer for one question."""
    question: SyntheticQuestion
    answer: str
    sources_used: List[Dict]
    faithfulness: float       # 0-1: is the answer grounded in retrieved docs?
    relevance: float          # 0-1: does the answer address the question?
    grade_reasoning: str


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

class RAGEvaluator:
    """
    Evaluate a RAGPipeline end-to-end using synthetic questions.

    Args:
        pipeline: Configured RAGPipeline instance.
        seed: Random seed for reproducible chunk sampling.
    """

    def __init__(self, pipeline: RAGPipeline, seed: int = 42):
        self.pipeline = pipeline
        self.rng = random.Random(seed)
        self._llm_ok = is_chat_available()
        if not self._llm_ok:
            logger.warning(
                "No LLM backend configured. Question generation and answer grading "
                "will be skipped. Set ANTHROPIC_API_KEY or LLM_BACKEND."
            )

    # ------------------------------------------------------------------
    # Step 1: Generate synthetic questions
    # ------------------------------------------------------------------

    def generate_questions(
        self,
        n_chunks: int = 20,
        questions_per_chunk: int = 2,
        min_chunk_chars: int = 150,
        llm_rate_limit_sleep: float = 0.5,
    ) -> List[SyntheticQuestion]:
        """
        Sample chunks from the corpus and use LLM to generate questions.

        Args:
            n_chunks: Number of chunks to sample.
            questions_per_chunk: Questions to generate per chunk (1-3 recommended).
            min_chunk_chars: Skip chunks shorter than this (likely table headers, etc.).
            llm_rate_limit_sleep: Seconds to sleep between LLM calls.

        Returns:
            List of SyntheticQuestion objects.
        """
        if not self._llm_ok:
            logger.error("LLM not available. Cannot generate synthetic questions.")
            return []

        chunks = getattr(self.pipeline._retriever, "chunks", [])
        if not chunks:
            logger.error("No chunks in corpus. Index documents first.")
            return []

        # Filter and sample
        eligible = [
            (i, c) for i, c in enumerate(chunks)
            if len(getattr(c, "text", "")) >= min_chunk_chars
        ]
        if len(eligible) < n_chunks:
            logger.warning(
                f"Only {len(eligible)} eligible chunks (need {n_chunks}). Using all."
            )
            n_chunks = len(eligible)

        sampled = self.rng.sample(eligible, n_chunks)
        logger.info(
            f"Generating {questions_per_chunk} question(s) each for {n_chunks} chunks..."
        )

        all_questions: List[SyntheticQuestion] = []
        for idx, (chunk_idx, chunk) in enumerate(sampled):
            text = getattr(chunk, "text", str(chunk))
            meta = getattr(chunk, "metadata", {})
            source_file = meta.get("source_file", meta.get("source", "?"))

            questions = self._generate_from_chunk(
                text, questions_per_chunk, source_file
            )
            for q in questions:
                all_questions.append(
                    SyntheticQuestion(
                        question=q,
                        source_text=text,
                        source_file=source_file,
                        chunk_index=chunk_idx,
                        metadata=dict(meta),
                    )
                )

            logger.info(f"  [{idx+1}/{n_chunks}] {source_file} → {len(questions)} Qs")
            if llm_rate_limit_sleep > 0 and idx < n_chunks - 1:
                time.sleep(llm_rate_limit_sleep)

        logger.info(f"Generated {len(all_questions)} synthetic questions total.")
        return all_questions

    def _generate_from_chunk(
        self,
        chunk_text: str,
        n_questions: int,
        source_file: str,
    ) -> List[str]:
        """Ask LLM to generate factoid questions answerable from this chunk."""
        prompt = (
            f"Read the following document passage and generate exactly {n_questions} "
            f"specific, factual questions that can be answered using ONLY the information "
            f"in this passage. Questions should be concrete and specific enough that "
            f"a search engine could retrieve this passage when given the question.\n\n"
            f"Passage:\n{chunk_text[:1500]}\n\n"
            f"Return ONLY a JSON array of question strings, e.g.:\n"
            f'["Question 1?", "Question 2?"]\n\n'
            f"JSON:"
        )
        try:
            raw = chat_completion(
                [{"role": "user", "content": prompt}],
                max_tokens=300,
            ).strip()
            # Extract JSON array
            match = re.search(r"\[.*?\]", raw, re.DOTALL)
            if match:
                questions = json.loads(match.group())
                return [str(q).strip() for q in questions if q][:n_questions]
        except Exception as exc:
            logger.warning(f"Question generation failed for {source_file}: {exc}")
        return []

    # ------------------------------------------------------------------
    # Step 2: Evaluate retrieval
    # ------------------------------------------------------------------

    def evaluate_retrieval(
        self,
        questions: List[SyntheticQuestion],
        top_k: int = 5,
    ) -> Dict[str, Any]:
        """
        Run retrieval for each question, check if the source chunk is found.

        A chunk is considered "found" if its text has >60% token overlap with
        the retrieved chunk text (exact match is too strict due to chunking
        boundary differences).

        Args:
            questions: List of SyntheticQuestion objects.
            top_k: Number of documents to retrieve.

        Returns:
            Dict with keys:
                - hit_at_k: fraction of questions where source was in top-k
                - mrr: Mean Reciprocal Rank
                - results: List[RetrievalResult] for detailed inspection
                - top_k: the k value used
                - n_questions: number of questions evaluated
        """
        if not questions:
            return {"hit_at_k": 0.0, "mrr": 0.0, "results": [], "top_k": top_k, "n_questions": 0}

        results: List[RetrievalResult] = []
        for sq in questions:
            retrieved = self.pipeline.retrieve(sq.question, top_k=top_k)
            hit, rank = self._find_source(sq.source_text, retrieved)
            rr = 1.0 / rank if rank is not None else 0.0
            results.append(RetrievalResult(
                question=sq,
                retrieved=retrieved,
                hit=hit,
                rank=rank,
                reciprocal_rank=rr,
            ))

        n = len(results)
        hit_at_k = sum(r.hit for r in results) / n
        mrr = sum(r.reciprocal_rank for r in results) / n

        logger.info(f"Retrieval eval — Hit@{top_k}: {hit_at_k:.2%}, MRR@{top_k}: {mrr:.3f}")

        return {
            "hit_at_k": hit_at_k,
            "mrr": mrr,
            "top_k": top_k,
            "n_questions": n,
            "results": results,
        }

    def _find_source(
        self,
        source_text: str,
        retrieved: List[Dict],
        overlap_threshold: float = 0.6,
    ) -> tuple[bool, Optional[int]]:
        """
        Check if the source chunk appears in the retrieved list.
        Uses token overlap instead of exact match to handle chunking differences.
        """
        source_tokens = set(source_text.lower().split())
        for rank, chunk in enumerate(retrieved, 1):
            candidate = chunk.get("text", "")
            cand_tokens = set(candidate.lower().split())
            if not source_tokens or not cand_tokens:
                continue
            intersection = source_tokens & cand_tokens
            # Overlap = |intersection| / |smaller set| (recall-oriented)
            overlap = len(intersection) / min(len(source_tokens), len(cand_tokens))
            if overlap >= overlap_threshold:
                return True, rank
        return False, None

    # ------------------------------------------------------------------
    # Step 3: Grade answer quality (LLM-as-judge)
    # ------------------------------------------------------------------

    def evaluate_answers(
        self,
        questions: List[SyntheticQuestion],
        top_k: int = 5,
        llm_rate_limit_sleep: float = 0.5,
    ) -> Dict[str, Any]:
        """
        Run full RAG (retrieve + generate) for each question, then grade with LLM.

        Grading rubric:
        - Faithfulness (0-1): Is every claim in the answer supported by retrieved docs?
        - Relevance (0-1): Does the answer directly address the question?

        Args:
            questions: List of SyntheticQuestion objects (use a small subset, e.g. 10-20).
            top_k: Documents to retrieve per question.
            llm_rate_limit_sleep: Sleep between LLM calls.

        Returns:
            Dict with faithfulness_mean, relevance_mean, and per-question results.
        """
        if not self._llm_ok:
            logger.error("LLM not available. Cannot grade answers.")
            return {"faithfulness_mean": 0.0, "relevance_mean": 0.0, "results": []}

        answer_results: List[AnswerResult] = []
        for i, sq in enumerate(questions):
            rag_out = self.pipeline.ask(sq.question, top_k=top_k)
            answer = rag_out.get("answer", "")
            sources = rag_out.get("sources", [])
            context = "\n\n".join(
                f"[{j+1}] {s['text_preview']}" for j, s in enumerate(sources)
            )
            faith, rel, reasoning = self._grade_answer(sq.question, answer, context)
            answer_results.append(AnswerResult(
                question=sq,
                answer=answer,
                sources_used=sources,
                faithfulness=faith,
                relevance=rel,
                grade_reasoning=reasoning,
            ))
            logger.info(
                f"  [{i+1}/{len(questions)}] faith={faith:.2f} rel={rel:.2f}  "
                f"Q: {sq.question[:60]}..."
            )
            if llm_rate_limit_sleep > 0 and i < len(questions) - 1:
                time.sleep(llm_rate_limit_sleep)

        if not answer_results:
            return {"faithfulness_mean": 0.0, "relevance_mean": 0.0, "results": []}

        faith_mean = sum(r.faithfulness for r in answer_results) / len(answer_results)
        rel_mean = sum(r.relevance for r in answer_results) / len(answer_results)

        logger.info(
            f"Answer eval — Faithfulness: {faith_mean:.2%}, Relevance: {rel_mean:.2%}"
        )

        return {
            "faithfulness_mean": faith_mean,
            "relevance_mean": rel_mean,
            "n_questions": len(answer_results),
            "results": answer_results,
        }

    def _grade_answer(
        self, question: str, answer: str, context: str
    ) -> tuple[float, float, str]:
        """LLM-as-judge: score faithfulness and relevance."""
        prompt = (
            "You are a strict grader evaluating a RAG system's answer.\n\n"
            f"Question: {question}\n\n"
            f"Retrieved context:\n{context[:3000]}\n\n"
            f"Answer: {answer[:1500]}\n\n"
            "Grade the answer on two dimensions (0.0 to 1.0):\n"
            "1. Faithfulness: Are all claims in the answer supported by the context? "
            "   (1.0 = fully grounded, 0.0 = hallucinated)\n"
            "2. Relevance: Does the answer directly address the question? "
            "   (1.0 = fully relevant, 0.0 = off-topic)\n\n"
            "Return ONLY a JSON object:\n"
            '{"faithfulness": 0.0-1.0, "relevance": 0.0-1.0, "reasoning": "one sentence"}'
        )
        try:
            raw = chat_completion(
                [{"role": "user", "content": prompt}],
                max_tokens=150,
            ).strip()
            # Extract JSON
            match = re.search(r"\{.*?\}", raw, re.DOTALL)
            if match:
                grades = json.loads(match.group())
                faith = float(grades.get("faithfulness", 0.5))
                rel = float(grades.get("relevance", 0.5))
                reasoning = str(grades.get("reasoning", ""))
                return min(1.0, max(0.0, faith)), min(1.0, max(0.0, rel)), reasoning
        except Exception as exc:
            logger.warning(f"Answer grading failed: {exc}")
        return 0.5, 0.5, "grading failed"

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def print_retrieval_report(self, report: Dict[str, Any], show_failures: bool = True) -> None:
        """Print a human-readable retrieval evaluation report."""
        k = report.get("top_k", "?")
        n = report.get("n_questions", 0)
        print(f"\n{'='*60}")
        print(f"RAG Retrieval Evaluation  (n={n}, k={k})")
        print(f"{'='*60}")
        print(f"Hit@{k}:  {report.get('hit_at_k', 0):.2%}  "
              f"({'fraction of questions where source chunk was retrieved'})")
        print(f"MRR@{k}:  {report.get('mrr', 0):.3f}  "
              f"({'mean reciprocal rank — higher = source ranked earlier'})")

        results: List[RetrievalResult] = report.get("results", [])
        if show_failures and results:
            failures = [r for r in results if not r.hit]
            print(f"\nFailed retrievals ({len(failures)}/{n}):")
            for r in failures[:5]:
                print(f"  Q: {r.question.question[:70]}...")
                print(f"     Source: {r.question.source_file}")
                if r.retrieved:
                    print(f"     Got instead: {r.retrieved[0]['metadata'].get('source_file', '?')}")

    def print_answer_report(self, report: Dict[str, Any]) -> None:
        """Print a human-readable answer quality report."""
        n = report.get("n_questions", 0)
        print(f"\n{'='*60}")
        print(f"RAG Answer Quality Evaluation  (n={n})")
        print(f"{'='*60}")
        print(f"Faithfulness:     {report.get('faithfulness_mean', 0):.2%}")
        print(f"Answer Relevance: {report.get('relevance_mean', 0):.2%}")

        results: List[AnswerResult] = report.get("results", [])
        if results:
            worst = sorted(results, key=lambda r: r.faithfulness + r.relevance)[:3]
            print(f"\nLowest-scoring answers:")
            for r in worst:
                print(f"  Q: {r.question.question[:60]}...")
                print(f"     Faithfulness={r.faithfulness:.2f} Relevance={r.relevance:.2f}")
                print(f"     {r.grade_reasoning}")

    def learn(
        self,
        retrieval_report: Dict,
        answer_report: Optional[Dict] = None,
        min_faithfulness: float = 0.7,
        min_relevance: float = 0.7,
    ) -> Dict[str, int]:
        """
        Feed evaluation results back into the pipeline's episodic memory.

        High-quality answers (faithfulness + relevance above threshold) are stored
        as few-shot examples. Failed retrievals are marked as knowledge gaps.

        Call this after ``evaluate_retrieval()`` / ``evaluate_answers()`` to close
        the self-improvement loop::

            questions = ev.generate_questions(30)
            ret_report  = ev.evaluate_retrieval(questions, top_k=5)
            ans_report  = ev.evaluate_answers(questions[:10], top_k=5)
            ev.learn(ret_report, ans_report)        # ← feeds memory

        Args:
            retrieval_report: From ``evaluate_retrieval()``.
            answer_report: From ``evaluate_answers()`` (optional).
            min_faithfulness / min_relevance: Quality thresholds.

        Returns:
            ``{"stored": N, "gaps": M}``
        """
        if self.pipeline.memory is None:
            logger.warning("Pipeline has no memory configured; learn() is a no-op.")
            return {"stored": 0, "gaps": 0}
        return self.pipeline.memory.learn_from_eval(
            retrieval_report=retrieval_report,
            answer_report=answer_report,
            min_faithfulness=min_faithfulness,
            min_relevance=min_relevance,
        )

    def save_report(self, retrieval_report: Dict, answer_report: Dict, path: str) -> None:
        """Save evaluation reports to JSON (serializes scores, not full objects)."""
        import json as _json

        def _serialize(obj):
            if isinstance(obj, (RetrievalResult, AnswerResult)):
                return {k: _serialize(v) for k, v in obj.__dict__.items()}
            if isinstance(obj, SyntheticQuestion):
                return {
                    "question": obj.question,
                    "source_file": obj.source_file,
                    "chunk_index": obj.chunk_index,
                    "source_text_preview": obj.source_text[:200],
                }
            if isinstance(obj, list):
                return [_serialize(i) for i in obj]
            if isinstance(obj, dict):
                return {k: _serialize(v) for k, v in obj.items()}
            return obj

        out = {
            "retrieval": {
                k: _serialize(v) for k, v in retrieval_report.items()
            },
            "answers": {
                k: _serialize(v) for k, v in answer_report.items()
            },
        }
        with open(path, "w") as f:
            _json.dump(out, f, indent=2, default=str)
        logger.info(f"Saved eval report to {path}")
