"""KG-grounded LLM query interface for causal supply chain reasoning.

Combines metapath-based subgraph extraction from the CausalKnowledgeGraph
with RAG document retrieval and LLM synthesis to answer causal questions
grounded in both structured knowledge and document evidence.

Reference: "Paths to Causality" (arXiv 2506.08771) — metapath-based
subgraph extraction outperforms similarity-based retrieval by ~7.7 F1.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from .knowledge_graph import (
    CausalKnowledgeGraph,
    Entity,
    EntityType,
    RelationType,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Data classes
# ============================================================================


@dataclass
class TypedPathStep:
    """A single step in a typed metapath."""

    entity_id: str
    entity_type: EntityType
    relation_to_next: Optional[RelationType] = None
    mechanism: str = ""


@dataclass
class KGPath:
    """A typed path through the knowledge graph with mechanism annotations."""

    steps: List[TypedPathStep]
    linearized: str = ""
    confidence: float = 0.0


# ============================================================================
# KGQueryEngine
# ============================================================================


class KGQueryEngine:
    """
    KG-grounded LLM query engine for causal supply chain reasoning.

    Combines:
    - Metapath-based subgraph extraction from CausalKnowledgeGraph
    - RAG document retrieval (any retriever with .retrieve(query, top_k))
    - LLM synthesis via chat_completion()

    Args:
        kg: A populated CausalKnowledgeGraph instance.
        retriever: Any retriever with .retrieve(query, top_k) -> List[Dict].
        llm_backend: Override LLM backend (default: from env LLM_BACKEND).
        max_path_depth: Maximum depth for metapath extraction.
    """

    def __init__(
        self,
        kg: CausalKnowledgeGraph,
        retriever: Any = None,
        llm_backend: Optional[str] = None,
        max_path_depth: int = 5,
    ) -> None:
        self._kg = kg
        self._retriever = retriever
        self._llm_backend = llm_backend
        self._max_path_depth = max_path_depth
        self._entity_index: Dict[str, str] = self._build_entity_index()

    # ------------------------------------------------------------------
    # Entity resolution
    # ------------------------------------------------------------------

    def _build_entity_index(self) -> Dict[str, str]:
        """Build a case-insensitive lookup from entity names/aliases to IDs."""
        index: Dict[str, str] = {}
        for entity_id, entity in self._kg._entities.items():
            index[entity_id.lower()] = entity_id
            index[entity_id.lower().replace("_", " ")] = entity_id
            for alias in entity.aliases:
                index[alias.lower()] = entity_id
        return index

    def resolve_entities(self, question: str) -> List[str]:
        """
        Identify KG entities mentioned in a natural language question.

        Uses longest-match-first substring matching against the entity index.
        Falls back to LLM extraction if fewer than 2 entities found.
        """
        question_lower = question.lower()
        found: List[str] = []
        found_spans: List[Tuple[int, int]] = []

        # Sort by length descending: "china_export_controls" before "china"
        sorted_keys = sorted(self._entity_index.keys(), key=len, reverse=True)

        for key in sorted_keys:
            if key in question_lower:
                start = question_lower.find(key)
                end = start + len(key)
                if not any(s <= start < e or s < end <= e for s, e in found_spans):
                    eid = self._entity_index[key]
                    if eid not in found:
                        found.append(eid)
                        found_spans.append((start, end))

        if len(found) < 2:
            llm_entities = self._llm_entity_extraction(question, found)
            for eid in llm_entities:
                if eid not in found:
                    found.append(eid)

        return found

    def _llm_entity_extraction(
        self, question: str, already_found: List[str]
    ) -> List[str]:
        """Use LLM to identify KG entities referenced in the question."""
        try:
            from ..llm.chat import chat_completion, is_chat_available

            if not is_chat_available(self._llm_backend):
                return []
        except ImportError:
            return []

        all_ids = sorted(self._kg._entities.keys())
        prompt = (
            f'Given this question about critical mineral supply chains:\n'
            f'"{question}"\n\n'
            f'Which of these knowledge graph entity IDs are referenced or relevant?\n'
            f'Available entities: {", ".join(all_ids)}\n\n'
            f'Already identified: {already_found}\n\n'
            f'Return ONLY a JSON array of additional entity ID strings (max 5). '
            f'If none, return [].\nJSON array:'
        )

        try:
            response = chat_completion(
                [{"role": "user", "content": prompt}],
                max_tokens=300,
                backend=self._llm_backend,
            )
            match = re.search(r"\[.*?\]", response, re.DOTALL)
            if match:
                ids = json.loads(match.group())
                return [eid for eid in ids if eid in self._kg._entities]
        except Exception as e:
            logger.warning(f"LLM entity extraction failed: {e}")
        return []

    # ------------------------------------------------------------------
    # Metapath extraction and linearization
    # ------------------------------------------------------------------

    def extract_metapaths(
        self,
        source_id: str,
        target_id: str,
        relation_types: Optional[List[RelationType]] = None,
    ) -> List[KGPath]:
        """
        Extract typed metapaths between two entities.

        Each path is annotated with mechanism text from CAUSES edges
        and linearized into a natural language representation.
        """
        raw_paths = self._kg.find_paths(
            source_id,
            target_id,
            relation_types=relation_types,
            max_depth=self._max_path_depth,
        )

        kg_paths: List[KGPath] = []
        for path_nodes in raw_paths:
            steps: List[TypedPathStep] = []
            path_confidence = 1.0

            for i, node_id in enumerate(path_nodes):
                entity = self._kg.get_entity(node_id)
                etype = entity.entity_type if entity else EntityType.COMMODITY

                rel_type = None
                mechanism = ""
                if i < len(path_nodes) - 1:
                    next_node = path_nodes[i + 1]
                    rels = self._kg.get_relationships(
                        source_id=node_id, target_id=next_node
                    )
                    if rels:
                        causes = [
                            r for r in rels if r.relation_type == RelationType.CAUSES
                        ]
                        chosen = causes[0] if causes else rels[0]
                        rel_type = chosen.relation_type
                        mechanism = chosen.mechanism
                        conf = {"HIGH": 1.0, "MEDIUM": 0.7, "LOW": 0.4}
                        path_confidence *= conf.get(chosen.confidence, 0.7)

                steps.append(
                    TypedPathStep(
                        entity_id=node_id,
                        entity_type=etype,
                        relation_to_next=rel_type,
                        mechanism=mechanism,
                    )
                )

            linearized = self._linearize_path(steps)
            kg_paths.append(
                KGPath(steps=steps, linearized=linearized, confidence=path_confidence)
            )

        kg_paths.sort(key=lambda p: p.confidence, reverse=True)
        return kg_paths

    def extract_neighborhood_paths(
        self, entity_id: str, max_paths: int = 10
    ) -> List[KGPath]:
        """
        Extract paths radiating from a single entity (when only one
        entity is identified in the question).
        """
        kg_paths: List[KGPath] = []
        outgoing = self._kg.get_relationships(
            source_id=entity_id, relation_type=RelationType.CAUSES
        )
        incoming = self._kg.get_relationships(
            target_id=entity_id, relation_type=RelationType.CAUSES
        )

        seen: Set[Tuple[str, str]] = set()
        for rel in outgoing + incoming:
            pair = (rel.source_id, rel.target_id)
            if pair in seen:
                continue
            seen.add(pair)

            src_ent = self._kg.get_entity(rel.source_id)
            tgt_ent = self._kg.get_entity(rel.target_id)
            steps = [
                TypedPathStep(
                    entity_id=rel.source_id,
                    entity_type=src_ent.entity_type if src_ent else EntityType.COMMODITY,
                    relation_to_next=rel.relation_type,
                    mechanism=rel.mechanism,
                ),
                TypedPathStep(
                    entity_id=rel.target_id,
                    entity_type=tgt_ent.entity_type if tgt_ent else EntityType.COMMODITY,
                ),
            ]
            conf = {"HIGH": 1.0, "MEDIUM": 0.7, "LOW": 0.4}
            kg_paths.append(
                KGPath(
                    steps=steps,
                    linearized=self._linearize_path(steps),
                    confidence=conf.get(rel.confidence, 0.7),
                )
            )

        kg_paths.sort(key=lambda p: p.confidence, reverse=True)
        return kg_paths[:max_paths]

    def _linearize_path(self, steps: List[TypedPathStep]) -> str:
        """Convert a typed path into a readable string.

        Example:
          china export controls (POLICY) --CAUSES--> graphite (COMMODITY)
            [mechanism: export restrictions reduce global supply]
        """
        parts: List[str] = []
        for step in steps:
            label = step.entity_id.replace("_", " ")
            parts.append(f"{label} ({step.entity_type.value.upper()})")
            if step.relation_to_next is not None:
                rel = step.relation_to_next.value.upper()
                mech = f"  [{step.mechanism}]" if step.mechanism else ""
                parts.append(f"  --{rel}-->{mech}")
        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Targeted RAG retrieval
    # ------------------------------------------------------------------

    def _targeted_rag_queries(
        self, kg_paths: List[KGPath], original_question: str
    ) -> List[str]:
        """Generate RAG queries from KG path mechanisms (not raw question)."""
        queries: List[str] = []
        seen: Set[str] = set()

        for path in kg_paths[:5]:
            for step in path.steps:
                if step.mechanism and step.mechanism not in seen:
                    seen.add(step.mechanism)
                    entities = " ".join(
                        s.entity_id.replace("_", " ") for s in path.steps
                    )
                    queries.append(f"{step.mechanism} {entities}")

        # Always include the original question as fallback
        queries.append(original_question)
        return queries

    def _retrieve_documents(
        self, queries: List[str], top_k_per_query: int = 3
    ) -> List[Dict[str, Any]]:
        """Retrieve documents using targeted queries, deduplicated."""
        if self._retriever is None:
            return []

        all_docs: List[Dict[str, Any]] = []
        seen: Set[str] = set()

        for query in queries:
            try:
                docs = self._retriever.retrieve(query=query, top_k=top_k_per_query)
            except Exception as e:
                logger.warning(f"RAG retrieval failed for '{query[:60]}': {e}")
                continue
            for doc in docs:
                key = doc.get("text", "")[:200]
                if key not in seen:
                    seen.add(key)
                    all_docs.append(doc)

        return all_docs

    # ------------------------------------------------------------------
    # LLM synthesis
    # ------------------------------------------------------------------

    def _synthesize_answer(
        self,
        question: str,
        kg_paths: List[KGPath],
        documents: List[Dict[str, Any]],
    ) -> Tuple[str, float]:
        """Synthesize an answer grounded in KG paths and documents."""
        try:
            from ..llm.chat import chat_completion, is_chat_available

            if not is_chat_available(self._llm_backend):
                return self._fallback_answer(kg_paths), 0.3
        except ImportError:
            return self._fallback_answer(kg_paths), 0.3

        paths_text = "\n\n".join(
            f"Path {i + 1} (confidence: {p.confidence:.2f}):\n{p.linearized}"
            for i, p in enumerate(kg_paths[:5])
        )
        docs_text = "\n\n---\n\n".join(
            f"[Source: {d.get('metadata', {}).get('source_file', d.get('metadata', {}).get('source', 'unknown'))}]\n"
            f"{d.get('text', '')[:500]}"
            for d in documents[:8]
        )

        prompt = f"""You are a critical minerals supply chain analyst. Answer the question using BOTH the causal knowledge graph paths AND the supporting documents.

QUESTION: {question}

CAUSAL KNOWLEDGE GRAPH PATHS (structured causal relationships):
{paths_text or "(No paths found)"}

SUPPORTING DOCUMENTS (from USGS, IEA, trade reports):
{docs_text or "(No documents retrieved)"}

Instructions:
1. Ground your answer in the KG paths — cite specific causal mechanisms.
2. Support claims with document evidence where available.
3. If paths and documents conflict, note the discrepancy.
4. Rate confidence (0.0-1.0) based on evidence quality.

Respond as JSON:
{{
  "answer": "Your detailed answer...",
  "confidence": 0.85,
  "key_mechanisms": ["mechanism 1", "mechanism 2"],
  "evidence_gaps": ["any gaps"]
}}

JSON:"""

        try:
            response = chat_completion(
                [{"role": "user", "content": prompt}],
                max_tokens=3000,
                backend=self._llm_backend,
            )
            match = re.search(r"\{.*\}", response, re.DOTALL)
            if match:
                parsed = json.loads(match.group())
                answer = parsed.get("answer", response)
                confidence = float(parsed.get("confidence", 0.7))
                return answer, min(confidence, 1.0)
            return response, 0.7
        except Exception as e:
            logger.warning(f"LLM synthesis failed: {e}")
            return self._fallback_answer(kg_paths), 0.3

    def _fallback_answer(self, kg_paths: List[KGPath]) -> str:
        """Generate answer from KG paths alone (no LLM)."""
        if not kg_paths:
            return "No causal paths found between the identified entities."
        lines = ["Based on the causal knowledge graph:\n"]
        for i, path in enumerate(kg_paths[:5]):
            lines.append(f"Path {i + 1} (confidence: {path.confidence:.2f}):")
            lines.append(path.linearized)
            lines.append("")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Main query interface
    # ------------------------------------------------------------------

    def query(self, question: str) -> dict:
        """
        Answer a causal question grounded in KG structure and documents.

        Returns dict with: answer, kg_paths, supporting_docs, confidence,
        entity_ids_found, raw_question.
        """
        # 1. Resolve entities
        entity_ids = self.resolve_entities(question)
        logger.info(f"Resolved entities: {entity_ids}")

        # 2. Extract metapaths
        kg_paths: List[KGPath] = []
        if len(entity_ids) >= 2:
            for i in range(len(entity_ids)):
                for j in range(len(entity_ids)):
                    if i != j:
                        paths = self.extract_metapaths(entity_ids[i], entity_ids[j])
                        kg_paths.extend(paths)
        elif len(entity_ids) == 1:
            kg_paths = self.extract_neighborhood_paths(entity_ids[0])

        # Deduplicate
        seen_linear: Set[str] = set()
        unique: List[KGPath] = []
        for p in kg_paths:
            if p.linearized not in seen_linear:
                seen_linear.add(p.linearized)
                unique.append(p)
        kg_paths = sorted(unique, key=lambda p: p.confidence, reverse=True)[:10]

        # 3. Targeted RAG retrieval
        rag_queries = self._targeted_rag_queries(kg_paths, question)
        documents = self._retrieve_documents(rag_queries)

        # 4. LLM synthesis
        answer, confidence = self._synthesize_answer(question, kg_paths, documents)

        return {
            "answer": answer,
            "kg_paths": [
                {
                    "steps": [
                        {
                            "entity_id": s.entity_id,
                            "entity_type": s.entity_type.value,
                            "relation": s.relation_to_next.value
                            if s.relation_to_next
                            else None,
                            "mechanism": s.mechanism,
                        }
                        for s in p.steps
                    ],
                    "linearized": p.linearized,
                    "confidence": p.confidence,
                }
                for p in kg_paths
            ],
            "supporting_docs": [
                {
                    "text": d.get("text", "")[:500],
                    "metadata": d.get("metadata", {}),
                }
                for d in documents[:10]
            ],
            "confidence": confidence,
            "entity_ids_found": entity_ids,
            "raw_question": question,
        }

    # ------------------------------------------------------------------
    # Mechanism completeness audit
    # ------------------------------------------------------------------

    def audit_mechanisms(self, entity_id: str) -> dict:
        """
        Audit mechanism completeness for a KG entity.

        Compares existing CAUSES mechanisms against what the document
        corpus describes. Identifies gaps where documents mention causal
        relationships not captured in the KG.
        """
        eid = self._kg.resolve_id(entity_id)
        entity = self._kg.get_entity(eid)
        if entity is None:
            return {
                "entity_id": entity_id,
                "existing_mechanisms": [],
                "suggested_new_mechanisms": [],
                "coverage_score": 0.0,
                "supporting_docs": [],
                "audit_summary": f"Entity '{entity_id}' not found in knowledge graph.",
            }

        outgoing = self._kg.get_relationships(
            source_id=eid, relation_type=RelationType.CAUSES
        )
        incoming = self._kg.get_relationships(
            target_id=eid, relation_type=RelationType.CAUSES
        )

        existing: List[Dict[str, Any]] = []
        for rel in outgoing + incoming:
            existing.append(
                {
                    "source": rel.source_id,
                    "target": rel.target_id,
                    "mechanism": rel.mechanism,
                    "confidence": rel.confidence,
                }
            )

        # Retrieve documents about this entity
        label = eid.replace("_", " ")
        rag_queries = [
            f"causal mechanisms {label} supply chain",
            f"what causes {label} disruption shortage",
            f"{label} impact factors drivers effects",
        ]
        all_docs: List[Dict[str, Any]] = []
        seen: Set[str] = set()
        for q in rag_queries:
            for doc in self._retrieve_documents([q], top_k_per_query=5):
                key = doc.get("text", "")[:200]
                if key not in seen:
                    seen.add(key)
                    all_docs.append(doc)

        # LLM audit
        suggested, coverage, summary = self._llm_mechanism_audit(
            eid, existing, all_docs
        )

        return {
            "entity_id": eid,
            "existing_mechanisms": existing,
            "suggested_new_mechanisms": suggested,
            "coverage_score": coverage,
            "supporting_docs": [
                {"text": d.get("text", "")[:500], "metadata": d.get("metadata", {})}
                for d in all_docs[:10]
            ],
            "audit_summary": summary,
        }

    def _llm_mechanism_audit(
        self,
        entity_id: str,
        existing: List[Dict[str, Any]],
        documents: List[Dict[str, Any]],
    ) -> Tuple[List[Dict[str, Any]], float, str]:
        """Ask LLM to compare KG mechanisms against document evidence."""
        try:
            from ..llm.chat import chat_completion, is_chat_available

            if not is_chat_available(self._llm_backend):
                return [], 0.5, "LLM unavailable; cannot audit."
        except ImportError:
            return [], 0.5, "LLM not available."

        existing_text = "\n".join(
            f"  - {m['source']} -> {m['target']}: {m['mechanism']}"
            for m in existing
        ) or "  (none)"

        docs_text = "\n\n---\n\n".join(
            f"[{d.get('metadata', {}).get('source_file', d.get('metadata', {}).get('source', '?'))}]\n"
            f"{d.get('text', '')[:400]}"
            for d in documents[:8]
        ) or "(no documents retrieved)"

        prompt = f"""You are auditing causal mechanism completeness for "{entity_id.replace('_', ' ')}" in a critical minerals knowledge graph.

EXISTING KG MECHANISMS:
{existing_text}

DOCUMENTS FROM CORPUS:
{docs_text}

Tasks:
1. Identify causal mechanisms in the documents NOT captured in the KG.
2. Rate coverage: fraction of document-evidenced mechanisms already in KG (0.0-1.0).
3. For each new mechanism, specify source entity, target entity, mechanism text, and document source.

JSON response:
{{
  "suggested_new_mechanisms": [
    {{"source": "entity_id", "target": "entity_id", "mechanism": "description", "evidence_source": "filename"}}
  ],
  "coverage_score": 0.75,
  "summary": "Brief findings"
}}

JSON:"""

        try:
            response = chat_completion(
                [{"role": "user", "content": prompt}],
                max_tokens=3000,
                backend=self._llm_backend,
            )
            match = re.search(r"\{.*\}", response, re.DOTALL)
            if match:
                parsed = json.loads(match.group())
                return (
                    parsed.get("suggested_new_mechanisms", []),
                    min(float(parsed.get("coverage_score", 0.5)), 1.0),
                    parsed.get("summary", "Audit complete."),
                )
            return [], 0.5, response
        except Exception as e:
            logger.warning(f"LLM mechanism audit failed: {e}")
            return [], 0.5, f"Audit failed: {e}"
