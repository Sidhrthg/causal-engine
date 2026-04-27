"""
RAG-driven Knowledge Graph extraction and enrichment.

Takes retrieved document chunks, uses an LLM to extract (subject, predicate,
object) triples, maps predicates to ``RelationType``, and merges the result
into an existing ``CausalKnowledgeGraph`` via ``kg.merge()``.

Each merge call reinforces existing edges (confidence accumulates) and adds
novel edges at medium confidence — so KGs literally build off one another by
running this repeatedly as new documents arrive.

Usage
-----
    from src.minerals.knowledge_graph import build_critical_minerals_kg
    from src.minerals.rag_pipeline import RAGPipeline
    from src.minerals.kg_extractor import KGExtractor

    kg = build_critical_minerals_kg()
    pipeline = RAGPipeline()
    extractor = KGExtractor(pipeline)

    # Enrich KG from a query (retrieves docs, extracts triples, merges)
    n = extractor.enrich(kg, "China graphite export restrictions 2023")
    print(f"Added/reinforced {n} relationships")

    # Save the enriched KG
    kg.save("data/knowledge_graph.json")

    # Later: load and keep enriching
    kg2 = CausalKnowledgeGraph.load("data/knowledge_graph.json")
    extractor.enrich(kg2, "lithium supply disruption Chile 2022")
    kg2.save("data/knowledge_graph.json")
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.llm.chat import chat_completion, is_chat_available
from src.minerals.knowledge_graph import (
    CausalKnowledgeGraph,
    Entity,
    EntityType,
    Relationship,
    RelationType,
)
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Predicate → RelationType mapping
# ---------------------------------------------------------------------------

# LLM is prompted to use these exact predicate strings.
# The mapping is case-insensitive.
_PREDICATE_MAP: Dict[str, RelationType] = {
    "causes":        RelationType.CAUSES,
    "leads_to":      RelationType.CAUSES,
    "results_in":    RelationType.CAUSES,
    "produces":      RelationType.PRODUCES,
    "mines":         RelationType.PRODUCES,
    "extracts":      RelationType.PRODUCES,
    "exports_to":    RelationType.EXPORTS_TO,
    "imports_from":  RelationType.IMPORTS_FROM,
    "supplies":      RelationType.SUPPLIES,
    "processes":     RelationType.PROCESSES,
    "depends_on":    RelationType.DEPENDS_ON,
    "relies_on":     RelationType.DEPENDS_ON,
    "substitutes":   RelationType.SUBSTITUTES,
    "replaces":      RelationType.SUBSTITUTES,
    "competes_with": RelationType.COMPETES_WITH,
    "inputs":        RelationType.INPUTS,
    "outputs":       RelationType.OUTPUTS,
    "regulates":     RelationType.REGULATES,
    "restricts":     RelationType.REGULATES,
    "bans":          RelationType.REGULATES,
    "disrupts":      RelationType.DISRUPTS,
    "mitigates":     RelationType.MITIGATES,
    "located_in":    RelationType.LOCATED_IN,
    "part_of":       RelationType.PART_OF,
    "is_a":          RelationType.IS_A,
    "consumes":      RelationType.CONSUMES,
    "enables":       RelationType.ENABLES,
}

# Entity type heuristics for auto-classification
_ENTITY_KEYWORDS: List[tuple] = [
    (EntityType.COUNTRY,    ["china", "drc", "congo", "australia", "chile", "india", "usa", "europe"]),
    (EntityType.COMMODITY,  ["graphite", "lithium", "cobalt", "copper", "nickel", "antimony",
                             "rare earth", "tungsten", "tantalum", "gallium", "germanium"]),
    (EntityType.COMPANY,    ["corp", "inc", "ltd", "gmbh", "company", "mine", "mining"]),
    (EntityType.POLICY,     ["policy", "regulation", "tax", "tariff", "quota", "restriction",
                             "ban", "export control", "sanction"]),
    (EntityType.ECONOMIC_INDICATOR, ["price", "demand", "supply", "inventory", "shortage",
                                      "production", "gdp", "trade", "export", "import"]),
]


def _classify_entity(name: str) -> EntityType:
    """Heuristically classify an entity name into an EntityType."""
    lower = name.lower()
    for entity_type, keywords in _ENTITY_KEYWORDS:
        if any(kw in lower for kw in keywords):
            return entity_type
    return EntityType.COMMODITY  # safe default


def _map_predicate(predicate: str) -> Optional[RelationType]:
    """Map a free-text predicate string to a RelationType."""
    key = predicate.lower().strip().replace(" ", "_").replace("-", "_")
    if key in _PREDICATE_MAP:
        return _PREDICATE_MAP[key]
    # Fuzzy: check if any known key is a substring
    for k, v in _PREDICATE_MAP.items():
        if k in key or key in k:
            return v
    return None


# ---------------------------------------------------------------------------
# Extractor
# ---------------------------------------------------------------------------

_EXTRACTION_PROMPT = """\
You are a knowledge graph extractor specialising in supply chain and commodities.

Read the passage below and extract ALL meaningful relationships as structured triples.

For each relationship output a JSON object with these fields:
  "subject"    – the entity that acts or causes (string)
  "predicate"  – relationship type, choose EXACTLY one of:
                 CAUSES, LEADS_TO, RESULTS_IN, PRODUCES, MINES, EXTRACTS,
                 EXPORTS_TO, IMPORTS_FROM, SUPPLIES, PROCESSES,
                 DEPENDS_ON, RELIES_ON, SUBSTITUTES, REPLACES, COMPETES_WITH,
                 INPUTS, OUTPUTS, REGULATES, RESTRICTS, BANS, DISRUPTS,
                 MITIGATES, LOCATED_IN, PART_OF, IS_A, CONSUMES, ENABLES
  "object"     – the entity that is affected (string)
  "confidence" – 0.0-1.0 (how certain you are from the text alone)
  "evidence"   – a short verbatim quote from the passage supporting this triple
  "year"       – year or period if explicitly mentioned (e.g. "2008" or "2010-2015"), else null

Return ONLY a JSON array, e.g.:
[
  {{"subject": "China", "predicate": "EXPORTS_TO", "object": "Japan",
    "confidence": 0.9, "evidence": "China exported...", "year": "2020"}},
  ...
]

Passage:
{text}

JSON:"""


class KGExtractor:
    """
    Extracts (subject, predicate, object) triples from text using an LLM
    and merges them into a ``CausalKnowledgeGraph``.

    Args:
        pipeline: Optional ``RAGPipeline`` for document retrieval.
            Required only for ``enrich()``.  Pass ``None`` if you only
            need ``extract_from_text()``.
        min_confidence: Triples below this confidence are discarded.
    """

    def __init__(
        self,
        pipeline=None,
        min_confidence: float = 0.4,
        cache_path: Optional[str] = None,
        use_cache: bool = True,
    ):
        self.pipeline = pipeline
        self.min_confidence = min_confidence
        # Triple extraction cache: sha1(text+source+year) -> list of triples.
        # Avoids repeat Claude calls when the same doc chunk is seen across
        # multiple scenarios or reruns. Override path via KG_EXTRACTOR_CACHE env.
        default_path = os.getenv(
            "KG_EXTRACTOR_CACHE",
            "outputs/cache/kg_extractor_triples.json",
        )
        self._cache_path: Path = Path(cache_path or default_path)
        self._use_cache: bool = use_cache
        self._cache: Dict[str, List[Dict[str, Any]]] = self._load_cache()

    def _load_cache(self) -> Dict[str, List[Dict[str, Any]]]:
        if not self._use_cache:
            return {}
        try:
            if self._cache_path.exists():
                return json.loads(self._cache_path.read_text())
        except Exception as exc:
            logger.warning(f"Could not load triple cache {self._cache_path}: {exc}")
        return {}

    def _save_cache(self) -> None:
        if not self._use_cache:
            return
        try:
            self._cache_path.parent.mkdir(parents=True, exist_ok=True)
            self._cache_path.write_text(json.dumps(self._cache, indent=2))
        except Exception as exc:
            logger.warning(f"Could not write triple cache {self._cache_path}: {exc}")

    @staticmethod
    def _cache_key(text: str, source: str, year: Optional[str]) -> str:
        payload = f"{text[:3000]}||{source}||{year or ''}"
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()

    # ------------------------------------------------------------------
    # Core extraction
    # ------------------------------------------------------------------

    def extract_from_text(
        self,
        text: str,
        source: str = "",
        year: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Use LLM to extract triples from *text*.

        Args:
            text: Document passage to extract from.
            source: Source file / document identifier (stored as provenance).
            year: Year hint to attach to triples that don't specify one.

        Returns:
            List of triple dicts::

                {"subject": str, "predicate": str, "object": str,
                 "confidence": float, "evidence": str,
                 "year": str | None, "source": str}
        """
        cache_key = self._cache_key(text, source, year) if self._use_cache else ""
        if cache_key and cache_key in self._cache:
            return self._cache[cache_key]

        if not is_chat_available():
            logger.warning("No LLM backend — extract_from_text returns []")
            return []

        prompt = _EXTRACTION_PROMPT.format(text=text[:3000])
        try:
            raw = chat_completion(
                [{"role": "user", "content": prompt}],
                max_tokens=1_500,
            ).strip()
        except Exception as exc:
            logger.warning(f"LLM extraction failed: {exc}")
            return []

        # Parse JSON array from response
        match = re.search(r"\[.*\]", raw, re.DOTALL)
        if not match:
            logger.debug(f"No JSON array in LLM response: {raw[:200]}")
            return []
        try:
            triples = json.loads(match.group())
        except json.JSONDecodeError as exc:
            logger.warning(f"JSON parse error in extraction: {exc}")
            return []

        results = []
        for t in triples:
            if not isinstance(t, dict):
                continue
            conf = float(t.get("confidence", 0.5))
            if conf < self.min_confidence:
                continue
            results.append({
                "subject":    str(t.get("subject", "")).strip(),
                "predicate":  str(t.get("predicate", "")).strip(),
                "object":     str(t.get("object", "")).strip(),
                "confidence": conf,
                "evidence":   str(t.get("evidence", "")).strip(),
                "year":       t.get("year") or year,
                "source":     source,
            })

        logger.debug(f"Extracted {len(results)} triples from '{source}'")

        if cache_key:
            self._cache[cache_key] = results
            self._save_cache()

        return results

    # ------------------------------------------------------------------
    # KG integration
    # ------------------------------------------------------------------

    def triples_to_relationships(
        self,
        triples: List[Dict[str, Any]],
    ) -> tuple[List[Entity], List[Relationship]]:
        """
        Convert raw triples to KG ``Entity`` and ``Relationship`` objects.

        Unknown predicates are skipped with a warning.
        """
        entities: Dict[str, Entity] = {}
        relationships: List[Relationship] = []

        for t in triples:
            subj = t["subject"]
            obj = t["object"]
            pred_str = t["predicate"]
            conf = t["confidence"]
            evidence = t.get("evidence", "")
            source = t.get("source", "")
            year = t.get("year")

            rel_type = _map_predicate(pred_str)
            if rel_type is None:
                logger.debug(f"Unknown predicate '{pred_str}' — skipping")
                continue

            # Auto-create entities
            for name in (subj, obj):
                if name and name not in entities:
                    entities[name] = Entity(
                        id=name,
                        entity_type=_classify_entity(name),
                        properties={"auto_extracted": True},
                    )

            if not subj or not obj:
                continue

            # Build relationship
            props: Dict[str, Any] = {
                "confidence_score": conf,
                "confidence": "HIGH" if conf >= 0.7 else ("MEDIUM" if conf >= 0.4 else "LOW"),
                "evidence": evidence,
                "provenance": source,
                "sources": [source] if source else [],
                "auto_extracted": True,
            }
            if year:
                props["year"] = year

            relationships.append(Relationship(
                source_id=subj,
                target_id=obj,
                relation_type=rel_type,
                properties=props,
            ))

        return list(entities.values()), relationships

    def merge_triples_into_kg(
        self,
        kg: CausalKnowledgeGraph,
        triples: List[Dict[str, Any]],
    ) -> int:
        """
        Convert *triples* to KG objects and merge into *kg*.

        Returns:
            Number of relationships added or reinforced.
        """
        entities, relationships = self.triples_to_relationships(triples)

        # Build a tiny KG from the new triples and merge it in
        new_kg = CausalKnowledgeGraph()
        for entity in entities:
            new_kg.add_entity(entity)
        for rel in relationships:
            new_kg.add_relationship(rel)

        before = kg.num_relationships
        kg.merge(new_kg)
        delta = kg.num_relationships - before

        logger.info(
            f"Merged {len(relationships)} triples → +{delta} new relationships "
            f"({len(relationships) - delta} reinforced existing)"
        )
        return len(relationships)

    # ------------------------------------------------------------------
    # Main entry point: retrieve → extract → merge
    # ------------------------------------------------------------------

    def enrich(
        self,
        kg: CausalKnowledgeGraph,
        query: str,
        top_k: int = 6,
        filters: Optional[Dict] = None,
        save_path: Optional[str] = None,
    ) -> int:
        """
        Retrieve documents for *query*, extract triples, merge into *kg*.

        This is the "KGs build off one another" entry point:
        - Each call adds new knowledge from retrieved documents
        - Existing edges get confidence reinforced if the new docs support them
        - The KG can be saved and reloaded between sessions

        Args:
            kg: The KG to enrich in-place.
            query: Query to retrieve relevant documents.
            top_k: Number of document chunks to retrieve.
            filters: Optional metadata filters for retrieval.
            save_path: If set, saves the enriched KG here after merging.

        Returns:
            Total number of triples extracted (added + reinforced).
        """
        if self.pipeline is None:
            raise RuntimeError("KGExtractor needs a RAGPipeline for enrich().")

        # Retrieve
        chunks = self.pipeline.retrieve(query, top_k=top_k, filters=filters)
        if not chunks:
            logger.warning(f"No chunks retrieved for '{query}'")
            return 0

        logger.info(f"Enriching KG from {len(chunks)} chunks for query: '{query}'")

        all_triples: List[Dict] = []
        for chunk in chunks:
            text = chunk.get("text", "")
            source = chunk.get("metadata", {}).get(
                "source_file", chunk.get("metadata", {}).get("source", "")
            )
            triples = self.extract_from_text(text, source=source)
            all_triples.extend(triples)

        if not all_triples:
            logger.info("No triples extracted from retrieved chunks.")
            return 0

        n = self.merge_triples_into_kg(kg, all_triples)

        if save_path:
            kg.save(save_path)
            logger.info(f"KG saved to {save_path}  ({kg.num_entities} entities, {kg.num_relationships} rels)")

        return n

    # ------------------------------------------------------------------
    # Batch enrichment across multiple queries
    # ------------------------------------------------------------------

    def enrich_batch(
        self,
        kg: CausalKnowledgeGraph,
        queries: List[str],
        top_k: int = 5,
        save_path: Optional[str] = None,
    ) -> Dict[str, int]:
        """
        Enrich *kg* from a list of queries, saving once at the end.

        Args:
            kg: KG to enrich in-place.
            queries: List of retrieval queries.
            top_k: Chunks per query.
            save_path: Save path after all queries complete.

        Returns:
            ``{"total_triples": N, "queries": len(queries)}``
        """
        total = 0
        for q in queries:
            try:
                total += self.enrich(kg, q, top_k=top_k)
            except Exception as exc:
                logger.warning(f"enrich failed for '{q}': {exc}")

        if save_path:
            kg.save(save_path)
            logger.info(f"Batch enrichment done — {total} triples, KG saved to {save_path}")

        return {"total_triples": total, "queries": len(queries)}
