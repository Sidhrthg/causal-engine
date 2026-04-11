"""Extract causal relationships from documents using LLM.

Uses the unified LLM backend (LLM_BACKEND: anthropic | openai | vllm | hybrid)
so vLLM can be used for extraction. Retrieval can use HippoRAG when available
(USE_HIPPORAG=1 and index built); otherwise SimpleRAGRetriever.

Also provides utilities for normalizing free-text node names (from LLM extraction)
to canonical DAG variable names, and loading discovered DAG JSON files into
CausalDAG objects usable by CausalInferenceEngine.
"""

import os
import json
import re
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from dataclasses import dataclass

from src.llm.chat import chat_completion, is_chat_available


@dataclass
class CausalEdge:
    """Represents a causal relationship extracted from text."""

    cause: str
    effect: str
    mechanism: str
    confidence: str  # HIGH, MEDIUM, LOW
    evidence: str  # Quote from source
    source_document: str
    validated: bool = False

    def to_dict(self):
        return {
            "cause": self.cause,
            "effect": self.effect,
            "mechanism": self.mechanism,
            "confidence": self.confidence,
            "evidence": self.evidence,
            "source_document": self.source_document,
            "validated": self.validated,
        }


class CausalDiscoveryAgent:
    """
    Extract causal edges from documents using LLM.

    Workflow:
    1. Retrieve relevant documents
    2. Extract causal claims with LLM
    3. Human validates claims
    4. Export to DAG format
    """

    def __init__(self, api_key: str = None, documents_dir: str = "data/documents", use_hipporag: bool = None):
        if not is_chat_available():
            raise ValueError(
                "No LLM backend available. Set ANTHROPIC_API_KEY (anthropic/hybrid), "
                "or LLM_BACKEND=vllm with VLLM_BASE_URL, or OPENAI_API_KEY (openai)."
            )
        self.api_key = api_key
        use_hipporag = use_hipporag if use_hipporag is not None else (os.getenv("USE_HIPPORAG", "1") != "0")
        try:
            from src.minerals.hipporag_retrieval import get_retriever
            self.retriever = get_retriever(use_hipporag=use_hipporag, documents_dir=documents_dir, api_key=api_key)
        except Exception:
            from src.minerals.rag_retrieval import SimpleRAGRetriever
            self.retriever = SimpleRAGRetriever(documents_dir=documents_dir, api_key=api_key)

        if getattr(self.retriever, "chunks", None) is not None and len(self.retriever.chunks) == 0:
            print("📚 Indexing documents...")
            self.retriever.ingest_documents(force_reindex=True, build_embeddings=True)
        elif getattr(self.retriever, "embeddings", None) is None and hasattr(self.retriever, "chunks"):
            embeddings_path = Path(documents_dir) / "embeddings.pkl"
            if embeddings_path.exists():
                with open(embeddings_path, "rb") as f:
                    self.retriever.embeddings = pickle.load(f)

    def extract_causal_edges(
        self,
        domain: str = "graphite supply chain",
        query: str = None,
        top_k_docs: int = 10,
    ) -> List[CausalEdge]:
        """
        Extract causal relationships from document corpus.

        Args:
            domain: Domain context (e.g., "graphite supply chain")
            query: Optional query to focus retrieval
            top_k_docs: Number of documents to analyze

        Returns:
            List of extracted causal edges
        """
        print(f"\n🔬 Extracting causal relationships for: {domain}")

        if query is None:
            query = f"{domain} causal relationships mechanisms"

        print("📖 Retrieving documents...")
        if hasattr(self.retriever, "embeddings") and self.retriever.embeddings is not None:
            docs = self.retriever.retrieve_semantic(query, top_k=top_k_docs)
        else:
            docs = self.retriever.retrieve(query, top_k=top_k_docs)

        print(f"✅ Retrieved {len(docs)} relevant documents\n")

        all_edges = []
        for i, doc in enumerate(docs):
            src = doc.get("metadata", {}).get("source_file", "?")
            print(f"  Analyzing document {i+1}/{len(docs)}: {src}")
            edges = self._extract_from_document(doc, domain)
            all_edges.extend(edges)
            print(f"    → Found {len(edges)} causal edges")

        print(f"\n✅ Extracted {len(all_edges)} total causal edges\n")

        return all_edges

    def _extract_from_document(self, doc: Dict, domain: str) -> List[CausalEdge]:
        """Extract causal edges from a single document."""
        extraction_prompt = f"""You are a causal inference expert analyzing documents about {domain}.

Extract **direct causal relationships** from this text. Focus on:
- Supply-demand mechanisms
- Policy effects
- Price dynamics
- Capacity adjustments
- Trade flows

For each causal relationship, provide:
1. Cause (clear, specific variable or event)
2. Effect (what changes as a result)
3. Mechanism (HOW the cause leads to the effect)
4. Confidence (HIGH/MEDIUM/LOW based on evidence strength)
5. Evidence (exact quote from text supporting this claim)

**Rules:**
- Only extract DIRECT causal claims, not correlations
- Be specific: "export restrictions" not just "policy"
- Quote exact text as evidence
- Rate confidence based on: HIGH = explicit statement, MEDIUM = strong implication, LOW = weak suggestion

Document:
{doc['text']}

Return JSON array of relationships:
```json
[
  {{
    "cause": "export quota reductions",
    "effect": "price increases",
    "mechanism": "reduced supply relative to demand",
    "confidence": "HIGH",
    "evidence": "exact quote from text"
  }}
]
```

JSON:"""

        try:
            text = chat_completion(
                messages=[{"role": "user", "content": extraction_prompt}],
                max_tokens=2000,
            )
            json_match = re.search(r"\[.*\]", text, re.DOTALL)
            if json_match:
                edges_data = json.loads(json_match.group())

                edges = [
                    CausalEdge(
                        cause=edge["cause"],
                        effect=edge["effect"],
                        mechanism=edge["mechanism"],
                        confidence=edge["confidence"],
                        evidence=edge["evidence"],
                        source_document=doc.get("metadata", {}).get("source_file", "?"),
                    )
                    for edge in edges_data
                ]

                return edges
            else:
                return []

        except Exception as e:
            print(f"    ⚠️  Extraction failed: {e}")
            return []

    def validate_edges(self, edges: List[CausalEdge]) -> List[CausalEdge]:
        """Human-in-the-loop validation of extracted edges."""
        print("\n" + "=" * 70)
        print("HUMAN VALIDATION")
        print("=" * 70)
        print("Review each extracted causal relationship.\n")

        validated = []

        for i, edge in enumerate(edges):
            print(f"\n[{i+1}/{len(edges)}] Source: {edge.source_document}")
            print(f"Cause: {edge.cause}")
            print(f"  ↓  ({edge.mechanism})")
            print(f"Effect: {edge.effect}")
            print(f"Confidence: {edge.confidence}")
            evidence_preview = edge.evidence[:150] + "..." if len(edge.evidence) > 150 else edge.evidence
            print(f"Evidence: \"{evidence_preview}\"")

            decision = input("\nAccept? (y/n/skip): ").lower()

            if decision == "y":
                edge.validated = True
                validated.append(edge)
                print("✅ Accepted")
            elif decision == "n":
                print("❌ Rejected")
            else:
                print("⏭️  Skipped")

        print(f"\n✅ Validated {len(validated)}/{len(edges)} edges\n")

        return validated

    def export_to_dag(self, edges: List[CausalEdge], output_path: Path):
        """Export validated edges to DAG format."""
        nodes = set()
        for edge in edges:
            nodes.add(edge.cause)
            nodes.add(edge.effect)

        dag_spec = {
            "nodes": sorted(list(nodes)),
            "edges": [
                {
                    "from": edge.cause,
                    "to": edge.effect,
                    "mechanism": edge.mechanism,
                    "confidence": edge.confidence,
                    "source": edge.source_document,
                    "evidence": edge.evidence,
                    "validated": edge.validated,
                }
                for edge in edges
            ],
            "metadata": {
                "total_edges": len(edges),
                "validated_edges": sum(1 for e in edges if e.validated),
                "sources": list(set(e.source_document for e in edges)),
            },
        }

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(dag_spec, f, indent=2)

        print(f"\n💾 DAG exported to: {output_path}")
        print(f"   Nodes: {len(dag_spec['nodes'])}")
        print(f"   Edges: {len(dag_spec['edges'])}")
        print(f"   Validated: {dag_spec['metadata']['validated_edges']}\n")


# ===========================================================================
# Normalization: free-text node names → canonical DAG variable names
# ===========================================================================

# Ordered rules: each rule is (keywords_any, canonical_name).
# Free-text node names from LLM extraction are matched in order; first wins.
# All matching is case-insensitive substring.
_NORMALIZATION_RULES: List[Tuple[List[str], str]] = [
    # ExportPolicy: any export restriction / quota language
    (["export quota", "export restrict", "export control", "export ban",
      "export licens", "licensing requirement", "trade restrict",
      "trade barrier", "trade sanction", "tariff", "sanction", "embargo"],
     "ExportPolicy"),
    # Capacity: production / environmental compliance
    (["production capacity", "mining capacity", "supply capacity",
      "processing cap", "refin", "mine shutdown",
      "environmental compli", "environmental regulat",
      "compliance standard", "operational cost",
      "capacity cut", "capacity reduction"],
     "Capacity"),
    # Inventory / Stockpile
    (["inventory", "stockpile", "strategic reserve", "buffer stock"],
     "Inventory"),
    # GlobalDemand: macro-level demand drivers
    (["global demand", "macro demand", "gdp", "economic growth",
      "global growth", "financial crisis", "recession",
      "demand shock", "demand destruction"],
     "GlobalDemand"),
    # Demand: sector-level demand
    (["demand", "consumption", "industrial demand", "battery demand",
      "ev demand", "steel demand", "automotive demand"],
     "Demand"),
    # Supply
    (["supply", "production", "output", "extraction", "mining output"],
     "Supply"),
    # Price
    (["price", "spot price", "market price", "price spike", "price increase",
      "price volatil"],
     "Price"),
    # TradeValue
    (["trade value", "import value", "export value", "trade flow",
      "bilateral trade", "import volume", "export volume",
      "natural graphite import"],
     "TradeValue"),
    # Shortage
    (["shortage", "deficit", "market tight", "constrain"],
     "Shortage"),
]


def normalize_to_dag_node(free_text: str) -> Optional[str]:
    """
    Map a free-text node name (as produced by LLM causal extraction) to a
    canonical GraphiteSupplyChainDAG variable name.

    Uses ordered keyword rules — first match wins. Returns None if no rule
    matches, meaning the node has no direct structural model equivalent.

    Examples:
        "China's export quotas reduced by 30-40% compared to 2009 levels"
            → "ExportPolicy"
        "spot price increases of 25-40% above 2007 levels"
            → "Price"
        "limited domestic production capacity"
            → "Capacity"
        "significant market responsiveness"
            → None  (abstract concept, no direct SCM mapping)
    """
    text = free_text.lower()
    for keywords, canonical in _NORMALIZATION_RULES:
        if any(kw in text for kw in keywords):
            return canonical
    return None


def load_dag_from_discovery_json(
    path: str = "dag_registry/discovered_graphite_causal_structure.json",
    min_confidence: str = "MEDIUM",
    normalize_nodes: bool = True,
) -> "CausalDAG":
    """
    Load a discovered causal DAG from the JSON format produced by
    CausalDiscoveryAgent.export_to_dag() and return a CausalDAG ready
    for use with CausalInferenceEngine.

    Args:
        path:            Path to the discovered DAG JSON file.
        min_confidence:  Minimum edge confidence to include ("HIGH", "MEDIUM", "LOW").
        normalize_nodes: If True, map free-text node names to canonical SCM variables
                         using normalize_to_dag_node(). Edges whose source or target
                         cannot be normalized are kept with their original name
                         (they will be treated as unobserved by default).

    Returns:
        CausalDAG with nodes and edges from the discovered structure.
        Nodes whose names appear in the canonical SCM observed set are marked
        observed; all others are marked unobserved.
    """
    from src.minerals.causal_inference import CausalDAG

    _CONFIDENCE_ORDER = {"HIGH": 3, "MEDIUM": 2, "LOW": 1}
    min_level = _CONFIDENCE_ORDER.get(min_confidence.upper(), 2)

    _OBSERVED_CANONICAL = {
        "ExportPolicy", "TradeValue", "Price", "Demand", "GlobalDemand",
    }

    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Discovered DAG JSON not found: {path}")

    with open(path_obj, "r") as f:
        spec = json.load(f)

    dag = CausalDAG()
    node_map: Dict[str, str] = {}  # original name → final DAG node name
    added_nodes: set = set()

    # Process edges
    for edge in spec.get("edges", []):
        confidence = edge.get("confidence", "MEDIUM").upper()
        if _CONFIDENCE_ORDER.get(confidence, 0) < min_level:
            continue

        src_raw = edge.get("from", "")
        tgt_raw = edge.get("to", "")

        if normalize_nodes:
            src = node_map.get(src_raw) or node_map.setdefault(
                src_raw, normalize_to_dag_node(src_raw) or src_raw
            )
            tgt = node_map.get(tgt_raw) or node_map.setdefault(
                tgt_raw, normalize_to_dag_node(tgt_raw) or tgt_raw
            )
        else:
            src = src_raw
            tgt = tgt_raw

        # Skip self-loops that can arise from normalization
        if src == tgt:
            continue

        if src not in added_nodes:
            dag.add_node(src, observed=(src in _OBSERVED_CANONICAL))
            added_nodes.add(src)
        if tgt not in added_nodes:
            dag.add_node(tgt, observed=(tgt in _OBSERVED_CANONICAL))
            added_nodes.add(tgt)

        dag.add_edge(src, tgt)

    return dag


def enrich_dag_from_kg(
    dag: "CausalDAG",
    commodity: str = "graphite",
) -> "CausalDAG":
    """
    Enrich a CausalDAG with additional causal edges from the knowledge graph.

    Adds all CAUSES edges from build_critical_minerals_kg() that involve
    the given commodity or its structural model variables, merging them
    with the existing DAG edges.

    Args:
        dag:       Existing CausalDAG to enrich (modified in place).
        commodity: Commodity filter (default "graphite").

    Returns:
        The enriched CausalDAG (same object, modified in place).
    """
    from src.minerals.knowledge_graph import (
        build_critical_minerals_kg, RelationType, EntityType
    )
    from src.minerals.causal_inference import CausalDAG

    _OBSERVED_CANONICAL = {
        "ExportPolicy", "TradeValue", "Price", "Demand", "GlobalDemand",
    }

    kg = build_critical_minerals_kg()
    added = 0
    for u, v, data in kg._graph.edges(data=True):
        rel = data.get("relationship")
        if rel is None or rel.relation_type != RelationType.CAUSES:
            continue
        # Only include if at least one end involves the commodity or SCM vars
        relevant = (
            commodity in u or commodity in v
            or u in _OBSERVED_CANONICAL or v in _OBSERVED_CANONICAL
            or u in {"Supply", "Shortage", "Inventory", "Capacity",
                     "ExportPolicy", "TradeValue", "Price", "Demand", "GlobalDemand"}
            or v in {"Supply", "Shortage", "Inventory", "Capacity",
                     "ExportPolicy", "TradeValue", "Price", "Demand", "GlobalDemand"}
        )
        if not relevant:
            continue
        if u not in dag.graph:
            dag.add_node(u, observed=(u in _OBSERVED_CANONICAL))
        if v not in dag.graph:
            dag.add_node(v, observed=(v in _OBSERVED_CANONICAL))
        if not dag.graph.has_edge(u, v):
            dag.add_edge(u, v)
            added += 1

    return dag


def main():
    """Demo causal discovery workflow."""
    agent = CausalDiscoveryAgent()

    edges = agent.extract_causal_edges(
        domain="graphite supply chain",
        query="graphite export restrictions price supply demand capacity",
        top_k_docs=5,
    )

    if len(edges) == 0:
        print("⚠️  No causal edges found. Check document content.")
        return

    validated_edges = agent.validate_edges(edges)

    if validated_edges:
        agent.export_to_dag(
            validated_edges,
            Path("dag_registry/discovered_graphite_causal_structure.json"),
        )
    else:
        print("⚠️  No edges validated. Skipping export.")


if __name__ == "__main__":
    main()
