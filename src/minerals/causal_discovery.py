"""Extract causal relationships from documents using LLM."""

import os
import json
import re
import pickle
from pathlib import Path
from typing import List, Dict
from dataclasses import dataclass

try:
    from anthropic import Anthropic
except ImportError:
    Anthropic = None

from src.minerals.rag_retrieval import SimpleRAGRetriever


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

    def __init__(self, api_key: str = None, documents_dir: str = "data/documents"):
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY not set")
        if Anthropic is None:
            raise ImportError("anthropic package required. pip install anthropic")

        self.client = Anthropic(api_key=self.api_key)
        self.retriever = SimpleRAGRetriever(documents_dir=documents_dir, api_key=api_key)

        if len(self.retriever.chunks) == 0:
            print("📚 Indexing documents...")
            self.retriever.ingest_documents(force_reindex=True, build_embeddings=True)
        elif not hasattr(self.retriever, "embeddings") or self.retriever.embeddings is None:
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
            print(f"  Analyzing document {i+1}/{len(docs)}: {doc['metadata']['source_file']}")
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
            response = self.client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=2000,
                messages=[{"role": "user", "content": extraction_prompt}],
            )

            text = response.content[0].text
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
                        source_document=doc["metadata"]["source_file"],
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
