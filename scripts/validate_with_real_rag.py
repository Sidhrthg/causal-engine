"""Model validation with proper RAG - retrieves relevant documents."""

import os
import json
import argparse
import sys
from pathlib import Path
from datetime import datetime

# Ensure project root on path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.llm.chat import chat_completion, is_chat_available
from src.minerals.hipporag_retrieval import get_retriever, hipporag_available
from scripts.validate_with_rag import ModelValidator as BaseValidator


class RAGModelValidator(BaseValidator):
    """Validator with proper document retrieval. Uses LLM_BACKEND. Optional HippoRAG when USE_HIPPORAG=1."""

    def __init__(self, api_key: str = None, documents_dir: str = "data/documents"):
        super().__init__(api_key)
        self._llm_available = is_chat_available()
        # Default to HippoRAG when available and index exists; set USE_HIPPORAG=0 to force classic
        use_hipporag = os.environ.get("USE_HIPPORAG", "1").strip().lower() not in ("0", "false", "no")
        self.retriever = get_retriever(
            use_hipporag=use_hipporag and hipporag_available(),
            documents_dir=documents_dir,
            api_key=api_key,
        )
        self.retriever_backend = type(self.retriever).__name__
        print(f"🔍 Retriever backend: {self.retriever_backend} ({len(self.retriever.chunks)} passages)")
        if len(self.retriever.chunks) == 0 and not (use_hipporag and hipporag_available()):
            print("📚 Indexing documents...")
            if hasattr(self.retriever, "ingest_documents"):
                self.retriever.ingest_documents(force_reindex=True)

    def validate_run(
        self,
        run_dir: str,
        reference_year: int = None,
        comtrade_path: str = "data/canonical/comtrade_graphite_trade.csv",
    ) -> dict:
        """Validate with RAG - retrieves relevant historical documents."""
        run_path = Path(run_dir)
        if not run_path.is_dir():
            raise ValueError(
                f"Run directory is not a valid folder: {run_dir!r}. "
                "Use a path like runs/graphite_baseline/20260120_123456 (from running a scenario)."
            )
        if not (run_path / "timeseries.csv").exists():
            raise ValueError(
                f"No timeseries.csv in {run_dir}. "
                "Run a scenario first (e.g. Run Scenario or Unified Workflow), then use the path it prints."
            )

        print(f"\n🔍 Validating run: {run_dir}")
        print(f"📊 Loading data...\n")

        sim_data = self._load_simulation(run_dir)
        actual_data = self._load_comtrade(comtrade_path)
        comparison = self._compare_data(sim_data, actual_data, reference_year)

        retrieved_docs = self._retrieve_relevant_context(
            sim_data=sim_data,
            reference_year=reference_year,
            comparison=comparison,
        )

        analysis = self._generate_rag_analysis_with_retrieval(
            sim_data=sim_data,
            actual_data=actual_data,
            comparison=comparison,
            retrieved_docs=retrieved_docs,
            reference_year=reference_year,
        )

        report = {
            "run_dir": run_dir,
            "reference_year": reference_year,
            "timestamp": datetime.now().isoformat(),
            "retriever_backend": self.retriever_backend,
            "comparison": comparison,
            "retrieved_documents": [
                {
                    "source": doc["metadata"]["source_file"],
                    "text_preview": doc["text"][:200] + "...",
                }
                for doc in retrieved_docs
            ],
            "llm_analysis": analysis,
            "data_sources": {
                "simulation": str(run_dir),
                "comtrade": comtrade_path,
                "retrieved_docs": len(retrieved_docs),
            },
        }

        output_path = Path(run_dir) / f"validation_rag_report_{reference_year or 'full'}.json"
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)

        print(f"\n💾 Report saved to: {output_path}\n")

        return report

    def _retrieve_relevant_context(
        self,
        sim_data: dict,
        reference_year: int,
        comparison: dict,
    ) -> list:
        """Retrieve relevant documents based on scenario and year (RAG retrieval step)."""
        print("📖 Retrieving relevant historical documents...")

        if reference_year:
            query = f"graphite trade supply shock disruption {reference_year}"
        else:
            query = "graphite trade patterns supply chain disruptions"

        filters = None

        retrieved = self.retriever.retrieve(
            query=query,
            top_k=5,
            filters=filters,
        )

        return retrieved

    def _generate_rag_analysis_with_retrieval(
        self,
        sim_data: dict,
        actual_data,
        comparison: dict,
        retrieved_docs: list,
        reference_year: int = None,
    ) -> str:
        """Generate analysis using RAG - includes retrieved documents in context (augmented generation)."""
        context = self._prepare_context(sim_data, actual_data, comparison)

        if retrieved_docs:
            retrieved_context = "\n\n".join(
                [
                    f"**Document {i+1}**: {doc['metadata']['source_file']}\n{doc['text']}"
                    for i, doc in enumerate(retrieved_docs)
                ]
            )
        else:
            retrieved_context = "No relevant historical documents found."

        kg_context = ""
        try:
            from src.minerals.knowledge_graph import build_critical_minerals_kg
            kg = build_critical_minerals_kg()
            kg_context = kg.summary()
            dag = kg.to_causal_dag()
            edges = list(dag.graph.edges())[:25]
            kg_context += "\n\nKey causal edges (cause → effect): " + ", ".join(f"{u}→{v}" for u, v in sorted(edges))
        except Exception:
            pass

        prompt = f"""You are analyzing a graphite supply chain causal model's predictions against actual data AND historical documents.

## Knowledge Graph Context (supply chain structure):
{kg_context or "Not available."}

## Model Predictions (Simulation):
{context['model_summary']}

## Actual Historical Data (UN Comtrade):
{context['actual_summary']}

## Comparison:
{context['comparison_summary']}

## Retrieved Historical Context (RAG):
{retrieved_context}

## Your Task:
Analyze this validation using the retrieved historical documents to inform your assessment:

1. **Historical Context Validation**: Do the retrieved documents support or contradict the model's mechanisms?

2. **Directional Accuracy**: Compare model predictions to both:
   - Actual trade data
   - Historical accounts of what happened

3. **Magnitude Assessment**: Are discrepancies explained by factors mentioned in historical documents?

4. **Mechanism Refinement**: What causal mechanisms from the documents should be incorporated?

5. **Parameter Calibration**: What parameter values do the historical documents suggest?

Be specific about which retrieved documents inform each conclusion. Quote relevant passages.
"""

        if not self._llm_available:
            return (
                "[LLM analysis skipped] Set LLM_BACKEND and credentials for full RAG analysis.\n"
                "E.g. ANTHROPIC_API_KEY (anthropic) or VLLM_BASE_URL + VLLM_MODEL (vllm).\n\n"
                "Comparison and retrieved documents are still in the report."
            )

        return chat_completion(
            [{"role": "user", "content": prompt}],
            max_tokens=3000,
            api_key=self.api_key,
        )


def main():
    parser = argparse.ArgumentParser(
        description="Validate model with RAG document retrieval",
    )
    parser.add_argument("--run-dir", required=True, help="Path to simulation run directory")
    parser.add_argument("--year", type=int, help="Reference year to focus validation on")
    parser.add_argument(
        "--comtrade",
        default="data/canonical/comtrade_graphite_trade.csv",
        help="Path to Comtrade data",
    )
    parser.add_argument(
        "--docs-dir",
        default="data/documents",
        help="Path to documents directory",
    )
    parser.add_argument("--api-key", help="Anthropic API key")

    args = parser.parse_args()

    validator = RAGModelValidator(
        api_key=args.api_key,
        documents_dir=args.docs_dir,
    )

    report = validator.validate_run(
        run_dir=args.run_dir,
        reference_year=args.year,
        comtrade_path=args.comtrade,
    )

    print("=" * 70)
    print("RAG-ENHANCED VALIDATION ANALYSIS")
    print("=" * 70)
    print(report["llm_analysis"])
    print("\n" + "=" * 70)

    print(f"\n✅ Full report saved")
    print(f"📚 Used {len(report['retrieved_documents'])} retrieved documents\n")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, FileNotFoundError) as e:
        print(f"❌ {e}", file=sys.stderr)
        sys.exit(1)
