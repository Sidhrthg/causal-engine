"""Model validation with proper RAG - retrieves relevant documents."""

import os
import json
import argparse
import sys
from pathlib import Path
from datetime import datetime

# Ensure project root on path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from anthropic import Anthropic
from src.minerals.rag_retrieval import SimpleRAGRetriever
from scripts.validate_with_rag import ModelValidator as BaseValidator


class RAGModelValidator(BaseValidator):
    """Validator with proper document retrieval."""

    def __init__(self, api_key: str = None, documents_dir: str = "data/documents"):
        super().__init__(api_key)

        self.retriever = SimpleRAGRetriever(
            documents_dir=documents_dir,
            api_key=api_key,
        )

        if len(self.retriever.chunks) == 0:
            print("📚 Indexing documents...")
            self.retriever.ingest_documents(force_reindex=True)

    def validate_run(
        self,
        run_dir: str,
        reference_year: int = None,
        comtrade_path: str = "data/canonical/comtrade_graphite_trade.csv",
    ) -> dict:
        """Validate with RAG - retrieves relevant historical documents."""
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

        prompt = f"""You are analyzing a graphite supply chain causal model's predictions against actual data AND historical documents.

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

        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=3000,
            messages=[{"role": "user", "content": prompt}],
        )

        return response.content[0].text


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
    main()
