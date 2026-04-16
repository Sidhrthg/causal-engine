"""
End-to-end MineralsCausalPipeline.

Wires all subsystems into a single coherent interface:

    RAG (HippoRAG / Simple)
        ↓  retrieve evidence
    CausalDiscoveryAgent
        ↓  LLM extracts causal edges → CausalDAG
    CausalInferenceEngine
        ↓  L1 association / L2 do() / L3 counterfactual
    SystemDynamicsModel  (run_scenario)
        ↓  simulation with causally-identified parameters
    FittedParameters     (parameter_fitting)

Usage
-----
    from src.pipeline import MineralsCausalPipeline

    pipe = MineralsCausalPipeline()

    # RAG question-answering
    answer = pipe.ask("What caused the 2012 graphite price spike?")
    print(answer["answer"])

    # Causal identification from the hardcoded graphite DAG
    result = pipe.identify("ExportPolicy", "Price")
    print(result.formula)

    # L2 simulation-based do()
    do_result = pipe.do("ExportPolicy", 0.4)   # 40 % export restriction
    print(do_result.effect_on_outcome)

    # L3 counterfactual
    cf = pipe.counterfactual(observed_df, {2012: {"export_restriction": 0.0}})
    print(cf.summary)

    # Fit empirical parameters from CEPII data
    params = pipe.fit_parameters()
    print(params.summary())

    # Run a full scenario (returns DataFrame + metrics dict)
    df, metrics = pipe.run_scenario(cfg)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

from src.minerals.causal_inference import CausalDAG, CommoditySupplyChainDAG, GraphiteSupplyChainDAG
from src.minerals.causal_engine import CausalInferenceEngine
from src.minerals.rag_pipeline import RAGPipeline
from src.minerals.parameter_fitting import FittedParameters, fit_commodity_parameters, fit_graphite_parameters
from src.minerals.transshipment import CircumventionEstimate, PathTrace, TransshipmentAnalyzer
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Discovery result
# ---------------------------------------------------------------------------

@dataclass
class DiscoveryResult:
    """Result of running CausalDiscoveryAgent against the document corpus."""
    dag: CausalDAG
    n_edges: int
    n_nodes: int
    source: str          # "discovered" | "canonical"
    export_path: Optional[str] = None


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class MineralsCausalPipeline:
    """
    Unified interface for the critical-minerals causal engine.

    Parameters
    ----------
    rag_backend : str
        One of ``"auto"``, ``"hipporag"``, ``"industrial"``, ``"simple"``.
        ``"auto"`` picks the best available backend.
    documents_dir : str
        Root directory containing the document corpus.
    cfg : ScenarioConfig | None
        Scenario configuration for simulation-based L2/L3 methods.
        If None, L2 ``do()`` and L3 ``counterfactual()`` raise ValueError
        until you call ``set_scenario(cfg)``.
    dag : CausalDAG | None
        Pre-built causal DAG.  If None, uses the hardcoded
        ``GraphiteSupplyChainDAG`` (correct for graphite analysis).
        Override with ``discover()`` to learn the DAG from documents.
    seed : int
        Random seed for bootstrapping.
    """

    def __init__(
        self,
        rag_backend: str = "auto",
        documents_dir: str = "data/documents",
        cfg=None,
        dag: Optional[CausalDAG] = None,
        commodity: str = "graphite",
        dominant_exporter: str = "China",
        seed: int = 42,
    ):
        self._seed = seed
        self._cfg = cfg
        self._commodity = commodity
        self._dominant_exporter = dominant_exporter
        self._fitted_params: Optional[FittedParameters] = None
        self._transshipment_analyzer: Optional[TransshipmentAnalyzer] = None

        # DAG — commodity-specific by default
        self._dag: CausalDAG = dag or CommoditySupplyChainDAG(
            commodity=commodity,
            dominant_exporter=dominant_exporter,
        )

        # RAG pipeline
        logger.info(f"Initialising RAG pipeline (backend={rag_backend!r})")
        self._rag = RAGPipeline(
            backend=rag_backend,
            documents_dir=documents_dir,
        )

        # Causal inference engine (wired to DAG + cfg)
        self._engine = CausalInferenceEngine(
            dag=self._dag,
            cfg=self._cfg,
            seed=self._seed,
        )

        logger.info(
            f"MineralsCausalPipeline ready "
            f"[commodity={self._commodity}, RAG={self._rag.backend_name}, "
            f"DAG={type(self._dag).__name__}, "
            f"cfg={'set' if self._cfg else 'None'}]"
        )

    # ------------------------------------------------------------------
    # Configuration helpers
    # ------------------------------------------------------------------

    def set_scenario(self, cfg) -> None:
        """Attach a ScenarioConfig for simulation-based L2/L3 methods."""
        self._cfg = cfg
        self._engine = CausalInferenceEngine(
            dag=self._dag, cfg=cfg, seed=self._seed
        )
        logger.info(f"Scenario set: {cfg.name!r}")

    def set_dag(self, dag: CausalDAG) -> None:
        """Replace the causal DAG and rebuild the engine."""
        self._dag = dag
        self._engine = CausalInferenceEngine(
            dag=dag, cfg=self._cfg, seed=self._seed
        )
        logger.info(f"DAG updated: {len(dag.graph.nodes())} nodes, {len(dag.graph.edges())} edges")

    # ------------------------------------------------------------------
    # Layer 0 — RAG question-answering
    # ------------------------------------------------------------------

    def ask(
        self,
        question: str,
        top_k: int = 8,
        filters: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """
        Answer a natural-language question grounded in the document corpus.

        Returns the same dict as ``RAGPipeline.ask()``:
            answer, sources, question, episode_id, n_retrieved, backend, llm_available
        """
        logger.info(f"RAG query: {question!r}")
        return self._rag.ask(question, top_k=top_k, filters=filters)

    def retrieve(self, query: str, top_k: int = 10) -> List[Dict]:
        """Return raw retrieved chunks (no LLM generation)."""
        return self._rag.retrieve(query, top_k=top_k)

    def rag_feedback(self, episode_id: str, rating: int, correction: Optional[str] = None) -> bool:
        """Record feedback for a past ``ask()`` answer."""
        return self._rag.feedback(episode_id, rating, correction)

    # ------------------------------------------------------------------
    # Causal discovery — RAG → DAG
    # ------------------------------------------------------------------

    def discover(
        self,
        domain: str = "graphite supply chain",
        query: Optional[str] = None,
        top_k_docs: int = 10,
        min_confidence: str = "MEDIUM",
        export_path: Optional[str] = None,
        documents_dir: Optional[str] = None,
    ) -> DiscoveryResult:
        """
        Extract a causal DAG from the document corpus using the LLM.

        Runs ``CausalDiscoveryAgent`` against the active retriever, extracts
        causal edges, and wires the resulting ``CausalDAG`` into the engine.

        Requires an LLM backend (ANTHROPIC_API_KEY / OPENAI_API_KEY / vLLM).

        Args:
            domain:         Domain context string for LLM extraction prompt.
            query:          Optional retrieval query (default: domain + "causal relationships").
            top_k_docs:     Number of document chunks to analyse.
            min_confidence: Minimum edge confidence to include in the DAG
                            (``"HIGH"`` | ``"MEDIUM"`` | ``"LOW"``).
            export_path:    If given, write the discovered DAG JSON to this path.
            documents_dir:  Override document directory (default: pipeline's dir).

        Returns:
            DiscoveryResult with the DAG, edge/node counts, and export path.
        """
        from src.minerals.causal_discovery import CausalDiscoveryAgent

        docs_dir = documents_dir or self._rag.documents_dir
        logger.info(f"Starting causal discovery: domain={domain!r}, docs={docs_dir!r}")

        agent = CausalDiscoveryAgent(
            documents_dir=docs_dir,
            use_hipporag=self._rag.backend_name == "hipporag",
        )
        edges = agent.extract_causal_edges(
            domain=domain,
            query=query,
            top_k_docs=top_k_docs,
        )

        # Filter by confidence
        _order = {"HIGH": 3, "MEDIUM": 2, "LOW": 1}
        min_level = _order.get(min_confidence.upper(), 2)
        edges = [e for e in edges if _order.get(e.confidence.upper(), 0) >= min_level]
        logger.info(f"Kept {len(edges)} edges at confidence >= {min_confidence}")

        # Build CausalDAG from extracted edges
        from src.minerals.causal_inference import CausalDAG
        from src.minerals.causal_discovery import normalize_to_dag_node
        _OBSERVED = {
            "ExportPolicy", "TradeValue", "Price", "Demand", "GlobalDemand",
            "SubstitutionSupply", "FringeSupply",
        }
        dag = CausalDAG()
        for edge in edges:
            cause = normalize_to_dag_node(edge.cause) or edge.cause
            effect = normalize_to_dag_node(edge.effect) or edge.effect
            dag.add_node(cause, observed=(cause in _OBSERVED))
            dag.add_node(effect, observed=(effect in _OBSERVED))
            dag.add_edge(cause, effect)

        # Write JSON if requested
        out_path: Optional[str] = None
        if export_path:
            import json
            Path(export_path).parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "edges": [e.to_dict() for e in edges],
                "nodes": list(dag.graph.nodes()),
            }
            Path(export_path).write_text(json.dumps(payload, indent=2))
            out_path = export_path
            logger.info(f"DAG exported to {export_path}")

        # Wire into engine
        self.set_dag(dag)

        return DiscoveryResult(
            dag=dag,
            n_edges=dag.graph.number_of_edges(),
            n_nodes=dag.graph.number_of_nodes(),
            source="discovered",
            export_path=out_path,
        )

    def use_canonical_dag(self) -> None:
        """Reset to the commodity's canonical CommoditySupplyChainDAG."""
        self.set_dag(CommoditySupplyChainDAG(
            commodity=self._commodity,
            dominant_exporter=self._dominant_exporter,
        ))
        logger.info(f"Switched to canonical {self._commodity} DAG")

    # ------------------------------------------------------------------
    # Layer 1 — Association
    # ------------------------------------------------------------------

    def correlate(self, data: "pd.DataFrame", variables: Optional[List[str]] = None):
        """Layer 1: pairwise correlation matrix (associational only)."""
        return self._engine.correlate(data, variables=variables)

    def test_independence(self, data: "pd.DataFrame", x: str, y: str, z: Optional[List[str]] = None):
        """Layer 1: conditional independence test X ⊥ Y | Z."""
        return self._engine.test_independence(data, x, y, z)

    def association(self, data: "pd.DataFrame", outcome: str, predictors: List[str]):
        """Layer 1: OLS regression (associational, NOT causal)."""
        return self._engine.regression_association(data, outcome, predictors)

    # ------------------------------------------------------------------
    # Layer 2 — Intervention
    # ------------------------------------------------------------------

    def identify(self, treatment: str, outcome: str):
        """
        Layer 2: Check if P(outcome | do(treatment)) is identifiable.
        Returns IdentificationResult with strategy, formula, and derivation.
        """
        return self._engine.identify(treatment, outcome)

    def do(self, treatment_var: str, treatment_value: float, outcome_vars: Optional[List[str]] = None):
        """
        Layer 2: Estimate P(Y | do(treatment_var = treatment_value)) by
        running the structural model with an injected shock.

        Requires a ScenarioConfig — call ``set_scenario(cfg)`` first.
        """
        return self._engine.do(treatment_var, treatment_value, outcome_vars=outcome_vars)

    def ate(self, data: "pd.DataFrame", treatment: str, outcome: str):
        """Layer 2: Backdoor-adjusted average treatment effect from observational data."""
        return self._engine.backdoor_estimate(data, treatment, outcome)

    # ------------------------------------------------------------------
    # Layer 3 — Counterfactual
    # ------------------------------------------------------------------

    def counterfactual(
        self,
        observed_data: "pd.DataFrame",
        do_overrides: Dict[int, Dict[str, float]],
    ):
        """
        Layer 3: Full counterfactual via Pearl's Abduction-Action-Prediction.

        Args:
            observed_data: DataFrame from a prior ``run_scenario()`` call.
            do_overrides:  {year: {shock_field: value}} specifying what
                           would have been different.
                           E.g. {2012: {"export_restriction": 0.0}} = "no restriction in 2012".

        Returns CounterfactualResult with factual/cf trajectories and deltas.
        """
        return self._engine.counterfactual(observed_data, do_overrides)

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------

    def run_scenario(self, cfg=None) -> Tuple["pd.DataFrame", Dict[str, float]]:
        """
        Run a full system-dynamics simulation.

        Uses ``cfg`` if supplied, else ``self._cfg``.
        Returns (timeseries_df, metrics_dict).
        """
        from src.minerals.simulate import run_scenario as _run

        scenario = cfg or self._cfg
        if scenario is None:
            raise ValueError(
                "No ScenarioConfig set. Pass cfg= or call set_scenario() first."
            )
        return _run(scenario)

    # ------------------------------------------------------------------
    # Parameter fitting
    # ------------------------------------------------------------------

    def fit_parameters(
        self,
        data_path: Optional[str] = None,
        commodity: Optional[str] = None,
        dominant_exporter: Optional[str] = None,
    ) -> FittedParameters:
        """
        Fit eta_D, alpha_P, tau_K from CEPII-format trade data.

        Defaults to the commodity set at pipeline construction time.
        Caches the result; call again with a different path to refresh.

        Args:
            data_path: Override path to CEPII CSV.
            commodity: Override commodity (default: pipeline's commodity).
            dominant_exporter: Override dominant exporter.

        Returns FittedParameters with empirical estimates and diagnostics.
        """
        if self._fitted_params is None or data_path is not None:
            logger.info("Fitting model parameters from trade data…")
            self._fitted_params = fit_commodity_parameters(
                commodity=commodity or self._commodity,
                data_path=data_path,
                dominant_exporter=dominant_exporter or self._dominant_exporter,
            )
            logger.info(self._fitted_params.summary())
        return self._fitted_params

    def apply_fitted_parameters(self, cfg, data_path: Optional[str] = None):
        """
        Fit parameters and return a new ScenarioConfig with eta_D, alpha_P,
        tau_K replaced by empirically estimated values.

        The existing scenario is left unchanged (Pydantic model_copy).
        """
        params = self.fit_parameters(data_path=data_path)
        fitted = params.as_dict()
        new_params = cfg.parameters.model_copy(update={
            "eta_D": fitted["eta_D"],
            "alpha_P": fitted["alpha_P"],
            "tau_K": fitted["tau_K"],
        })
        return cfg.model_copy(update={"parameters": new_params})

    # ------------------------------------------------------------------
    # Transshipment analysis
    # ------------------------------------------------------------------

    def _get_transshipment_analyzer(
        self,
        data_path: Optional[str] = None,
        known_producers: Optional[set] = None,
    ) -> TransshipmentAnalyzer:
        """
        Build (or return cached) TransshipmentAnalyzer from trade data.

        Requires a CEPII-format CSV (same as parameter fitting).
        Caches the instance; pass ``data_path`` to override.
        """
        if self._transshipment_analyzer is None or data_path is not None:
            from src.minerals.parameter_fitting import _COMMODITY_DEFAULTS
            from pathlib import Path

            defaults = _COMMODITY_DEFAULTS.get(self._commodity, {})
            resolved = Path(data_path) if data_path else Path(defaults.get("data_path", ""))
            if not resolved.exists():
                raise FileNotFoundError(
                    f"Trade data not found at '{resolved}'. "
                    "Pass data_path= to transshipment_analysis()."
                )
            import pandas as pd
            df = pd.read_csv(resolved)
            self._transshipment_analyzer = TransshipmentAnalyzer(
                df=df,
                commodity=self._commodity,
                dominant_exporter=self._dominant_exporter,
                known_producers=known_producers,
            )
        return self._transshipment_analyzer

    def transshipment_analysis(
        self,
        destination: str,
        event_years: List[int],
        data_path: Optional[str] = None,
        year: Optional[int] = None,
        max_hops: int = 4,
        pre_window: int = 3,
        post_window: int = 3,
        significance_level: float = 0.10,
        known_producers: Optional[set] = None,
    ) -> Dict[str, Any]:
        """
        Full transshipment and circumvention analysis.

        Traces multi-hop flows from the dominant exporter to ``destination``,
        detects statistically significant rerouting through non-producer hubs
        after each restriction event, and estimates the effective circumvention rate.

        Args:
            destination:        Final destination country (e.g. "United States").
            event_years:        Years when export restrictions were imposed.
            data_path:          Path to CEPII CSV (uses commodity default if None).
            year:               Year for path tracing (default: last event year).
            max_hops:           Maximum chain length for path tracing.
            pre_window:         Years of pre-event data for rerouting baseline.
            post_window:        Years of post-event data for rerouting test.
            significance_level: p-value threshold for flagging rerouting.
            known_producers:    Override set of known producing countries.

        Returns:
            Dict with keys:
                paths       — list of PathTrace objects (sorted by bottleneck)
                rerouting   — DataFrame of ReroutingSignal records
                circumvention — CircumventionEstimate dataclass
                report      — human-readable summary string
                analyzer    — the TransshipmentAnalyzer instance
        """
        ta = self._get_transshipment_analyzer(data_path=data_path, known_producers=known_producers)

        trace_year = year or (event_years[-1] if event_years else int(ta.df["year"].max()))
        paths = ta.trace_paths(
            self._dominant_exporter, destination, year=trace_year, max_hops=max_hops
        )
        rerouting = ta.detect_rerouting(
            event_years=event_years,
            pre_window=pre_window,
            post_window=post_window,
            significance_level=significance_level,
        )
        circumvention = ta.estimate_circumvention_rate(
            event_years=event_years,
            pre_window=pre_window,
            significance_level=significance_level,
        )
        report = ta.summary_report(
            destination=destination,
            event_years=event_years,
            year=trace_year,
            max_hops=max_hops,
        )

        logger.info(
            f"Transshipment analysis complete: "
            f"{len(paths)} paths {self._dominant_exporter}→{destination}, "
            f"circumvention rate={circumvention.circumvention_rate:.1%}"
        )

        return {
            "paths": paths,
            "rerouting": rerouting,
            "circumvention": circumvention,
            "report": report,
            "analyzer": ta,
        }

    def trace_trade_paths(
        self,
        source: str,
        destination: str,
        year: int,
        data_path: Optional[str] = None,
        max_hops: int = 5,
    ) -> List[PathTrace]:
        """
        Trace all multi-hop paths from ``source`` to ``destination`` in a given year.

        Convenience wrapper around TransshipmentAnalyzer.trace_paths().
        """
        ta = self._get_transshipment_analyzer(data_path=data_path)
        return ta.trace_paths(source, destination, year=year, max_hops=max_hops)

    def corrected_supply_fit(
        self,
        event_years: List[int],
        data_path: Optional[str] = None,
        commodity: Optional[str] = None,
        dominant_exporter: Optional[str] = None,
        circumvention_rate: Optional[float] = None,
    ) -> FittedParameters:
        """
        Fit model parameters using circumvention-corrected supply series.

        This removes the bias in eta_D and tau_K caused by China rerouting
        exports through third-country hubs to evade reported restrictions.

        The corrected supply adds back the estimated rerouted volume to the
        dominant exporter's reported supply before fitting.

        Args:
            event_years:        Restriction event years (used for correction).
            data_path:          Path to CEPII CSV.
            commodity:          Override commodity.
            dominant_exporter:  Override dominant exporter.
            circumvention_rate: Override rate (else estimated from data).

        Returns:
            FittedParameters with corrected point estimates.
        """
        from src.minerals.parameter_fitting import (
            _build_panel, _fit_eta_D, _fit_alpha_P, _fit_tau_K,
            _COMMODITY_DEFAULTS,
        )
        from pathlib import Path

        commodity = commodity or self._commodity
        dominant_exporter = dominant_exporter or self._dominant_exporter
        defaults = _COMMODITY_DEFAULTS.get(commodity, {})
        resolved = Path(data_path) if data_path else Path(defaults.get("data_path", ""))
        if not resolved.exists():
            raise FileNotFoundError(f"Trade data not found at '{resolved}'.")

        import pandas as pd
        df = pd.read_csv(resolved)

        # Get corrected supply series
        ta = TransshipmentAnalyzer(
            df=df, commodity=commodity, dominant_exporter=dominant_exporter
        )
        corrected = ta.corrected_dom_supply(
            event_years=event_years,
            circumvention_rate=circumvention_rate,
        )
        logger.info(
            f"Circumvention correction applied: rate="
            f"{corrected['circumvention_applied'].iloc[0]:.1%}"
        )

        # Replace dominant exporter's quantity_tonnes in the raw df with corrected values
        df_corrected = df.copy()
        year_to_adj = corrected.set_index("year")["rerouted_adjustment_t"].to_dict()
        dom_mask = df_corrected["exporter"] == dominant_exporter

        # Scale each row proportionally to add the correction
        for yr, adj_t in year_to_adj.items():
            if adj_t == 0:
                continue
            yr_dom_mask = dom_mask & (df_corrected["year"] == yr)
            total = df_corrected.loc[yr_dom_mask, "quantity_tonnes"].sum()
            if total > 0:
                scale = (total + adj_t) / total
                df_corrected.loc[yr_dom_mask, "quantity_tonnes"] *= scale

        panel = _build_panel(df_corrected, dominant_exporter=dominant_exporter)
        notes: Dict[str, str] = {
            "correction": (
                f"Circumvention-corrected supply used (events={event_years}, "
                f"rate={corrected['circumvention_applied'].iloc[0]:.1%})"
            )
        }

        import numpy as np
        try:
            eta_D, eta_D_ci, eta_D_se, eta_D_F = _fit_eta_D(panel)
        except Exception as e:
            eta_D, eta_D_ci, eta_D_se, eta_D_F = -0.3, (-0.6, 0.0), 0.15, 0.0
            notes["eta_D"] = f"Fitting failed ({e}); fallback -0.3"

        try:
            alpha_P, alpha_P_ci, alpha_P_se = _fit_alpha_P(panel)
        except Exception as e:
            alpha_P, alpha_P_ci, alpha_P_se = 0.5, (0.2, 0.8), 0.15
            notes["alpha_P"] = f"Fitting failed ({e}); fallback 0.5"

        try:
            tau_K, tau_K_ci, tau_K_se = _fit_tau_K(panel)
        except Exception as e:
            tau_K, tau_K_ci, tau_K_se = 5.0, (3.0, 8.0), 1.0
            notes["tau_K"] = f"Fitting failed ({e}); fallback 5.0"

        return FittedParameters(
            eta_D=eta_D,
            eta_D_ci=eta_D_ci,
            eta_D_se=eta_D_se,
            eta_D_first_stage_F=eta_D_F,
            alpha_P=alpha_P,
            alpha_P_ci=alpha_P_ci,
            alpha_P_se=alpha_P_se,
            tau_K=tau_K,
            tau_K_ci=tau_K_ci,
            tau_K_se=tau_K_se,
            n_obs=len(panel),
            year_range=(int(panel["year"].min()), int(panel["year"].max())),
            notes=notes,
            commodity=commodity,
            dominant_exporter=dominant_exporter,
        )

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def status(self) -> Dict[str, Any]:
        """Return a summary of the pipeline's current state."""
        return {
            "rag_backend": self._rag.backend_name,
            "dag_type": type(self._dag).__name__,
            "dag_nodes": self._dag.graph.number_of_nodes(),
            "dag_edges": self._dag.graph.number_of_edges(),
            "scenario": self._cfg.name if self._cfg else None,
            "fitted_params": self._fitted_params is not None,
            "rag_stats": self._rag.stats(),
        }
