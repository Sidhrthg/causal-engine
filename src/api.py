"""
FastAPI application for causal modeling engine.

This module provides HTTP endpoints for causal effect estimation and
intervention simulation. It wraps the core estimation and simulation
functions with a REST API interface.
"""

from pathlib import Path
from typing import List, Optional, Tuple, Union
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from src.config import Config
from src.utils.logging_utils import get_logger, setup_logger
from src.ingest import load_dataset
from src.estimate import estimate_from_dag_path
from src.simulate import simulate_from_dag_path, Intervention

# Set up logging
setup_logger(__name__, Config.LOG_LEVEL)
logger = get_logger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Causal Modeling Engine API",
    description="API for causal effect estimation and policy simulation",
    version="0.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request/Response models
class EstimateRequest(BaseModel):
    """Request model for effect estimation."""
    dataset_path: str = Field(..., description="Path to the dataset file (CSV or Parquet)")
    treatment: str = Field(..., description="Name of the treatment variable")
    outcome: str = Field(..., description="Name of the outcome variable")
    controls: list[str] = Field(..., description="List of control/covariate variables")
    dag_path: str = Field(..., description="Path to the DAG file (DOT format)")


class EstimateResponse(BaseModel):
    """Response model for effect estimation."""
    ate: float = Field(..., description="Average Treatment Effect estimate")
    ate_ci: Tuple[float, float] = Field(..., description="Confidence interval as (lower, upper) tuple")
    method: str = Field(..., description="Method used for estimation")


class SimulationRequest(BaseModel):
    """Request model for intervention simulation."""
    dataset_path: str = Field(..., description="Path to the dataset file (CSV or Parquet)")
    treatment: str = Field(..., description="Name of the treatment variable")
    outcome: str = Field(..., description="Name of the outcome variable")
    controls: list[str] = Field(..., description="List of control/covariate variables")
    dag_path: str = Field(..., description="Path to the DAG file (DOT format)")
    node: str = Field(..., description="Name of the node to intervene on")
    value: Union[float, int, str] = Field(..., description="Value to set for the intervention")


class SimulationResponse(BaseModel):
    """Response model for intervention simulation."""
    outcomes: dict[str, float | None] = Field(..., description="Dictionary of outcome metrics; percent_change is null when baseline mean is zero")


@app.get("/")
async def root() -> dict[str, str]:
    """Root endpoint."""
    return {
        "message": "Causal Modeling Engine API",
        "version": "0.1.0",
        "endpoints": ["/estimate", "/simulate"]
    }


@app.get("/health")
async def health() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "healthy"}


@app.post("/estimate", response_model=EstimateResponse)
async def estimate_endpoint(req: EstimateRequest) -> EstimateResponse:
    """
    Estimate Average Treatment Effect (ATE) from a dataset.
    
    This endpoint:
    1) Loads the dataset from req.dataset_path (supports CSV and Parquet)
    2) Calls estimate_from_dag_path from estimate.py
    3) Returns the ATE and confidence interval
    
    Args:
        req: Estimation request with dataset path, treatment, outcome, controls, and DAG path
    
    Returns:
        Estimation results including ATE and confidence interval
    
    Raises:
        HTTPException: If dataset or DAG file cannot be loaded
    """
    logger.info(f"Received estimation request: {req.treatment} -> {req.outcome}")
    logger.info(f"Dataset: {req.dataset_path}, DAG: {req.dag_path}")
    
    try:
        # Load dataset
        logger.info(f"Loading dataset from {req.dataset_path}")
        df = load_dataset(req.dataset_path)
        logger.info(f"Loaded dataset with {len(df)} rows and {len(df.columns)} columns")
        
    except FileNotFoundError as e:
        logger.error(f"Dataset file not found: {req.dataset_path}")
        raise HTTPException(
            status_code=404,
            detail=f"Dataset file not found: {req.dataset_path}. Error: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        raise HTTPException(
            status_code=400,
            detail=f"Failed to load dataset from {req.dataset_path}: {str(e)}"
        )
    
    try:
        # Estimate ATE
        logger.info("Calling estimate_from_dag_path")
        result = estimate_from_dag_path(
            df=df,
            treatment=req.treatment,
            outcome=req.outcome,
            controls=req.controls,
            dag_path=req.dag_path
        )
        
        logger.info(f"Estimation completed: ATE = {result.ate:.6f}")
        
        return EstimateResponse(
            ate=result.ate,
            ate_ci=result.ate_ci,
            method=result.method
        )
    
    except FileNotFoundError as e:
        logger.error(f"DAG file not found: {req.dag_path}")
        raise HTTPException(
            status_code=404,
            detail=f"DAG file not found: {req.dag_path}. Error: {str(e)}"
        )
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(
            status_code=400,
            detail=f"Validation error: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Estimation error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Estimation failed: {str(e)}"
        )


@app.post("/simulate", response_model=SimulationResponse)
async def simulate_endpoint(req: SimulationRequest) -> SimulationResponse:
    """
    Simulate the effects of an intervention on a causal model.
    
    This endpoint:
    1) Loads the dataset from req.dataset_path
    2) Builds an Intervention from req.node and req.value
    3) Calls simulate_from_dag_path
    4) Returns the resulting outcomes
    
    Args:
        req: Simulation request with dataset path, treatment, outcome, controls, DAG path, and intervention details
    
    Returns:
        Simulation results with outcome metrics (baseline, intervened, differences, etc.)
    
    Raises:
        HTTPException: If dataset or DAG file cannot be loaded, or simulation fails
    """
    logger.info(f"Received simulation request: intervention on {req.node} = {req.value}")
    logger.info(f"Dataset: {req.dataset_path}, DAG: {req.dag_path}")
    
    try:
        # Load dataset
        logger.info(f"Loading dataset from {req.dataset_path}")
        df = load_dataset(req.dataset_path)
        logger.info(f"Loaded dataset with {len(df)} rows and {len(df.columns)} columns")
        
    except FileNotFoundError as e:
        logger.error(f"Dataset file not found: {req.dataset_path}")
        raise HTTPException(
            status_code=404,
            detail=f"Dataset file not found: {req.dataset_path}. Error: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        raise HTTPException(
            status_code=400,
            detail=f"Failed to load dataset from {req.dataset_path}: {str(e)}"
        )
    
    try:
        # Build Intervention object
        intervention = Intervention(node=req.node, value=req.value)
        logger.info(f"Created intervention: {intervention.node} = {intervention.value}")
        
        # Run simulation
        logger.info("Calling simulate_from_dag_path")
        result = simulate_from_dag_path(
            df=df,
            treatment=req.treatment,
            outcome=req.outcome,
            controls=req.controls,
            dag_path=req.dag_path,
            intervention=intervention,
            num_samples=1000
        )
        
        logger.info(f"Simulation completed: baseline_mean = {result.outcomes.get('baseline_mean', 'N/A')}")
        
        return SimulationResponse(
            outcomes=result.outcomes
        )
    
    except FileNotFoundError as e:
        logger.error(f"DAG file not found: {req.dag_path}")
        raise HTTPException(
            status_code=404,
            detail=f"DAG file not found: {req.dag_path}. Error: {str(e)}"
        )
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(
            status_code=400,
            detail=f"Validation error: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Simulation error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Simulation failed: {str(e)}"
        )


# ─── Minerals endpoints ───────────────────────────────────────────────────────

_SUPPORTED_COMMODITIES = ["graphite", "lithium", "cobalt", "nickel", "copper"]
_HS_CODES = {
    "graphite": "250490",
    "lithium": "283691",
    "cobalt": "810520",
    "nickel": "750110",
    "copper": "260300",
}


class RouteResult(BaseModel):
    rank: int
    path: List[str]
    bottleneck_t: float
    pct_of_source: float
    is_circumvention: bool
    non_producer_intermediaries: List[str]
    hops: int


class TransshipmentRequest(BaseModel):
    commodity: str = Field("graphite", description="Commodity name")
    source: str = Field("China", description="Dominant exporter / source country")
    destination: str = Field("USA", description="Destination country")
    year: int = Field(2024, description="Trade data year")
    event_years: List[int] = Field([2024], description="Policy event years for circumvention test")
    data_path: str = Field("data/canonical", description="Directory containing cepii_<commodity>.csv files")
    max_hops: int = Field(4, ge=2, le=6, description="Maximum path length")
    nominal_restriction: float = Field(0.30, ge=0.0, le=1.0, description="Fraction of source exports that are restricted")


class TransshipmentResponse(BaseModel):
    commodity: str
    source: str
    destination: str
    year: int
    routes: List[RouteResult]
    circumvention_rate: float
    circumvention_rate_ci: Tuple[float, float]
    nominal_restriction_t: float
    detected_rerouted_t: float
    significant_hubs: List[str]
    notes: List[str]
    summary: str


class QueryRequest(BaseModel):
    question: str = Field(..., description="Question about minerals supply chains")
    commodity: Optional[str] = Field(None, description="Filter context to this commodity")
    top_k: int = Field(5, ge=1, le=20)


class SourceChunk(BaseModel):
    text: str
    source: str
    similarity: float


class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: List[SourceChunk]
    backend: str
    episode_id: str


@app.get("/minerals/commodities")
async def minerals_commodities() -> dict:
    """List supported commodities and their HS codes."""
    return {
        "commodities": _SUPPORTED_COMMODITIES,
        "hs_codes": _HS_CODES,
    }


@app.post("/minerals/transshipment", response_model=TransshipmentResponse)
async def minerals_transshipment(req: TransshipmentRequest) -> TransshipmentResponse:
    """
    Trace trade routes and estimate circumvention for a commodity.

    Loads the pre-processed CEPII BACI canonical CSV
    (``<data_path>/cepii_<commodity>.csv``) and runs TransshipmentAnalyzer.
    """
    try:
        import pandas as pd
        from pathlib import Path
        from src.minerals.transshipment import TransshipmentAnalyzer

        csv_path = Path(req.data_path) / f"cepii_{req.commodity}.csv"
        if not csv_path.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Data file not found: {csv_path}. "
                       f"Expected pre-processed CEPII CSV at this path.",
            )

        df = pd.read_csv(csv_path)
        analyzer = TransshipmentAnalyzer(
            df, commodity=req.commodity, dominant_exporter=req.source
        )

        paths = analyzer.trace_paths(
            source=req.source,
            destination=req.destination,
            year=req.year,
            max_hops=req.max_hops,
        )

        est = analyzer.estimate_circumvention_rate(
            event_years=req.event_years,
            nominal_restriction=req.nominal_restriction,
        )

        summary = analyzer.summary_report(
            destination=req.destination,
            event_years=req.event_years,
            year=req.year,
            max_hops=req.max_hops,
            nominal_restriction=req.nominal_restriction,
        )

        routes = [
            RouteResult(
                rank=i + 1,
                path=p.path,
                bottleneck_t=p.bottleneck_t,
                pct_of_source=p.pct_of_source,
                is_circumvention=p.is_circumvention_candidate,
                non_producer_intermediaries=p.non_producer_intermediaries,
                hops=p.hops,
            )
            for i, p in enumerate(paths)
        ]

        return TransshipmentResponse(
            commodity=req.commodity,
            source=req.source,
            destination=req.destination,
            year=req.year,
            routes=routes,
            circumvention_rate=est.circumvention_rate,
            circumvention_rate_ci=est.circumvention_rate_ci,
            nominal_restriction_t=est.nominal_restriction_t,
            detected_rerouted_t=est.detected_rerouted_t,
            significant_hubs=est.significant_hubs,
            notes=est.notes,
            summary=summary,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Transshipment analysis failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/minerals/query", response_model=QueryResponse)
async def minerals_query(req: QueryRequest) -> QueryResponse:
    """
    Answer a question using the HippoRAG / minerals knowledge base.
    """
    try:
        from src.minerals.rag_pipeline import RAGPipeline

        rag = RAGPipeline()
        result = rag.ask(req.question, top_k=req.top_k)

        sources = [
            SourceChunk(
                text=str(s.get("text", s.get("passage", "")))[:400],
                source=str(s.get("metadata", {}).get("source_file", s.get("metadata", {}).get("source", "unknown"))),
                similarity=float(s.get("similarity", s.get("hybrid_score", s.get("score", 0.0))) or 0.0),
            )
            for s in result.get("sources", [])
        ]

        return QueryResponse(
            question=req.question,
            answer=str(result.get("answer", "")),
            sources=sources,
            backend=str(result.get("backend", "unknown")),
            episode_id=str(result.get("episode_id", "")),
        )

    except Exception as e:
        logger.error(f"RAG query failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ─── Counterfactual endpoints ─────────────────────────────────────────────────

_SCENARIOS_DIR = Path(__file__).parents[1] / "scenarios"
_CALIBRATED_DIR = _SCENARIOS_DIR / "calibrated"


class TrajectoryRow(BaseModel):
    year: int
    P: float
    Q_total: float
    shortage: float
    tight: float
    K: float
    D: float
    Q_sub: float
    Q_fringe: float


class CounterfactualRequest(BaseModel):
    scenario: str = Field(
        "graphite_2023_china_export_controls_substitution",
        description="Scenario name (file in scenarios/ or scenarios/calibrated/)",
    )
    cf_type: str = Field(
        "substitution",
        description="'substitution' | 'fringe' | 'trajectory'",
    )
    # substitution CF
    cf_elasticity: Optional[float] = Field(None, description="Counterfactual substitution elasticity")
    cf_cap: Optional[float] = Field(None, description="Counterfactual substitution cap (0–1)")
    # fringe CF
    cf_capacity_share: Optional[float] = Field(None, description="Counterfactual fringe capacity share")
    cf_entry_price: Optional[float] = Field(None, description="Counterfactual fringe entry price (normalised)")
    # trajectory CF — shock overrides by year
    shock_overrides: Optional[dict] = Field(
        None,
        description="Dict {year_str: {shock_field: value}}, e.g. {'2023': {'export_restriction': 0.0}}",
    )
    use_calibrated: bool = Field(
        True,
        description="Prefer scenarios/calibrated/<name>_calibrated.yaml if it exists",
    )


class CounterfactualResponse(BaseModel):
    scenario: str
    cf_type: str
    description: str
    factual: List[TrajectoryRow]
    counterfactual: List[TrajectoryRow]
    ate: dict  # {outcome: mean(cf) - mean(factual)}
    factual_params: dict  # key params from the scenario
    cf_params: dict       # what was actually changed


def _load_scenario_cfg(name: str, use_calibrated: bool):
    from src.minerals.schema import load_scenario

    # Prefer calibrated copy if requested and available
    if use_calibrated:
        cal = _CALIBRATED_DIR / f"{name}_calibrated.yaml"
        if cal.exists():
            return load_scenario(str(cal))

    base = _SCENARIOS_DIR / f"{name}.yaml"
    if base.exists():
        return load_scenario(str(base))

    raise FileNotFoundError(
        f"Scenario '{name}' not found. "
        f"Available: {[f.stem for f in _SCENARIOS_DIR.glob('*.yaml')]}"
    )


def _df_to_rows(df) -> List[TrajectoryRow]:
    rows = []
    for _, r in df.iterrows():
        rows.append(TrajectoryRow(
            year=int(r["year"]),
            P=float(r.get("P", 0)),
            Q_total=float(r.get("Q_total", 0)),
            shortage=float(r.get("shortage", 0)),
            tight=float(r.get("tight", 0)),
            K=float(r.get("K", 0)),
            D=float(r.get("D", 0)),
            Q_sub=float(r.get("Q_sub", 0)),
            Q_fringe=float(r.get("Q_fringe", 0)),
        ))
    return rows


@app.get("/minerals/scenarios")
async def minerals_scenarios() -> dict:
    """
    List available scenario names and whether a calibrated copy exists.
    """
    scenarios = []
    for f in sorted(_SCENARIOS_DIR.glob("*.yaml")):
        try:
            import yaml as _yaml
            with open(f) as fh:
                meta = _yaml.safe_load(fh)
            cal = _CALIBRATED_DIR / f"{f.stem}_calibrated.yaml"
            scenarios.append({
                "name": f.stem,
                "commodity": meta.get("commodity"),
                "description": meta.get("description") or meta.get("name", f.stem),
                "start_year": meta.get("time", {}).get("start_year"),
                "end_year": meta.get("time", {}).get("end_year"),
                "has_shocks": bool(meta.get("shocks")),
                "calibrated": cal.exists(),
            })
        except Exception:
            continue
    return {"scenarios": scenarios}


@app.post("/minerals/counterfactual", response_model=CounterfactualResponse)
async def minerals_counterfactual(req: CounterfactualRequest) -> CounterfactualResponse:
    """
    Run a Layer-3 counterfactual (Abduction-Action-Prediction).

    cf_type='substitution'  — P(Y | do(substitution_elasticity=cf_elasticity))
                              vs factual trajectory. Answers: "What would prices
                              have been if buyers had/hadn't diversified supply?"

    cf_type='fringe'        — P(Y | do(fringe_capacity_share=cf_capacity_share))
                              Answers: "What if high-cost fringe producers had entered?"

    cf_type='trajectory'    — Shock overrides by year.
                              Answers: "What if the 2023 export restriction had never happened?"

    The noise sequence ε_t is abduced from the factual run (fixed RNG seed),
    ensuring the same exogenous shocks in both worlds — only the structural
    mechanism changes.
    """
    try:
        cfg = _load_scenario_cfg(req.scenario, req.use_calibrated)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    try:
        from src.minerals import pearl_layers as pl

        outcomes = ["P", "Q_total", "shortage", "tight", "K", "D", "Q_sub", "Q_fringe"]

        factual_params = {
            "eta_D": cfg.parameters.eta_D,
            "alpha_P": cfg.parameters.alpha_P,
            "tau_K": cfg.parameters.tau_K,
            "substitution_elasticity": cfg.parameters.substitution_elasticity,
            "fringe_capacity_share": cfg.parameters.fringe_capacity_share,
        }

        # ── substitution CF ──────────────────────────────────────────────────
        if req.cf_type == "substitution":
            if req.cf_elasticity is None:
                raise HTTPException(
                    status_code=422,
                    detail="cf_type='substitution' requires cf_elasticity",
                )
            result = pl.counterfactual_substitution(
                cfg,
                cf_elasticity=req.cf_elasticity,
                cf_cap=req.cf_cap,
            )
            cf_params = {"substitution_elasticity": req.cf_elasticity}
            if req.cf_cap is not None:
                cf_params["substitution_cap"] = req.cf_cap

        # ── fringe CF ────────────────────────────────────────────────────────
        elif req.cf_type == "fringe":
            if req.cf_capacity_share is None:
                raise HTTPException(
                    status_code=422,
                    detail="cf_type='fringe' requires cf_capacity_share",
                )
            result = pl.counterfactual_fringe(
                cfg,
                cf_capacity_share=req.cf_capacity_share,
                cf_entry_price=req.cf_entry_price,
            )
            cf_params = {"fringe_capacity_share": req.cf_capacity_share}
            if req.cf_entry_price is not None:
                cf_params["fringe_entry_price"] = req.cf_entry_price

        # ── trajectory CF ────────────────────────────────────────────────────
        elif req.cf_type == "trajectory":
            if not req.shock_overrides:
                raise HTTPException(
                    status_code=422,
                    detail="cf_type='trajectory' requires shock_overrides dict",
                )
            import numpy as np
            from src.minerals.model import State

            state_0 = State(
                year=cfg.time.start_year,
                t_index=0,
                K=cfg.baseline.K0,
                I=cfg.baseline.I0,
                P=cfg.baseline.P0,
            )
            rng = np.random.default_rng(cfg.seed)
            overrides_by_year = {
                int(yr): vals for yr, vals in req.shock_overrides.items()
            }

            # Run factual
            factual_df, _ = pl._run_scenario_inner(cfg)

            # Run counterfactual trajectory
            from src.minerals.simulate import run_scenario as _run_scenario
            from src.minerals.shocks import shocks_for_year
            from src.minerals.model import step as _step
            import pandas as pd

            rng_cf = np.random.default_rng(cfg.seed)  # same seed = abduction
            s = state_0
            rows = []
            for idx, year in enumerate(cfg.years):
                overrides = overrides_by_year.get(year, {})
                if overrides:
                    s_next, res = pl.counterfactual_step(s, cfg, year, overrides, rng_cf)
                else:
                    shock = shocks_for_year(cfg.shocks, year)
                    s_next, res = _step(cfg, s, shock, rng_cf)
                rows.append({
                    "year": year,
                    "K": s.K, "P": s.P, "D": res.D,
                    "Q_total": res.Q_total, "Q_sub": res.Q_sub,
                    "Q_fringe": res.Q_fringe, "shortage": res.shortage,
                    "tight": res.tight,
                })
                s = s_next
            cf_df = pd.DataFrame(rows)

            ate = pl._compute_ate(factual_df, cf_df, ["P", "Q_total", "shortage", "tight"])
            # Build a pseudo CounterfactualResult
            from src.minerals.pearl_layers import CounterfactualResult
            result = CounterfactualResult(
                factual=factual_df,
                counterfactual=cf_df,
                ate=ate,
                description=f"L3 trajectory counterfactual — shock overrides: {req.shock_overrides}",
                noise_sequence=[],
            )
            cf_params = req.shock_overrides

        else:
            raise HTTPException(
                status_code=422,
                detail=f"Unknown cf_type '{req.cf_type}'. Use: substitution | fringe | trajectory",
            )

        return CounterfactualResponse(
            scenario=req.scenario,
            cf_type=req.cf_type,
            description=result.description,
            factual=_df_to_rows(result.factual),
            counterfactual=_df_to_rows(result.counterfactual),
            ate={k: round(v, 4) for k, v in result.ate.items()},
            factual_params=factual_params,
            cf_params=cf_params,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Counterfactual failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    logger.info(f"Starting Causal Modeling Engine API on {Config.API_HOST}:{Config.API_PORT}")
    uvicorn.run(
        "src.api:app",
        host=Config.API_HOST,
        port=Config.API_PORT,
        reload=Config.API_DEBUG
    )
