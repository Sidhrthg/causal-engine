"""
FastAPI backend for Critical Minerals Causal Engine.
Wraps all Gradio functions as REST endpoints for the Next.js/Vercel frontend.

Run with:
    uvicorn api:app --host 0.0.0.0 --port 8000 --reload
"""

import math
import re
import shutil
import tempfile
import time
from pathlib import Path
from typing import Optional

import numpy as np

import matplotlib
matplotlib.use("Agg")

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Import all business logic from app.py.
# Gradio blocks are defined but not launched on import.
import app as engine

app = FastAPI(title="Critical Minerals Causal Engine API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Lock down to your Vercel URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve generated KG scenario PNGs. Mounted under /api/static/* so the Next.js
# /api/:path* rewrite proxies it in production without extra config.
_KG_SCENARIO_DIR = Path("outputs/kg_scenarios")
_KG_SCENARIO_DIR.mkdir(parents=True, exist_ok=True)
(_KG_SCENARIO_DIR / "custom").mkdir(parents=True, exist_ok=True)
app.mount(
    "/api/static/kg_scenarios",
    StaticFiles(directory=str(_KG_SCENARIO_DIR)),
    name="kg_scenarios",
)


# ── Request models ────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str
    commodity: Optional[str] = None
    top_k: int = 5

class TransshipmentRequest(BaseModel):
    commodity: str
    source: str
    destination: str
    year: int
    event_years: list[int] = []
    max_hops: int = 3
    data_path: str = ""
    nominal_restriction: float = 0.3

class CausalAskRequest(BaseModel):
    question: str
    scenario_name: str = ""
    top_k: int = 5

class UnifiedQueryRequest(BaseModel):
    question: str
    top_k: int = 5

class QueryModelRequest(BaseModel):
    query: str

class RAGSearchRequest(BaseModel):
    query: str
    top_k: int = 5
    use_kg_context: bool = False
    use_classic_search_only: bool = False

class RAGAskRequest(BaseModel):
    query: str
    top_k: int = 5

class RAGFeedbackRequest(BaseModel):
    episode_id: str
    rating: float

class RAGEvalRequest(BaseModel):
    n_questions: int = 10
    top_k: int = 5

class KGShockRequest(BaseModel):
    origin_id: str

class KGEnrichRequest(BaseModel):
    query: str
    top_k: int = 5

class KGBatchEnrichRequest(BaseModel):
    top_k: int = 3

class KGDAGRequest(BaseModel):
    simplified: bool = True

class KGRenderScenarioRequest(BaseModel):
    """User-defined scenario for the on-demand KG renderer."""
    year: int
    shock_origin: str
    commodity: str
    title: str
    scenario_id: Optional[str] = None

class ScenarioRequest(BaseModel):
    scenario_name: str

class ValidateRequest(BaseModel):
    run_dir: str
    year: Optional[int] = None

class POmdpRequest(BaseModel):
    data_path: str = ""
    priors_path: str = ""

class POmdpIntegratedRequest(BaseModel):
    data_path: str = ""

class ThreeLayersRequest(BaseModel):
    run_dir: str = ""
    layer: str = "2 — Intervention"
    treatment: str = "ExportPolicy"
    outcome: str = "Price"
    cf_year: str = ""
    cf_value: str = ""

class DoInterventionRequest(BaseModel):
    """
    Layer 2 — Intervention: P(Y | do(parameter=value)).

    Graph surgery on the named structural parameter. Severs whatever normally
    determines that parameter and pins it to the supplied value.

    Fields:
        scenario_name: name of a YAML in scenarios/
        parameter_overrides: dict of {param_name: value} to apply via do(·).
            Supported params (all in ParametersConfig):
              substitution_elasticity, substitution_cap,
              fringe_capacity_share, fringe_entry_price,
              eta_D, alpha_P, tau_K, eta_K
        outcomes: list of output columns to include in comparison table.
            Defaults to ["P", "Q_total", "Q_sub", "Q_fringe", "shortage"].
    """
    scenario_name: str
    parameter_overrides: dict
    outcomes: list[str] = ["P", "Q_total", "Q_sub", "Q_fringe", "shortage"]

class CounterfactualRequest(BaseModel):
    """
    Layer 3 — Counterfactual: P(Y_x | factual trajectory).

    Abduction-Action-Prediction on the structural model.

    Fields:
        scenario_name: name of a YAML in scenarios/ (defines the factual world)
        mechanism: "substitution" or "fringe"
        cf_elasticity: counterfactual substitution_elasticity (mechanism=substitution)
        cf_cap: counterfactual substitution_cap (optional)
        cf_capacity_share: counterfactual fringe_capacity_share (mechanism=fringe)
        cf_entry_price: counterfactual fringe_entry_price (optional)
    """
    scenario_name: str
    mechanism: str  # "substitution" or "fringe"
    cf_elasticity: Optional[float] = None
    cf_cap: Optional[float] = None
    cf_capacity_share: Optional[float] = None
    cf_entry_price: Optional[float] = None

class ExportReportRequest(BaseModel):
    content: str

class UnifiedWorkflowRequest(BaseModel):
    domain: str = "Mineral supply chain"
    scenario_name: str = ""
    sensor_data: str = ""
    sensor_priors: str = ""


# ── Helpers ───────────────────────────────────────────────────────────────────

def _image_response(path: str | None) -> FileResponse:
    if not path or not Path(path).exists():
        raise HTTPException(status_code=404, detail="Image not found. Build the KG or run a scenario first.")
    return FileResponse(path, media_type="image/png")


def _extract_shock_sources(gradio_update) -> list[str]:
    """Pull choices out of a gr.update(...) return value."""
    if isinstance(gradio_update, dict):
        return gradio_update.get("choices", [])
    # Gradio 4+ returns a dataclass-like object
    if hasattr(gradio_update, "choices"):
        return gradio_update.choices or []
    return engine.get_kg_shock_sources()


# ── Health ───────────────────────────────────────────────────────────────────

@app.get("/api/health")
def health():
    return {"status": "healthy"}


# ── Commodities ───────────────────────────────────────────────────────────────

@app.get("/api/commodities")
def get_commodities():
    return {
        "commodities": ["graphite", "lithium", "cobalt", "nickel", "copper", "soybeans"],
        "hs_codes": {
            "graphite": "HS 250490",
            "lithium":  "HS 283691",
            "cobalt":   "HS 810520",
            "nickel":   "HS 750110",
            "copper":   "HS 740311",
            "soybeans": "HS 120190",
        },
    }


# ── Scenarios ─────────────────────────────────────────────────────────────────

@app.get("/api/scenarios")
def get_scenarios():
    return {"scenarios": engine.list_scenarios()}


# ── Knowledge Query (frontend shorthand) ─────────────────────────────────────

@app.post("/api/query")
def knowledge_query(req: QueryRequest):
    """
    POST /api/query — used by the Next.js Knowledge Query page.

    Wraps the full RAG pipeline (retrieve + LLM answer + memory) and returns
    a structured response with sources the frontend can render.
    """
    try:
        pipeline = engine._get_pipeline()
        q = req.question.strip()
        if not q:
            raise HTTPException(status_code=400, detail="question is required")

        result = pipeline.ask(q, top_k=req.top_k, use_memory=True)

        raw_sources = result.get("sources", [])
        sources = []
        for s in raw_sources[:10]:
            meta = s.get("metadata", {}) if isinstance(s.get("metadata"), dict) else {}
            sources.append({
                "text": (s.get("text") or "")[:600],
                "source": meta.get("source_file", s.get("source", "unknown")),
                "similarity": round(float(s.get("similarity", 0.0)), 4),
            })

        return {
            "question": req.question,
            "answer": result.get("answer", "No answer returned."),
            "sources": sources,
            "backend": getattr(pipeline, "backend_name", "rag"),
            "episode_id": result.get("episode_id", ""),
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ── Transshipment analysis (frontend shorthand) ───────────────────────────────

@app.post("/api/transshipment")
def transshipment_analysis(req: TransshipmentRequest):
    """
    POST /api/transshipment — used by the Next.js Transshipment page.

    Loads the CEPII canonical file for the requested commodity, runs the
    TransshipmentAnalyzer, and returns ranked routes + circumvention estimate.
    """
    import pandas as pd
    from src.minerals.transshipment import TransshipmentAnalyzer

    # Resolve data path
    commodity_files = {
        "graphite": "data/canonical/cepii_graphite.csv",
        "lithium":  "data/canonical/cepii_lithium.csv",
        "cobalt":   "data/canonical/cepii_cobalt.csv",
        "nickel":   "data/canonical/cepii_nickel.csv",
        "copper":   "data/canonical/cepii_copper.csv",
        "soybeans": "data/canonical/cepii_soybeans.csv",
    }
    data_path = req.data_path or commodity_files.get(req.commodity.lower(), "")
    if not data_path or not Path(data_path).exists():
        raise HTTPException(
            status_code=404,
            detail=f"No CEPII data file found for commodity '{req.commodity}'. "
                   f"Expected at {data_path or commodity_files.get(req.commodity.lower(), '?')}",
        )

    dominant_exporters = {
        "graphite": "China", "cobalt": "Democratic Republic of the Congo",
        "lithium": "Australia", "nickel": "Indonesia",
        "copper": "Chile", "soybeans": "USA",
    }
    dominant = dominant_exporters.get(req.commodity.lower(), req.source)

    df = pd.read_csv(data_path)
    ta = TransshipmentAnalyzer(df, commodity=req.commodity.lower(), dominant_exporter=dominant)

    # Trace routes from source to destination for the requested year
    try:
        paths = ta.trace_paths(req.source, req.destination, year=req.year, max_hops=req.max_hops)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"trace_paths failed: {exc}")

    routes = []
    for i, p in enumerate(paths[:20], 1):
        routes.append({
            "rank": i,
            "path": p.path,
            "bottleneck_t": round(p.bottleneck_t, 1),
            "pct_of_source": round(p.pct_of_source, 4),
            "is_circumvention": p.is_circumvention_candidate,
            "non_producer_intermediaries": p.non_producer_intermediaries,
            "hops": p.hops,
        })

    # Circumvention estimate
    event_years = req.event_years or []
    circ_rate = 0.0
    circ_ci: list[float] = [0.0, 0.0]
    sig_hubs: list[str] = []
    notes: list[str] = []
    nominal_t = 0.0
    rerouted_t = 0.0

    if event_years:
        try:
            # Nominal restriction tonnage: source's total exports * nominal_restriction
            yr_df = df[df["year"] == req.year]
            src_df = yr_df[yr_df["exporter"] == req.source] if req.source in df["exporter"].values else yr_df[yr_df["exporter"].str.contains(req.source, case=False, na=False)]
            nominal_t = float(src_df["quantity_tonnes"].sum()) * req.nominal_restriction
            est = ta.estimate_circumvention_rate(event_years=event_years, nominal_restriction=nominal_t)
            circ_rate = round(est.circumvention_rate, 4)
            circ_ci = [round(est.circumvention_rate_ci[0], 4), round(est.circumvention_rate_ci[1], 4)]
            sig_hubs = est.significant_hubs
            notes = est.notes
            rerouted_t = round(est.detected_rerouted_t, 1)
        except Exception as exc:
            notes = [f"Circumvention estimation skipped: {exc}"]

    n_circ = sum(1 for r in routes if r["is_circumvention"])
    summary = (
        f"{len(routes)} routes found from {req.source} to {req.destination} "
        f"({req.year}). {n_circ} pass through non-producer intermediaries."
    )

    return {
        "commodity": req.commodity,
        "source": req.source,
        "destination": req.destination,
        "year": req.year,
        "routes": routes,
        "circumvention_rate": circ_rate,
        "circumvention_rate_ci": circ_ci,
        "nominal_restriction_t": round(nominal_t, 1),
        "detected_rerouted_t": rerouted_t,
        "significant_hubs": sig_hubs,
        "notes": notes,
        "summary": summary,
    }


# ── Causal Ask ────────────────────────────────────────────────────────────────

@app.post("/api/causal/ask")
def causal_ask(req: CausalAskRequest):
    return {"result": engine.causal_ask(req.question, req.scenario_name, req.top_k)}


# ── Ask a Question ────────────────────────────────────────────────────────────

@app.post("/api/query/unified")
def unified_query(req: UnifiedQueryRequest):
    return {"result": engine.unified_query(req.question, req.top_k)}

@app.post("/api/query/model")
def query_model(req: QueryModelRequest):
    return {"result": engine.query_model(req.query)}

@app.post("/api/export-report")
def export_report(req: ExportReportRequest):
    path = engine.export_report(req.content)
    if not path:
        raise HTTPException(status_code=400, detail="No content to export.")
    return FileResponse(path, media_type="text/markdown", filename=Path(path).name)


# ── Run Scenario ──────────────────────────────────────────────────────────────

@app.post("/api/scenario/run")
def run_scenario(req: ScenarioRequest):
    output, run_dir = engine.run_scenario_tab(req.scenario_name)
    return {"output": output, "run_dir": run_dir}

@app.post("/api/scenario/run-causal")
def run_scenario_causal(req: ScenarioRequest):
    return {"result": engine.run_scenario_causal(req.scenario_name)}


# ── Search Documents (RAG) ────────────────────────────────────────────────────

@app.post("/api/rag/search")
def rag_search(req: RAGSearchRequest):
    return {"result": engine.rag_search(req.query, req.top_k, req.use_kg_context, req.use_classic_search_only)}

@app.post("/api/rag/ask")
def rag_ask(req: RAGAskRequest):
    answer, episode_id = engine.rag_ask(req.query, req.top_k)
    return {"answer": answer, "episode_id": episode_id}

@app.post("/api/rag/feedback")
def rag_feedback(req: RAGFeedbackRequest):
    return {"result": engine.rag_feedback(req.episode_id, req.rating)}

@app.post("/api/rag/eval")
def run_rag_eval(req: RAGEvalRequest):
    return {"result": engine.run_rag_eval(req.n_questions, req.top_k)}

@app.get("/api/rag/memory-stats")
def rag_memory_stats():
    return {"result": engine.rag_memory_stats()}

@app.post("/api/rag/reindex")
def reindex_rag():
    return {"result": engine.reindex_rag()}

@app.post("/api/rag/build-hipporag")
def build_hipporag():
    return {"result": engine.build_hipporag_index()}

@app.post("/api/rag/upload")
async def upload_documents(files: list[UploadFile] = File(...)):
    """Save uploaded files to a temp dir, then pass paths into the engine."""
    tmp_dir = Path(tempfile.mkdtemp())
    try:
        paths = []
        for f in files:
            dest = tmp_dir / (f.filename or "upload")
            with dest.open("wb") as out:
                shutil.copyfileobj(f.file, out)
            paths.append({"name": str(dest)})
        result = engine.save_uploaded_documents(paths)
        return {"result": result}
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ── Causal Analysis & DAG ─────────────────────────────────────────────────────

@app.post("/api/causal/analysis")
def causal_analysis(req: ScenarioRequest):
    return {"result": engine.show_causal_analysis(req.scenario_name)}

@app.get("/api/causal/dag-image")
def dag_image():
    return _image_response(engine.get_dag_image_path())

@app.post("/api/causal/dag-image/refresh")
def refresh_dag_image():
    return _image_response(engine.generate_dag_image())


# ── Validate ──────────────────────────────────────────────────────────────────

@app.post("/api/validate")
def validate(req: ValidateRequest):
    return {"result": engine.validate_with_rag(req.run_dir, req.year)}


@app.get("/api/validate/predictability")
def validate_predictability():
    """
    Run the causal engine predictability evaluation across all CEPII episodes.

    Returns structured metrics per episode:
    - directional_accuracy: fraction of year-on-year price moves predicted correctly
    - spearman_rho: rank correlation of price index trajectory
    - log_price_rmse: RMSE of log-price index (scale-invariant)
    - magnitude_ratio: median |model %Δ| / |CEPII %Δ| (1.0 = perfect)
    - grade: A/B/C/F composite score
    - known_gap: documented structural limitation for failing episodes
    """
    from src.minerals.predictability import run_predictability_evaluation
    results = run_predictability_evaluation()
    return {
        "episodes": [r.to_dict() for r in results],
        "summary": {
            "n_episodes": len(results),
            "grades": {r.name: r.grade for r in results},
            "mean_directional_accuracy": round(
                float(np.mean([r.directional_accuracy for r in results
                                if not math.isnan(r.directional_accuracy)])), 3
            ),
            "mean_spearman_rho": round(
                float(np.mean([r.spearman_rho for r in results
                                if not math.isnan(r.spearman_rho)])), 3
            ),
        },
    }


@app.get("/api/validate/oos")
def validate_oos():
    """
    Out-of-sample validation: structural parameters calibrated on one episode
    are applied to a different episode. Shocks remain episode-specific.

    Tests whether the causal mechanism generalises across time periods without
    re-fitting parameters.

    Returns per-episode OOS DA alongside in-sample DA, plus a summary showing
    how much accuracy degrades when parameters are transferred.
    """
    from src.minerals.predictability import run_oos_evaluation
    return run_oos_evaluation()


@app.get("/api/validate/counterfactual")
def validate_counterfactual():
    """
    Pearl L3 counterfactual analysis.

    For three key episodes, runs the model twice:
      - actual:         the real shocks that occurred
      - counterfactual: the key policy intervention removed (do(shock=0))

    The price difference between trajectories is the causal effect of the
    intervention — a Pearl Layer 3 query that purely statistical models cannot
    answer.  Episodes covered:
      1. Graphite 2023 export controls: price impact of China's Oct-2023 ban
      2. Soybeans 2018 trade war:       price impact of US-China tariffs
      3. Graphite 2008 export quota:    price impact of China's 2010-2011 quota
    """
    from src.minerals.predictability import run_counterfactual_analysis
    return {"counterfactuals": run_counterfactual_analysis()}


@app.get("/api/validate/sensitivity")
def validate_sensitivity():
    """
    Return pre-computed fixed-parameter sensitivity grid results.

    Grid: cover_star × u0 × lambda_cover (48 points).
    Calibrated parameters (alpha_P, eta_D, tau_K, g) held fixed at
    episode-specific values. Reports per-point DA for 8 in-sample
    episodes and 3 OOS transfers, plus summary statistics.
    """
    import json
    grid_path = Path("data/sensitivity_grid.json")
    if not grid_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Sensitivity grid not yet computed. Run: python scripts/sensitivity_grid.py",
        )
    with open(grid_path) as f:
        rows = json.load(f)

    in_sample_keys = [
        "graphite_2008", "graphite_2022", "lithium_2022",
        "soybeans_2011", "soybeans_2015", "soybeans_2018",
        "soybeans_2020", "soybeans_2022",
    ]
    oos_keys = ["oos_graphite_2022", "oos_graphite_2008", "oos_soybeans_2022"]

    def _safe(v):
        return None if (v is None or (isinstance(v, float) and v != v)) else round(v, 3)

    summaries = []
    for row in rows:
        is_vals = [row[k] for k in in_sample_keys if isinstance(row.get(k), float) and row[k] == row[k]]
        oos_vals = [row[k] for k in oos_keys if isinstance(row.get(k), float) and row[k] == row[k]]
        summaries.append({
            "cover_star": row["cover_star"],
            "u0": row["u0"],
            "lambda_cover": row["lambda_cover"],
            "mean_is_da": _safe(sum(is_vals)/len(is_vals) if is_vals else None),
            "mean_oos_da": _safe(sum(oos_vals)/len(oos_vals) if oos_vals else None),
            **{k: _safe(row.get(k)) for k in in_sample_keys + oos_keys},
        })

    return {
        "n_grid_points": len(rows),
        "grid_axes": {"cover_star": [0.10,0.15,0.20,0.25], "u0": [0.85,0.90,0.92,0.95], "lambda_cover": [0.40,0.60,0.80]},
        "results": summaries,
    }


@app.get("/api/validate/extractor")
def validate_extractor():
    """
    Evaluate the rule-based EventShockMapper against a 20-example gold standard.

    Metrics:
    - commodity F1: does the extractor identify the right commodity?
    - type F1:      does the extractor identify the right shock type?
    - direction accuracy: among type-correct extractions, is the sign correct?

    Gold standard: hand-labeled headlines covering graphite, cobalt, lithium,
    nickel, and soybeans (17 positive + 3 true-negative examples).
    """
    from src.minerals.extractor_eval import run_extractor_eval
    return run_extractor_eval().to_dict()


@app.get("/api/validate/baseline-comparison")
def validate_baseline_comparison():
    """
    Compare causal engine directional accuracy against four statistical baselines.

    Baselines (receive no shock information):
    - Random Walk:    predict no change (P_{t+1} = P_t)
    - Momentum:       predict same direction as previous year-on-year move
    - AR(1):          fit on pre-episode history, predict forward
    - Mean Reversion: predict price moves toward long-run pre-episode mean

    Returns per-episode results plus a summary table showing causal engine
    improvement over the best statistical baseline.
    """
    from src.minerals.baseline_comparison import run_baseline_comparison, summary_stats
    from src.minerals.predictability import run_predictability_evaluation

    causal_results = run_predictability_evaluation()
    results = run_baseline_comparison(causal_results=causal_results)
    return {
        "episodes": [r.to_dict() for r in results],
        "summary": summary_stats(results),
    }


@app.get("/api/validate/historical")
def validate_historical():
    """
    Run all three historical CEPII validation episodes and return structured
    pass/fail results with model vs CEPII comparison metrics.

    Episodes
    --------
    - graphite_2008: 2008 demand spike + 2009 GFC + 2010-11 China quota
    - graphite_2023: 2022 EV surge + Oct 2023 export controls
    - lithium_2022:  2022 EV demand boom + 2024 price correction

    Returns
    -------
    JSON with ``episodes`` list.  Each episode has:
    - ``name``, ``commodity``, ``status`` (pass/fail/error)
    - ``checks``: list of {name, passed, model_value, cepii_value, notes}
    """
    import pandas as pd
    from src.minerals.schema import (
        BaselineConfig, DemandGrowthConfig, OutputsConfig,
        ParametersConfig, PolicyConfig, ScenarioConfig,
        ShockConfig, TimeConfig, load_scenario,
    )
    from src.minerals.simulate import run_scenario as _run

    DATA_PATH_GRAPHITE = "data/canonical/cepii_graphite.csv"
    DATA_PATH_LITHIUM  = "data/canonical/cepii_lithium.csv"

    def _cepii_china(path=DATA_PATH_GRAPHITE):
        df = pd.read_csv(path)
        exporter = "China" if "graphite" in path else "Chile"
        g = (
            df[df["exporter"] == exporter]
            .groupby("year")
            .agg(value_kusd=("value_kusd", "sum"), qty_tonnes=("quantity_tonnes", "sum"))
            .reset_index()
        )
        g["implied_price"] = g["value_kusd"] / g["qty_tonnes"]
        return g.set_index("year")

    def _pct(a, b):
        return (b - a) / max(abs(a), 1e-9)

    def _check(name, passed, model_val, cepii_val, notes=""):
        return {
            "name": name,
            "passed": bool(passed),
            "model_value": round(float(model_val), 4) if model_val is not None else None,
            "cepii_value": round(float(cepii_val), 4) if cepii_val is not None else None,
            "notes": notes,
        }

    episodes = []

    # ── Episode 1+2: graphite 2008 ────────────────────────────────────────────
    try:
        cfg = load_scenario("scenarios/graphite_2008_calibrated.yaml")
        df, _ = _run(cfg)
        m = df.set_index("year")
        cg = _cepii_china(DATA_PATH_GRAPHITE)

        checks = [
            _check(
                "price_rises_2008",
                m.loc[2009, "P"] > m.loc[2007, "P"] * 1.20,
                _pct(m.loc[2007, "P"], m.loc[2009, "P"]),
                _pct(cg.loc[2007, "implied_price"], cg.loc[2008, "implied_price"]),
                "Model P_2009 vs P_2007 (lag-adjusted); CEPII 2007→2008",
            ),
            _check(
                "shortage_positive_2008",
                m.loc[2008, "shortage"] > 0,
                float(m.loc[2008, "shortage"]),
                None,
            ),
            _check(
                "demand_falls_2009_gfc",
                m.loc[2009, "D"] < m.loc[2008, "D"],
                float(m.loc[2009, "D"]),
                _pct(cg.loc[2008, "qty_tonnes"], cg.loc[2009, "qty_tonnes"]),
                "Model D_2009 < D_2008; CEPII vol pct change",
            ),
            _check(
                "quota_reduces_qeff_2010",
                m.loc[2010, "Q_eff"] < m.loc[2009, "Q_eff"],
                float(m.loc[2010, "Q_eff"]),
                float(cg.loc[2010, "qty_tonnes"]),
            ),
            _check(
                "price_rises_2010_2011",
                m.loc[2011, "P"] > m.loc[2010, "P"],
                _pct(m.loc[2010, "P"], m.loc[2011, "P"]),
                _pct(cg.loc[2010, "implied_price"], cg.loc[2011, "implied_price"]),
            ),
        ]
        n_pass = sum(c["passed"] for c in checks)
        episodes.append({
            "name": "graphite_2008_demand_spike_and_quota",
            "commodity": "graphite",
            "status": "pass" if n_pass == len(checks) else f"partial ({n_pass}/{len(checks)})",
            "checks": checks,
        })
    except Exception as exc:
        episodes.append({"name": "graphite_2008_demand_spike_and_quota", "status": "error", "error": str(exc)})

    # ── Episode 3: graphite 2022-23 ───────────────────────────────────────────
    try:
        cfg3 = ScenarioConfig(
            name="ep3_api", commodity="graphite", seed=42,
            time=TimeConfig(dt=1.0, start_year=2019, end_year=2025),
            baseline=BaselineConfig(P_ref=1.0, P0=1.0, K0=108.695652, I0=20.0, D0=100.0),
            parameters=ParametersConfig(
                eps=1e-9, u0=0.92, beta_u=0.10, u_min=0.70, u_max=1.00,
                tau_K=3.0, eta_K=0.40, retire_rate=0.0, eta_D=-0.25,
                demand_growth=DemandGrowthConfig(type="constant", g=1.0),
                alpha_P=0.80, cover_star=0.20, lambda_cover=0.60, sigma_P=0.0,
            ),
            policy=PolicyConfig(),
            shocks=[
                ShockConfig(type="demand_surge",       start_year=2022, end_year=2022, magnitude=0.10),
                ShockConfig(type="export_restriction", start_year=2023, end_year=2024, magnitude=0.35),
            ],
            outputs=OutputsConfig(metrics=["total_shortage", "peak_shortage", "avg_price", "final_inventory_cover"]),
        )
        df3, _ = _run(cfg3)
        m3 = df3.set_index("year")
        cg3 = _cepii_china(DATA_PATH_GRAPHITE)

        checks3 = [
            _check(
                "price_rises_after_2022_surge",
                m3.loc[2023, "P"] > m3.loc[2021, "P"],
                _pct(m3.loc[2021, "P"], m3.loc[2023, "P"]),
                _pct(cg3.loc[2021, "implied_price"], cg3.loc[2022, "implied_price"]),
                "Model 2021→2023 (lag-adjusted); CEPII 2021→2022",
            ),
            _check(
                "qeff_drops_under_export_controls_2023",
                m3.loc[2023, "Q_eff"] < m3.loc[2022, "Q_eff"],
                float(m3.loc[2023, "Q_eff"]),
                float(cg3.loc[2023, "qty_tonnes"]),
            ),
            _check(
                "shortage_under_export_controls_2023",
                m3.loc[2023, "shortage"] > 0,
                float(m3.loc[2023, "shortage"]),
                None,
            ),
        ]
        n3 = sum(c["passed"] for c in checks3)
        episodes.append({
            "name": "graphite_2022_ev_surge_and_export_controls",
            "commodity": "graphite",
            "status": "pass" if n3 == len(checks3) else f"partial ({n3}/{len(checks3)})",
            "checks": checks3,
        })
    except Exception as exc:
        episodes.append({"name": "graphite_2022_ev_surge_and_export_controls", "status": "error", "error": str(exc)})

    # ── Lithium 2022 ──────────────────────────────────────────────────────────
    try:
        cfg_li = ScenarioConfig(
            name="lithium_2022_api", commodity="lithium", seed=42,
            time=TimeConfig(dt=1.0, start_year=2019, end_year=2025),
            baseline=BaselineConfig(P_ref=1.0, P0=1.0, K0=108.695652, I0=20.0, D0=100.0),
            parameters=ParametersConfig(
                eps=1e-9, u0=0.92, beta_u=0.10, u_min=0.70, u_max=1.00,
                tau_K=3.0, eta_K=0.40, retire_rate=0.0, eta_D=-0.25,
                demand_growth=DemandGrowthConfig(type="constant", g=1.0),
                alpha_P=0.80, cover_star=0.20, lambda_cover=0.60, sigma_P=0.0,
            ),
            policy=PolicyConfig(),
            shocks=[ShockConfig(type="demand_surge", start_year=2022, end_year=2022, magnitude=0.30)],
            outputs=OutputsConfig(metrics=["total_shortage", "peak_shortage", "avg_price", "final_inventory_cover"]),
        )
        df_li, _ = _run(cfg_li)
        m_li = df_li.set_index("year")
        cl = _cepii_china(DATA_PATH_LITHIUM)

        checks_li = [
            _check(
                "cepii_price_surges_2022",
                cl.loc[2022, "implied_price"] > cl.loc[2021, "implied_price"] * 3.0,
                None,
                _pct(cl.loc[2021, "implied_price"], cl.loc[2022, "implied_price"]),
                "Ground-truth check: CEPII Chile lithium price >200 % in 2022",
            ),
            _check(
                "model_price_rises_after_surge",
                m_li.loc[2023, "P"] > m_li.loc[2021, "P"],
                _pct(m_li.loc[2021, "P"], m_li.loc[2023, "P"]),
                _pct(cl.loc[2021, "implied_price"], cl.loc[2022, "implied_price"]),
                "Model 2021→2023 (lag-adjusted); CEPII 2021→2022",
            ),
            _check(
                "demand_cools_after_surge",
                m_li.loc[2023, "D"] < m_li.loc[2022, "D"],
                float(m_li.loc[2023, "D"]),
                float(m_li.loc[2022, "D"]),
                "demand_reversion_rate=0.60 → D falls from peak in 2023",
            ),
        ]
        n_li = sum(c["passed"] for c in checks_li)
        episodes.append({
            "name": "lithium_2022_ev_boom",
            "commodity": "lithium",
            "status": "pass" if n_li == len(checks_li) else f"partial ({n_li}/{len(checks_li)})",
            "checks": checks_li,
        })
    except Exception as exc:
        episodes.append({"name": "lithium_2022_ev_boom", "status": "error", "error": str(exc)})

    # ── Soybeans 2011 food price spike ───────────────────────────────────────
    try:
        from src.minerals.predictability import (
            _soybeans_2011_food_crisis,
            _soybeans_2015_supply_glut,
            _soybeans_2020_phase1,
            _soybeans_2022_ukraine_shock,
        )
        for ep_fn, ep_name in [
            (_soybeans_2011_food_crisis,  "soybeans_2011_food_price_spike"),
            (_soybeans_2015_supply_glut,  "soybeans_2015_supply_glut"),
            (_soybeans_2020_phase1,       "soybeans_2020_phase1_la_nina"),
            (_soybeans_2022_ukraine_shock,"soybeans_2022_ukraine_commodity_shock"),
        ]:
            r = ep_fn()
            episodes.append({
                "name":      r.name,
                "commodity": r.commodity,
                "status":    r.grade,
                "checks": [
                    {
                        "name":        "directional_accuracy",
                        "passed":      r.directional_accuracy >= 0.60,
                        "model_value": round(r.directional_accuracy, 3),
                        "cepii_value": None,
                        "notes":       "Fraction of year-on-year price moves predicted correctly",
                    },
                    {
                        "name":        "spearman_rho",
                        "passed":      r.spearman_rho >= 0.30,
                        "model_value": round(r.spearman_rho, 3),
                        "cepii_value": None,
                        "notes":       "Rank correlation of price index trajectory",
                    },
                    {
                        "name":        "log_price_rmse",
                        "passed":      r.log_price_rmse < 0.30,
                        "model_value": round(r.log_price_rmse, 3),
                        "cepii_value": None,
                        "notes":       "Log-price RMSE (lower is better)",
                    },
                ],
                "known_gap": r.known_gap,
            })
    except Exception as exc:
        episodes.append({"name": "soybeans_historical", "status": "error", "error": str(exc)})

    total = sum(1 for e in episodes if e.get("status") in ("pass", "A", "B"))
    return {
        "summary": f"{total}/{len(episodes)} episodes fully validated",
        "episodes": episodes,
    }


# ── Soybeans — forward projection & scenario comparison ──────────────────────

@app.get("/api/soybeans/tariff-scenarios")
def soybeans_tariff_scenarios():
    """
    Run the 2025 US-China tariff scenarios (BASE and NO DEAL) and return
    a structured comparison.

    BASE:    Partial deal signed in 2027 — China resumes some purchases.
    NO DEAL: Tariffs persist through 2028, structural US acreage shift.

    Returns year-by-year trajectories for both paths plus a Pearl L3
    counterfactual delta (shortage and price differential).

    Calibrated from FAS Q1 2026 data showing US exports -41% vs Q1 2024.
    Validated against 5 historical episodes (2 grade A, 2 grade B, 1 grade C).
    """
    import pandas as pd
    from src.minerals.schema import load_scenario
    from src.minerals.simulate import run_scenario as _run

    try:
        base_cfg    = load_scenario("scenarios/soybeans_2025_tariff_escalation.yaml")
        nodeal_cfg  = load_scenario("scenarios/soybeans_2025_tariff_no_deal.yaml")
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    base_df,   base_m   = _run(base_cfg)
    nodeal_df, nodeal_m = _run(nodeal_cfg)

    base_df   = base_df.set_index("year")
    nodeal_df = nodeal_df.set_index("year")
    ref_P     = float(base_df.loc[2023, "P"])

    def _row(df, yr, ref):
        r = df.loc[yr]
        return {
            "year":        yr,
            "price_index": round(float(r["P"]) / ref, 3),
            "q_effective": round(float(r["Q_eff"]), 1),
            "q_total":     round(float(r["Q_total"]), 1),
            "shortage":    round(float(r["shortage"]), 1),
            "inventory_cover": round(float(r["cover"]), 3),
        }

    years = list(base_df.index)

    return {
        "commodity": "soybeans",
        "base_year": 2023,
        "price_ref_usd_per_tonne": 533,
        "data_source": "Calibrated from UN Comtrade 2009-2024 + USDA FAS Q1 2026",
        "caveat": (
            "Global clearing price model. Brazil absorbs US export losses so "
            "actual world price impact is muted. Shortage numbers more reliable "
            "than price index for this commodity. Grade C on 2018 bilateral episode."
        ),
        "scenarios": {
            "base": {
                "description": "Partial deal in 2027 — China resumes ~60% of lost purchases",
                "trajectory": [_row(base_df, yr, ref_P) for yr in years],
                "metrics": {
                    "total_shortage":        round(base_m["total_shortage"], 1),
                    "peak_shortage":         round(base_m["peak_shortage"], 1),
                    "avg_price_index":       round(base_m["avg_price"], 3),
                    "final_inventory_cover": round(base_m["final_inventory_cover"], 3),
                },
            },
            "no_deal": {
                "description": "No deal — tariffs persist, structural US acreage shift 2027-2028",
                "trajectory": [_row(nodeal_df, yr, ref_P) for yr in years],
                "metrics": {
                    "total_shortage":        round(nodeal_m["total_shortage"], 1),
                    "peak_shortage":         round(nodeal_m["peak_shortage"], 1),
                    "avg_price_index":       round(nodeal_m["avg_price"], 3),
                    "final_inventory_cover": round(nodeal_m["final_inventory_cover"], 3),
                },
            },
        },
        "l3_counterfactual": {
            "description": "P(outcome | no_deal) - P(outcome | partial_deal)",
            "additional_shortage": round(
                nodeal_m["total_shortage"] - base_m["total_shortage"], 1
            ),
            "price_differential_2028_pct": round(
                (float(nodeal_df.loc[2028, "P"]) / float(base_df.loc[2028, "P"]) - 1) * 100, 1
            ),
            "interpretation": (
                "A partial deal in 2027 avoids this much cumulative shortage "
                "and price premium vs the no-deal path by 2028."
            ),
        },
    }


# ── Counterfactual (frontend shorthand) ──────────────────────────────────────

class FrontendCounterfactualRequest(BaseModel):
    scenario: str
    cf_type: str  # "substitution" | "fringe" | "trajectory"
    cf_elasticity: Optional[float] = None
    cf_cap: Optional[float] = None
    cf_capacity_share: Optional[float] = None
    cf_entry_price: Optional[float] = None
    shock_overrides: Optional[dict] = None
    use_calibrated: Optional[bool] = True

@app.post("/api/counterfactual")
def counterfactual(req: FrontendCounterfactualRequest):
    """
    POST /api/counterfactual — used by the Next.js Counterfactual page.

    Bridges the frontend's cf_type/scenario format to the Pearl L3 engine.
    """
    from src.minerals.schema import load_scenario
    from src.minerals.pearl_layers import counterfactual_substitution, counterfactual_fringe

    scenario_name = req.scenario
    # Try calibrated version first if requested
    if req.use_calibrated:
        cal_path = f"scenarios/calibrated/{scenario_name}_calibrated.yaml"
        if Path(cal_path).exists():
            scenario_name = f"calibrated/{scenario_name}_calibrated"

    try:
        cfg = load_scenario(f"scenarios/{scenario_name}.yaml")
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Scenario not found: {scenario_name}")

    try:
        if req.cf_type == "substitution":
            if req.cf_elasticity is None:
                raise HTTPException(status_code=400, detail="cf_elasticity required")
            result = counterfactual_substitution(cfg, cf_elasticity=req.cf_elasticity, cf_cap=req.cf_cap)
        elif req.cf_type == "fringe":
            if req.cf_capacity_share is None:
                raise HTTPException(status_code=400, detail="cf_capacity_share required")
            result = counterfactual_fringe(cfg, cf_capacity_share=req.cf_capacity_share, cf_entry_price=req.cf_entry_price)
        else:
            raise HTTPException(status_code=400, detail=f"cf_type '{req.cf_type}' not supported via this endpoint. Use /api/pearl/l3/counterfactual.")
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    return {
        "scenario":       req.scenario,
        "cf_type":        req.cf_type,
        "description":    result.description,
        "factual":        result.factual.to_dict(orient="records"),
        "counterfactual": result.counterfactual.to_dict(orient="records"),
        "ate":            {k: round(v, 6) for k, v in result.ate.items()},
        "factual_params": {k: round(v, 4) if isinstance(v, float) else v
                           for k, v in (result.factual_params if hasattr(result, "factual_params") else {}).items()},
        "cf_params":      result.cf_params if hasattr(result, "cf_params") else {},
    }


# ── Unified Workflow ──────────────────────────────────────────────────────────

@app.post("/api/workflow/run")
def run_workflow(req: UnifiedWorkflowRequest):
    if "Sensor" in (req.domain or ""):
        result = engine.run_full_pipeline_sensor(req.sensor_data, req.sensor_priors)
        return {"output": result, "run_dir": ""}
    output, run_dir = engine.run_full_pipeline_mineral(req.scenario_name)
    return {"output": output, "run_dir": run_dir}


# ── Synthetic Control ─────────────────────────────────────────────────────────

@app.post("/api/synthetic-control")
def synthetic_control():
    return {"result": engine.run_synthetic_control()}


# ── POMDP ─────────────────────────────────────────────────────────────────────

@app.post("/api/pomdp/build")
def pomdp_build(req: POmdpRequest):
    return {"result": engine.run_pomdp_build(req.data_path, req.priors_path)}

@app.post("/api/pomdp/and-causal")
def pomdp_and_causal(req: POmdpRequest):
    return {"result": engine.run_pomdp_and_causal(req.data_path, req.priors_path)}

@app.post("/api/pomdp/integrated")
def pomdp_integrated(req: POmdpIntegratedRequest):
    return {"result": engine.run_pomdp_causal_integrated(req.data_path)}


# ── Knowledge Graph ───────────────────────────────────────────────────────────

@app.post("/api/kg/rebuild")
def kg_rebuild():
    summary, shock_update = engine.kg_rebuild()
    return {"summary": summary, "shock_sources": _extract_shock_sources(shock_update)}

@app.get("/api/kg/shock-sources")
def shock_sources():
    return {"sources": engine.get_kg_shock_sources()}

@app.post("/api/kg/shock")
def kg_shock(req: KGShockRequest):
    return {"result": engine.run_kg_shock_propagation(req.origin_id)}

@app.get("/api/kg/identifiability")
def kg_identifiability():
    return {"result": engine.run_kg_identifiability()}

@app.get("/api/kg/dag-edges")
def kg_dag_edges():
    return {"result": engine.get_kg_dag_edges()}

@app.post("/api/kg/dag-image")
def kg_dag_image(req: KGDAGRequest):
    return _image_response(engine.get_kg_dag_image(req.simplified))

@app.post("/api/kg/dag-interactive")
def kg_dag_interactive(req: KGDAGRequest):
    return {"html": engine.get_kg_dag_interactive_html(req.simplified)}

@app.post("/api/kg/enrich")
def kg_enrich(req: KGEnrichRequest):
    return {"result": engine.kg_enrich_from_corpus(req.query, req.top_k)}

@app.post("/api/kg/batch-enrich")
def kg_batch_enrich(req: KGBatchEnrichRequest):
    return {"result": engine.kg_batch_enrich(req.top_k)}


# ── KG Scenario Renderer (custom user scenarios) ─────────────────────────────

# Lazy singletons. The first call to /api/kg/render-scenario warms them up
# (~30s for HippoRAG init); subsequent calls reuse the same objects, so each
# render is dominated by Claude API time (~30-90s) rather than re-init.
_KG_RENDERER_STATE: dict = {"kg_obj": None, "pipeline": None, "extractor": None}


def _get_scenario_renderer():
    state = _KG_RENDERER_STATE
    if state["kg_obj"] is None:
        from src.minerals.knowledge_graph import CausalKnowledgeGraph
        enriched_path = Path("data/canonical/enriched_kg.json")
        if not enriched_path.exists():
            raise HTTPException(
                status_code=500,
                detail=f"Enriched KG not found at {enriched_path}. "
                       "Run scripts/build_enriched_kg.py first.",
            )
        state["kg_obj"] = CausalKnowledgeGraph.load(str(enriched_path))
    if state["pipeline"] is None:
        from src.minerals.rag_pipeline import RAGPipeline
        # backend="auto" tries raganything → hipporag → industrial → simple,
        # using whichever is available. Production image only has "simple"
        # installed (hipporag pulls torch==2.5.1 which conflicts with other
        # deps); locally, hipporag is preferred when available.
        state["pipeline"] = RAGPipeline(backend="auto")
    if state["extractor"] is None:
        from src.minerals.kg_extractor import KGExtractor
        state["extractor"] = KGExtractor(pipeline=state["pipeline"])
    return state["kg_obj"], state["pipeline"], state["extractor"]


_SCENARIO_ID_RE = re.compile(r"[^A-Za-z0-9_\-]+")


@app.get("/api/kg/temporal-comparison")
def kg_temporal_comparison(commodity: Optional[str] = None):
    """
    Return year-by-year KG snapshots for one or all commodities.

    For each commodity, returns 3 chronological snapshots (pre/during/post
    a structural break) with PNG URLs and effective-control deltas — useful
    for showing how supply chains shift over time (thesis figures).

    Query params:
      commodity: filter to one commodity (graphite, rare_earths, cobalt,
                 lithium, nickel, uranium). Omit to return all 6 series.
    """
    from scripts.run_knowledge_graph import TEMPORAL_SERIES_DEFS, VALIDATION_SCENARIOS

    def _png_url(sid: str) -> tuple[str, bool]:
        # Validation PNGs live under validation/, temporal-only under temporal/
        for subdir in ("validation", "temporal", "predictive"):
            png_path = _KG_SCENARIO_DIR / subdir / f"{sid}.png"
            if png_path.exists():
                return f"/api/static/kg_scenarios/{subdir}/{sid}.png", True
        return f"/api/static/kg_scenarios/temporal/{sid}.png", False

    # Compute effective control at each snapshot year via the live KG so the
    # frontend can show structural deltas (e.g. China graphite PROCESSES:
    # 65% (2008) → 95% (2022)).
    try:
        from src.minerals.knowledge_graph import CausalKnowledgeGraph
        enriched_path = Path("data/canonical/enriched_kg.json")
        if enriched_path.exists():
            kg_obj = CausalKnowledgeGraph.load(str(enriched_path))
        else:
            kg_obj = None
    except Exception:
        kg_obj = None

    def _control(commodity_name: str, origin: str, year: int):
        if kg_obj is None:
            return None
        try:
            ctrl = kg_obj.effective_control_at(origin, commodity_name, year)
            if not ctrl:
                return None
            return {
                "effective_share": float(ctrl["effective_share"]) if ctrl.get("effective_share") is not None else None,
                "produces_share": float(ctrl["produces_share"]) if ctrl.get("produces_share") is not None else None,
                "processes_share": float(ctrl["processes_share"]) if ctrl.get("processes_share") is not None else None,
                "binding": ctrl.get("binding", "unknown"),
            }
        except Exception:
            return None

    series_filter = (commodity or "").strip().lower() or None
    out: dict = {}
    for cmd, entries in TEMPORAL_SERIES_DEFS.items():
        if series_filter and cmd != series_filter:
            continue
        snapshots = []
        for sid, year, origin, title in entries:
            url, available = _png_url(sid)
            entry_commodity = cmd
            ctrl = _control(entry_commodity, origin, year)
            snapshots.append({
                "scenario_id": sid,
                "year": year,
                "shock_origin": origin,
                "commodity": entry_commodity,
                "title": title,
                "image_url": url,
                "available": available,
                "effective_share": ctrl["effective_share"] if ctrl else None,
                "produces_share": ctrl["produces_share"] if ctrl else None,
                "processes_share": ctrl["processes_share"] if ctrl else None,
                "binding": ctrl["binding"] if ctrl else None,
            })
        out[cmd] = snapshots
    return out


@app.get("/api/kg/yearly-shares")
def kg_yearly_shares(commodity: str):
    """
    Return PRODUCES and PROCESSES share trajectories for every country that
    holds a non-zero share for the given commodity, across every year covered
    by the seed `yearly_share` data.

    Used by the temporal-comparison frontend to render share-over-time line
    charts. Cheap (no rendering, no LLM calls).

    Response:
      { commodity, years: [...], series: [{country, kind: produces|processes, share: [...]}, ...] }
    """
    from src.minerals.knowledge_graph import (
        CausalKnowledgeGraph, RelationType,
    )

    enriched_path = Path("data/canonical/enriched_kg.json")
    if not enriched_path.exists():
        raise HTTPException(status_code=500, detail="Enriched KG not loaded.")
    kg = CausalKnowledgeGraph.load(str(enriched_path))

    commodity_id = kg.resolve_id(commodity.lower())
    # Walk all PRODUCES + PROCESSES edges into this commodity, collect per-country yearly shares
    series: dict = {}  # (country, kind) -> {year: share}
    all_years: set[int] = set()

    for u, v, data in kg._graph.edges(data=True):
        if v != commodity_id:
            continue
        rel = data["relationship"]
        if rel.relation_type not in (RelationType.PRODUCES, RelationType.PROCESSES):
            continue
        kind = "produces" if rel.relation_type == RelationType.PRODUCES else "processes"
        yearly = (rel.properties or {}).get("yearly_share") or {}
        # yearly may have str keys (JSON) or int keys (Python). Normalize to int.
        if not yearly:
            # Single-share entry: spread across the full range as a constant
            s = (rel.properties or {}).get("share")
            if s is None:
                continue
            yearly = {y: float(s) for y in range(2000, 2025)}
        norm = {int(y): float(v_) for y, v_ in yearly.items()}
        series.setdefault((u, kind), {}).update(norm)
        all_years.update(norm.keys())

    if not all_years:
        return {"commodity": commodity, "years": [], "series": []}

    year_min, year_max = min(all_years), max(all_years)
    full_years = list(range(year_min, year_max + 1))

    # Linear-interpolate the per-country sparse year keys onto the full range
    def _interp(data_dict: dict, years: list[int]) -> list[Optional[float]]:
        if not data_dict:
            return [None] * len(years)
        sorted_years = sorted(data_dict.keys())
        out: list[Optional[float]] = []
        for y in years:
            if y in data_dict:
                out.append(data_dict[y])
            elif y < sorted_years[0]:
                out.append(data_dict[sorted_years[0]])
            elif y > sorted_years[-1]:
                out.append(data_dict[sorted_years[-1]])
            else:
                # Linear interp between bracketing keys
                lo = max(yy for yy in sorted_years if yy <= y)
                hi = min(yy for yy in sorted_years if yy >= y)
                if lo == hi:
                    out.append(data_dict[lo])
                else:
                    t = (y - lo) / (hi - lo)
                    out.append(data_dict[lo] * (1 - t) + data_dict[hi] * t)
        return out

    out_series = []
    for (country, kind), values_by_year in series.items():
        out_series.append({
            "country": country,
            "kind": kind,
            "share": _interp(values_by_year, full_years),
        })
    # Sort: processes ahead of produces (visual layering); within kind, larger peak first
    out_series.sort(key=lambda s: (s["kind"] != "processes", -max((x for x in s["share"] if x is not None), default=0.0)))

    return {"commodity": commodity, "years": full_years, "series": out_series}


@app.get("/api/kg/snapshots-export")
def kg_snapshots_export():
    """
    Return the pre-built KG snapshots appendix PDF (every pre-rendered KG
    snapshot, 2 per page). Generated locally and committed to git so the Fly
    machine doesn't OOM on the 2GB worker limit when bundling 24 PNGs.

    Regenerate via `python scripts/build_kg_snapshots_pdf.py` after adding
    new scenarios.
    """
    pdf_path = _KG_SCENARIO_DIR / "kg_snapshots_appendix.pdf"
    if not pdf_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Snapshots PDF not built. Run scripts/build_kg_snapshots_pdf.py.",
        )
    return FileResponse(
        path=str(pdf_path),
        media_type="application/pdf",
        filename="kg_snapshots_appendix.pdf",
    )


@app.get("/api/kg/yearly-grid-export")
def kg_yearly_grid_export():
    """
    Pre-built yearly grid appendix PDF: every-2-year KG snapshots from earliest
    to latest yearly_share data, per commodity, 1 KG per page.
    Regenerate via `python scripts/build_kg_yearly_grid_pdf.py`.
    """
    pdf_path = _KG_SCENARIO_DIR / "kg_yearly_grid_appendix.pdf"
    if not pdf_path.exists():
        raise HTTPException(status_code=404, detail="Yearly grid PDF not built.")
    return FileResponse(
        path=str(pdf_path),
        media_type="application/pdf",
        filename="kg_yearly_grid_appendix.pdf",
    )


@app.get("/api/kg/trajectory-export")
def kg_trajectory_export():
    """
    Render share-trajectory line charts for all 6 critical minerals as a
    publication-ready multi-page PDF (2 charts per page, landscape A4).

    Used as the thesis appendix figure: every year's PRODUCES (dashed) and
    PROCESSES (solid) share for every supplier, per commodity.
    """
    import io
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from src.minerals.knowledge_graph import CausalKnowledgeGraph, RelationType

    enriched_path = Path("data/canonical/enriched_kg.json")
    if not enriched_path.exists():
        raise HTTPException(status_code=500, detail="Enriched KG not loaded.")
    kg = CausalKnowledgeGraph.load(str(enriched_path))

    commodities = ["graphite", "rare_earths", "cobalt", "lithium", "nickel", "uranium"]
    commodity_titles = {
        "graphite": "Graphite — supplier shares 1995–2024",
        "rare_earths": "Rare Earths — supplier shares 2005–2024",
        "cobalt": "Cobalt — supplier shares 2010–2024",
        "lithium": "Lithium — supplier shares 2010–2024",
        "nickel": "Nickel — supplier shares 2010–2024",
        "uranium": "Uranium — supplier shares 2003–2024",
    }
    country_color = {
        "china": "#dc2626", "drc": "#a16207", "indonesia": "#7c3aed",
        "australia": "#16a34a", "chile": "#2563eb", "russia": "#475569",
        "kazakhstan": "#0891b2", "canada": "#ea580c", "philippines": "#9333ea",
        "mozambique": "#65a30d", "madagascar": "#84cc16", "brazil": "#10b981",
    }

    def _series_for(commodity: str):
        cid = kg.resolve_id(commodity)
        bucket: dict = {}
        all_years: set[int] = set()
        for u, v, data in kg._graph.edges(data=True):
            if v != cid:
                continue
            rel = data["relationship"]
            if rel.relation_type not in (RelationType.PRODUCES, RelationType.PROCESSES):
                continue
            kind = "produces" if rel.relation_type == RelationType.PRODUCES else "processes"
            yearly = (rel.properties or {}).get("yearly_share") or {}
            if not yearly:
                continue
            data_dict = {int(y): float(s) for y, s in yearly.items()}
            bucket[(u, kind)] = data_dict
            all_years.update(data_dict.keys())
        if not all_years:
            return [], []
        years = list(range(min(all_years), max(all_years) + 1))
        out = []
        for (country, kind), d in bucket.items():
            sorted_keys = sorted(d.keys())
            interp = []
            for y in years:
                if y in d:
                    interp.append(d[y])
                elif y < sorted_keys[0]:
                    interp.append(d[sorted_keys[0]])
                elif y > sorted_keys[-1]:
                    interp.append(d[sorted_keys[-1]])
                else:
                    lo = max(k for k in sorted_keys if k <= y)
                    hi = min(k for k in sorted_keys if k >= y)
                    t = 0 if lo == hi else (y - lo) / (hi - lo)
                    interp.append(d[lo] * (1 - t) + d[hi] * t)
            peak = max(interp) if interp else 0
            out.append({"country": country, "kind": kind, "share": interp, "peak": peak})
        out.sort(key=lambda s: (s["kind"] != "processes", -s["peak"]))
        return years, out

    def _plot(ax, commodity: str):
        years, series = _series_for(commodity)
        if not series:
            ax.text(0.5, 0.5, f"No share data for {commodity}", ha="center", va="center",
                    transform=ax.transAxes, fontsize=11, color="#94a3b8")
            ax.set_axis_off()
            return
        for s in series:
            color = country_color.get(s["country"], "#94a3b8")
            linestyle = "--" if s["kind"] == "produces" else "-"
            label = f"{s['country']} {s['kind']}"
            ax.plot(years, s["share"], color=color, linestyle=linestyle,
                    linewidth=1.8, label=label, alpha=0.9)
        ax.set_title(commodity_titles[commodity], fontsize=11, fontweight="bold", loc="left", pad=8)
        ax.set_xlabel("Year", fontsize=9)
        ax.set_ylabel("Share of global supply", fontsize=9)
        ax.set_ylim(0, 1.0)
        ax.set_xlim(years[0], years[-1])
        ax.grid(True, alpha=0.25, linewidth=0.5)
        ax.tick_params(axis="both", which="major", labelsize=8)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
        ax.legend(loc="best", fontsize=7, framealpha=0.9, ncol=2)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    buf = io.BytesIO()
    with PdfPages(buf) as pdf:
        # Title page
        fig = plt.figure(figsize=(11, 8.5))
        fig.text(0.5, 0.65, "Critical Minerals Causal Engine",
                 ha="center", fontsize=22, fontweight="bold")
        fig.text(0.5, 0.55, "Appendix — Year-by-Year Supplier Share Trajectories",
                 ha="center", fontsize=14, color="#475569")
        fig.text(0.5, 0.45,
                 "PRODUCES (dashed) and PROCESSES (solid) share per supplier, per commodity.\n"
                 "Sources: USGS MCS 2024, World Nuclear Association, CEPII BACI,\n"
                 "Cobalt Institute, Benchmark Mineral Intelligence.",
                 ha="center", fontsize=10, color="#64748b")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # 2 charts per page (landscape A4 ≈ 11" × 8.5")
        for i in range(0, len(commodities), 2):
            fig, axes = plt.subplots(2, 1, figsize=(11, 8.5),
                                     gridspec_kw={"hspace": 0.45})
            _plot(axes[0], commodities[i])
            if i + 1 < len(commodities):
                _plot(axes[1], commodities[i + 1])
            else:
                axes[1].set_axis_off()
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

    buf.seek(0)
    from fastapi.responses import StreamingResponse
    return StreamingResponse(
        buf,
        media_type="application/pdf",
        headers={"Content-Disposition": 'attachment; filename="kg_supplier_share_trajectories.pdf"'},
    )


@app.get("/api/kg/year-snapshot")
def kg_year_snapshot(commodity: str, year: int, shock_origin: Optional[str] = None):
    """
    On-demand KG snapshot for any (commodity, year) pair. Renders fast (~2-5s)
    by skipping HippoRAG retrieval and Claude triple extraction — focal nodes
    are just {shock_origin, commodity}, and the year-specific shares come from
    the KG's effective_control_at(year). Filesystem-cached at
    outputs/kg_scenarios/yearly/<commodity>_<origin>_<year>.png so subsequent
    requests for the same (commodity, year) are instant.

    Used by the temporal-comparison year slider.
    """
    try:
        commodity = commodity.strip().lower()
        if not shock_origin:
            shock_origin = _DEFAULT_ORIGIN_BY_COMMODITY.get(commodity, commodity)
        shock_origin = shock_origin.strip().lower()
        scenario_id = f"{commodity}_{shock_origin}_{year}"
        scenario_id = _SCENARIO_ID_RE.sub("_", scenario_id).strip("_")

        out_dir = _KG_SCENARIO_DIR / "yearly"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{scenario_id}.png"
        image_url = f"/api/static/kg_scenarios/yearly/{scenario_id}.png"

        # Always compute the share data so the frontend has something to show
        # even before the PNG renders.
        from src.minerals.knowledge_graph import CausalKnowledgeGraph
        enriched_path = Path("data/canonical/enriched_kg.json")
        kg_obj = CausalKnowledgeGraph.load(str(enriched_path)) if enriched_path.exists() else None
        ctrl = None
        if kg_obj is not None:
            try:
                c = kg_obj.effective_control_at(shock_origin, commodity, int(year))
                ctrl = {
                    "effective_share": float(c["effective_share"]) if c.get("effective_share") is not None else None,
                    "produces_share": float(c["produces_share"]) if c.get("produces_share") is not None else None,
                    "processes_share": float(c["processes_share"]) if c.get("processes_share") is not None else None,
                    "binding": c.get("binding", "unknown"),
                }
            except Exception:
                ctrl = None

        if out_path.exists():
            return {
                "scenario_id": scenario_id, "image_url": image_url,
                "year": int(year), "commodity": commodity, "shock_origin": shock_origin,
                "cached": True, "control": ctrl,
            }

        if kg_obj is None:
            raise HTTPException(status_code=500, detail="Enriched KG not loaded.")

        # Fast render — skip HippoRAG/Claude. Focal nodes = {shock_origin, commodity}.
        from scripts.run_knowledge_graph import _render_scenario as render_fn
        scenario = {
            "year": int(year), "shock_origin": shock_origin,
            "commodity": commodity,
            "title": f"{commodity.title()} {year} — {shock_origin.upper()} snapshot",
        }
        stats = render_fn(
            kg_obj=kg_obj, scenario_id=scenario_id, scenario=scenario,
            output_path=str(out_path),
            pipeline=None, extractor=None, enriched=True,
        )
        return {
            "scenario_id": scenario_id, "image_url": image_url,
            "year": int(year), "commodity": commodity, "shock_origin": shock_origin,
            "cached": False, "control": ctrl,
            "node_count": stats.get("node_count", 0),
            "edge_count": stats.get("edge_count", 0),
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}")


_DEFAULT_ORIGIN_BY_COMMODITY = {
    "graphite": "china",
    "rare_earths": "china",
    "cobalt": "drc",
    "lithium": "chile",
    "nickel": "indonesia",
    "uranium": "russia",
    "copper": "chile",
    "soybeans": "usa",
}


@app.get("/api/kg/scenario-presets")
def kg_scenario_presets():
    """
    Return the catalogue of pre-rendered scenarios (10 validation + 6 predictive)
    plus their PNG URLs. Validation PNGs are baked into the Docker image and
    served instantly; predictive PNGs may not exist yet (404 if so).

    Frontend uses this to populate the scenario dropdown / gallery on the
    Scenario Builder page.
    """
    from scripts.run_knowledge_graph import VALIDATION_SCENARIOS, PREDICTIVE_SCENARIOS

    def _entries(table: dict, kind: str, subdir: str):
        items = []
        for sid, s in table.items():
            png_path = _KG_SCENARIO_DIR / subdir / f"{sid}.png"
            items.append({
                "scenario_id": sid,
                "kind": kind,
                "year": s["year"],
                "shock_origin": s["shock_origin"],
                "commodity": s["commodity"],
                "title": s["title"],
                "image_url": f"/api/static/kg_scenarios/{subdir}/{sid}.png",
                "available": png_path.exists(),
            })
        return items

    return {
        "validation": _entries(VALIDATION_SCENARIOS, "validation", "validation"),
        "predictive": _entries(PREDICTIVE_SCENARIOS, "predictive", "predictive"),
    }


@app.post("/api/kg/render-scenario")
def render_scenario(req: KGRenderScenarioRequest):
    """
    Generate a knowledge-graph PNG for a user-defined scenario.

    Reuses _render_scenario from scripts/run_knowledge_graph.py — same
    HippoRAG retrieval → Claude triple extraction → focal nodes → subgraph
    render pipeline as the validation/predictive scenario suite.

    Takes 30-90s per call (Claude API time). The first request also warms
    up the HippoRAG + KGExtractor singletons (~30s extra).
    """
    try:
        commodity = req.commodity.strip().lower()
        shock_origin = req.shock_origin.strip().lower()
        if not commodity or not shock_origin or not req.title.strip():
            raise HTTPException(
                status_code=400,
                detail="commodity, shock_origin, and title are all required",
            )

        scenario_id = req.scenario_id or f"{commodity}_{req.year}_custom_{int(time.time())}"
        scenario_id = _SCENARIO_ID_RE.sub("_", scenario_id).strip("_") or f"custom_{int(time.time())}"

        out_path = _KG_SCENARIO_DIR / "custom" / f"{scenario_id}.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)

        kg_obj, pipeline, extractor = _get_scenario_renderer()

        from scripts.run_knowledge_graph import _render_scenario as render_fn
        scenario_dict = {
            "year": int(req.year),
            "shock_origin": shock_origin,
            "commodity": commodity,
            "title": req.title.strip(),
        }
        stats = render_fn(
            kg_obj=kg_obj,
            scenario_id=scenario_id,
            scenario=scenario_dict,
            output_path=str(out_path),
            pipeline=pipeline,
            extractor=extractor,
            enriched=True,
        )

        return {
            "scenario_id": scenario_id,
            "image_url": f"/api/static/kg_scenarios/custom/{scenario_id}.png",
            "node_count": stats.get("node_count", 0),
            "focal_count": stats.get("focal_count", 0),
            "edge_count": stats.get("edge_count", 0),
            "impact_count": stats.get("impact_count", 0),
            "effective_share": stats.get("effective_share"),
            "binding": stats.get("binding"),
            "query": stats.get("query", ""),
            "skipped": stats.get("skipped", False),
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}")


# ── Three Layers ──────────────────────────────────────────────────────────────

@app.post("/api/three-layers")
def three_layers(req: ThreeLayersRequest):
    return {"result": engine.three_layers_query(
        req.run_dir, req.layer, req.treatment, req.outcome, req.cf_year, req.cf_value
    )}


# ── Pearl L1/L2/L3 — structured causal endpoints ─────────────────────────────

@app.get("/api/pearl/summary")
def pearl_summary():
    """
    Return the three-layer implementation map.

    Layer 1 (Seeing):   observational_conditional, observe_substitution_association
    Layer 2 (Doing):    do_substitution, do_fringe_supply, do_compare
    Layer 3 (Imagining): counterfactual_substitution, counterfactual_fringe
    """
    from src.minerals.pearl_layers import three_layers_summary
    return {"summary": three_layers_summary()}


@app.post("/api/pearl/l2/do")
def pearl_do_intervention(req: DoInterventionRequest):
    """
    Layer 2 — Intervention: P(Y | do(parameters)).

    Applies graph surgery on the named structural parameters (substitution_elasticity,
    fringe_capacity_share, etc.) and runs the simulation forward.  Returns:
      - factual: year-by-year output under the original scenario
      - intervention: year-by-year output under the do(·) scenario
      - ate_per_year: per-year Average Treatment Effect for each outcome
      - ate_mean: mean ATE over all years for each outcome

    Example request:
      {"scenario_name": "graphite_2023_no_substitution",
       "parameter_overrides": {"substitution_elasticity": 0.8},
       "outcomes": ["P", "Q_sub", "shortage"]}
    """
    from src.minerals.schema import load_scenario
    from src.minerals.pearl_layers import do_compare

    try:
        cfg = load_scenario(f"scenarios/{req.scenario_name}.yaml")
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Scenario not found: {req.scenario_name}")

    supported = {
        "substitution_elasticity", "substitution_cap",
        "fringe_capacity_share", "fringe_entry_price",
        "eta_D", "alpha_P", "tau_K", "eta_K",
    }
    bad = set(req.parameter_overrides) - supported
    if bad:
        raise HTTPException(status_code=400, detail=f"Unsupported parameters: {bad}. Supported: {supported}")

    try:
        compare_df = do_compare(cfg, req.parameter_overrides, outcomes=req.outcomes)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    records = compare_df.to_dict(orient="records")
    ate_mean = {}
    for col in req.outcomes:
        ate_col = f"{col}_ate"
        if ate_col in compare_df.columns:
            ate_mean[col] = round(float(compare_df[ate_col].mean()), 6)

    return {
        "scenario": req.scenario_name,
        "intervention": req.parameter_overrides,
        "layer": "L2 — Intervention P(Y|do(X))",
        "outcomes": req.outcomes,
        "trajectory": records,
        "ate_mean": ate_mean,
        "description": (
            f"Graph surgery: parameters {list(req.parameter_overrides.keys())} were severed "
            f"from their upstream causes and pinned to {list(req.parameter_overrides.values())}."
        ),
    }


@app.post("/api/pearl/l3/counterfactual")
def pearl_counterfactual(req: CounterfactualRequest):
    """
    Layer 3 — Counterfactual: P(Y_x | factual trajectory).

    Abduction-Action-Prediction:
      1. Abduction:  run factual scenario, capture noise sequence ε_t
      2. Action:     apply do(mechanism parameters = cf values)
      3. Prediction: replay with same ε_t, modified structural equation

    Returns factual and counterfactual trajectories, ATE per outcome,
    noise sequence, and human-readable description.

    Example request (substitution):
      {"scenario_name": "graphite_2023_no_substitution",
       "mechanism": "substitution",
       "cf_elasticity": 0.8}

    Example request (fringe):
      {"scenario_name": "lithium_2022_ev_boom_with_fringe",
       "mechanism": "fringe",
       "cf_capacity_share": 0.4,
       "cf_entry_price": 1.1}
    """
    from src.minerals.schema import load_scenario
    from src.minerals.pearl_layers import counterfactual_substitution, counterfactual_fringe

    try:
        cfg = load_scenario(f"scenarios/{req.scenario_name}.yaml")
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Scenario not found: {req.scenario_name}")

    try:
        if req.mechanism == "substitution":
            if req.cf_elasticity is None:
                raise HTTPException(status_code=400, detail="cf_elasticity required for mechanism=substitution")
            result = counterfactual_substitution(cfg, cf_elasticity=req.cf_elasticity, cf_cap=req.cf_cap)
        elif req.mechanism == "fringe":
            if req.cf_capacity_share is None:
                raise HTTPException(status_code=400, detail="cf_capacity_share required for mechanism=fringe")
            result = counterfactual_fringe(cfg, cf_capacity_share=req.cf_capacity_share, cf_entry_price=req.cf_entry_price)
        else:
            raise HTTPException(status_code=400, detail=f"Unknown mechanism '{req.mechanism}'. Use 'substitution' or 'fringe'.")
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    return {
        "scenario": req.scenario_name,
        "mechanism": req.mechanism,
        "layer": "L3 — Counterfactual P(Y_x | X', Y')",
        "description": result.description,
        "ate": {k: round(v, 6) for k, v in result.ate.items()},
        "noise_sequence": result.noise_sequence,
        "factual_trajectory": result.factual.to_dict(orient="records"),
        "counterfactual_trajectory": result.counterfactual.to_dict(orient="records"),
        "abduction_note": (
            "Noise sequence ε_t was abduced from the factual run (same RNG seed). "
            "Both factual and counterfactual use identical ε_t — only the structural "
            "equation for the named mechanism differs."
        ),
    }


@app.post("/api/pearl/l1/association")
def pearl_l1_association(req: ScenarioRequest):
    """
    Layer 1 — Association: compute observational statistics from a scenario run.

    Runs the named scenario and returns:
      - substitution: Spearman ρ(Q_sub, export_restriction) binned by restriction status
      - fringe:       Spearman ρ(Q_fringe, P) binned by price quartile

    These are purely observational (no causal claim). To get causal effects use L2/L3.
    """
    from src.minerals.schema import load_scenario
    from src.minerals.simulate import run_scenario as _run
    from src.minerals.pearl_layers import observe_substitution_association, observe_fringe_association

    try:
        cfg = load_scenario(f"scenarios/{req.scenario_name}.yaml")
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Scenario not found: {req.scenario_name}")

    df, _ = _run(cfg)

    try:
        sub_summary = observe_substitution_association(df).to_dict(orient="records")
    except Exception as exc:
        sub_summary = {"error": str(exc)}

    try:
        fringe_summary = observe_fringe_association(df).to_dict(orient="records")
    except Exception as exc:
        fringe_summary = {"error": str(exc)}

    return {
        "scenario": req.scenario_name,
        "layer": "L1 — Association P(Y|X)",
        "warning": "Observational correlations only — not causal effects. Use /api/pearl/l2/do for interventional estimates.",
        "substitution_association": sub_summary,
        "fringe_association": fringe_summary,
    }


# ── Knowledge Graph — structured causal KG endpoints ─────────────────────────

class KGQueryRequest(BaseModel):
    commodity: Optional[str] = None  # filter to entities related to this commodity
    include_relationships: bool = True


class ExtractShockRequest(BaseModel):
    text: str
    use_llm: bool = False             # True → KGExtractor (requires LLM API key); False → rule-based
    default_duration: int = 2


class PredictFromTextRequest(BaseModel):
    text: str
    commodity: str = "graphite"
    start_year: int = 2023
    end_year: int = 2026
    use_llm: bool = False
    baseline_P0: float = 1.0
    baseline_K0: float = 108.695652


@app.get("/api/knowledge-graph")
def get_knowledge_graph(commodity: Optional[str] = None, include_relationships: bool = True):
    """
    Return the Critical Minerals Causal Knowledge Graph structure.

    Nodes are typed entities (Commodity, Country, Company, Policy, Index).
    Edges are typed causal relationships (PRODUCES, EXPORTS_TO, REGULATES, etc.).

    Optionally filter to entities related to a specific commodity.
    """
    try:
        from src.minerals.knowledge_graph import build_critical_minerals_kg, EntityType

        kg = build_critical_minerals_kg()
        data = kg.to_dict()

        if commodity:
            commodity_lower = commodity.lower()
            # Find entity IDs that match the commodity
            commodity_ids = {
                e["id"] for e in data["entities"]
                if commodity_lower in e["id"].lower()
                or any(commodity_lower in a.lower() for a in e.get("aliases", []))
            }
            if commodity_ids:
                # Include entities within 1 hop of the commodity nodes
                related_ids = set(commodity_ids)
                for rel in data.get("relationships", []):
                    if rel["source_id"] in commodity_ids or rel["target_id"] in commodity_ids:
                        related_ids.add(rel["source_id"])
                        related_ids.add(rel["target_id"])
                data["entities"] = [e for e in data["entities"] if e["id"] in related_ids]
                if include_relationships:
                    data["relationships"] = [
                        r for r in data.get("relationships", [])
                        if r["source_id"] in related_ids and r["target_id"] in related_ids
                    ]
                else:
                    data.pop("relationships", None)
                data["metadata"]["filtered_by"] = commodity
                data["metadata"]["num_entities"] = len(data["entities"])
                data["metadata"]["num_relationships"] = len(data.get("relationships", []))

        if not include_relationships:
            data.pop("relationships", None)

        return data

    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/extract-shock")
def extract_shock_from_text(req: ExtractShockRequest):
    """
    Extract ShockConfig objects from free-form text (news articles, policy docs).

    Pipeline (Option C — Event Extraction → Shock Graph):
      1. Rule-based scan for (country, predicate_keyword, commodity) triples
         — or LLM-based triple extraction if use_llm=True
      2. Predicate keyword → shock_type mapping
      3. KG propagation to find affected supply chain entities
      4. Returns ranked ShockMappings with commodity, shock parameters, and reasoning

    This bridges the Knowledge Graph extraction pipeline to the causal ODE model.
    Set use_llm=True if an LLM API key is configured and you want higher recall.
    """
    try:
        from src.minerals.event_shock_mapper import EventShockMapper

        mapper = EventShockMapper()

        extractor = None
        if req.use_llm:
            try:
                from src.minerals.kg_extractor import KGExtractor
                extractor = KGExtractor()
            except Exception:
                pass  # fall back to rule-based if extractor unavailable

        mappings = mapper.text_to_shocks(
            req.text,
            extractor=extractor,
            default_duration=req.default_duration,
        )

        return {
            "n_shocks_extracted": len(mappings),
            "extraction_method": "llm" if (req.use_llm and extractor is not None) else "rule_based",
            "shocks": mapper.mappings_to_dict(mappings),
        }

    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/predict-from-text")
def predict_from_text(req: PredictFromTextRequest):
    """
    Full pipeline: raw text → extract shocks → run causal scenario → return trajectory.

    Steps:
      1. Extract ShockConfig objects from text via EventShockMapper
      2. Filter to shocks for the requested commodity
      3. Build a ScenarioConfig with the extracted shocks
      4. Run the causal ODE model
      5. Return year-by-year trajectory + extracted shocks

    This implements the complete Option A + C multimodal KG pipeline:
    text → KG event extraction → shock specification → causal model → price trajectory.
    """
    try:
        from src.minerals.event_shock_mapper import EventShockMapper
        from src.minerals.schema import (
            BaselineConfig, DemandGrowthConfig, OutputsConfig,
            ParametersConfig, PolicyConfig, ScenarioConfig,
            ShockConfig, TimeConfig,
        )
        from src.minerals.simulate import run_scenario as _run

        mapper = EventShockMapper()

        extractor = None
        if req.use_llm:
            try:
                from src.minerals.kg_extractor import KGExtractor
                extractor = KGExtractor()
            except Exception:
                pass

        mappings = mapper.text_to_shocks(req.text, extractor=extractor)

        # Filter to requested commodity and within requested time window
        commodity_shocks = [
            m for m in mappings
            if m.commodity == req.commodity.lower()
            and m.shock.start_year >= req.start_year - 2  # allow some look-back
        ]

        # Build ShockConfig list from mappings (clamp to time window)
        shock_cfgs = []
        for m in commodity_shocks:
            sc = m.shock
            shock_cfgs.append(ShockConfig(
                type=sc.type,
                start_year=max(sc.start_year, req.start_year),
                end_year=min(sc.end_year, req.end_year),
                magnitude=sc.magnitude,
            ))

        cfg = ScenarioConfig(
            name=f"text_inferred_{req.commodity}_{req.start_year}",
            commodity=req.commodity,
            seed=42,
            time=TimeConfig(dt=1.0, start_year=req.start_year, end_year=req.end_year),
            baseline=BaselineConfig(
                P_ref=1.0,
                P0=req.baseline_P0,
                K0=req.baseline_K0,
                I0=20.0,
                D0=100.0,
            ),
            parameters=ParametersConfig(
                eps=1e-9, u0=0.92, beta_u=0.10, u_min=0.70, u_max=1.00,
                tau_K=3.0, eta_K=0.40, retire_rate=0.0, eta_D=-0.25,
                demand_growth=DemandGrowthConfig(type="constant", g=1.0),
                alpha_P=0.80, cover_star=0.20, lambda_cover=0.60, sigma_P=0.0,
            ),
            policy=PolicyConfig(),
            shocks=shock_cfgs,
            outputs=OutputsConfig(metrics=["total_shortage", "peak_shortage", "avg_price"]),
        )

        df, metrics = _run(cfg)

        return {
            "commodity": req.commodity,
            "text_length": len(req.text),
            "n_shocks_extracted": len(mappings),
            "n_shocks_applied": len(shock_cfgs),
            "extraction_method": "llm" if (req.use_llm and extractor is not None) else "rule_based",
            "extracted_shocks": mapper.mappings_to_dict(commodity_shocks),
            "trajectory": df.to_dict(orient="records"),
            "metrics": {k: round(float(v), 4) for k, v in metrics.items()},
        }

    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
