"""
FastAPI backend for Critical Minerals Causal Engine.
Wraps all Gradio functions as REST endpoints for the Next.js/Vercel frontend.

Run with:
    uvicorn api:app --host 0.0.0.0 --port 8000 --reload
"""

import math
import shutil
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np

import matplotlib
matplotlib.use("Agg")

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
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


# ── Request models ────────────────────────────────────────────────────────────

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


# ── Scenarios ─────────────────────────────────────────────────────────────────

@app.get("/api/scenarios")
def get_scenarios():
    return {"scenarios": engine.list_scenarios()}


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


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
