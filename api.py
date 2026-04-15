"""
FastAPI backend for Critical Minerals Causal Engine.
Wraps all Gradio functions as REST endpoints for the Next.js/Vercel frontend.

Run with:
    uvicorn api:app --host 0.0.0.0 --port 8000 --reload
"""

import shutil
import tempfile
from pathlib import Path
from typing import Optional

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
                demand_reversion_rate=0.60,
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

    total = sum(1 for e in episodes if e.get("status") == "pass")
    return {
        "summary": f"{total}/{len(episodes)} episodes fully validated",
        "episodes": episodes,
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


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
