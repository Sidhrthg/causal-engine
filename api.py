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
