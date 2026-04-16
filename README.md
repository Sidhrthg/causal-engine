# Causal Engine

A causal inference engine for supply chain policy simulation, built on Pearl's Ladder of Causation. Designed for critical minerals analysis — model supply disruptions, run counterfactual scenarios, extract causal structure from documents, and validate outputs against real trade data.

**Live demo:** [causal-engine.vercel.app](https://causal-engine.vercel.app) · **Backend:** FastAPI · **Frontend:** Next.js on Vercel

---

## What it does

- **Causal simulation** — year-by-year supply chain model under shocks: export restrictions, demand surges, cost spikes, stockpile releases
- **Pearl's three layers** — observational queries (L1), interventions `do(X)` (L2), and counterfactuals `P(Y_x | X', Y')` (L3)
- **Causal identification** — check whether `P(Y | do(X))` is identifiable using backdoor/frontdoor criteria and do-calculus rules
- **Knowledge graph** — typed KG of countries, commodities, and policies with CAUSES/PRODUCES/EXPORTS_TO edges; LLM-extracted from documents
- **RAG pipeline** — tiered document retrieval (TF-IDF → ChromaDB → HippoRAG) with episodic memory and self-evaluation
- **POMDP** — sensor degradation and maintenance scheduling under partial observability
- **Validation** — compare simulation outputs against UN Comtrade historical data via RAG-powered LLM analysis

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Next.js Frontend (Vercel)               │
│   Ask · Scenarios · RAG · KG · Causal · Validate · POMDP   │
└──────────────────────┬──────────────────────────────────────┘
                       │ REST (rewrites → BACKEND_URL)
┌──────────────────────▼──────────────────────────────────────┐
│                    FastAPI Backend  (api.py)                 │
└──┬──────────┬────────────┬──────────┬──────────┬────────────┘
   │          │            │          │          │
   ▼          ▼            ▼          ▼          ▼
 [SCM]   [3-Layer     [Knowledge   [RAG        [POMDP
          Engine]      Graph]      Pipeline]   Module]
```

### 1. Structural Causal Model — `src/scm.py`, `src/simulate.py`, `src/estimate.py`

The SCM is a system-dynamics model of a commodity supply chain. At each time step it computes capacity `K`, inventory `I`, demand `D`, price `P`, and utilisation `u` using differential equations governed by:

- **Demand price elasticity** (`eta_D`) — how demand responds to price
- **Capacity adjustment lag** (`tau_K`) — years for new capacity to come online
- **Shock signals** — additive shocks to supply, demand, cost, or policy applied to specific years

Estimation uses DoWhy's causal model + EconML's Doubly Robust Learner for ATE estimation with 95% bootstrap CIs.

### 2. Pearl's Three-Layer Engine — `src/minerals/causal_engine.py`, `src/minerals/pearl_layers.py`

Implements the Causal Hierarchy Theorem (Bareinboim et al., 2022):

| Layer | Query | Meaning |
|-------|-------|---------|
| L1 — Association | `P(Y\|X)` | What do I observe when I see X? |
| L2 — Intervention | `P(Y\|do(X))` | What happens if I force X? |
| L3 — Counterfactual | `P(Y_x \| X', Y')` | What would Y have been, had X been x? |

Each layer builds on the layer below. L2 requires a causal DAG to apply do-calculus; L3 additionally requires abduction over the SCM noise variables.

### 3. Causal Identification — `src/minerals/causal_identification.py`, `src/minerals/do_calculus.py`

Given a DAG, checks whether `P(Y | do(X))` is identifiable without running an experiment:

1. **Backdoor criterion** — find a set Z blocking all back-door paths from X to Y
2. **Frontdoor criterion** — find a mediator set M if backdoor fails
3. **Do-calculus rules** (Pearl's Rules 1–3) — symbolic derivation with step-by-step trace
4. Returns the adjustment formula, the strategy used, and the full derivation

### 4. Causal Knowledge Graph — `src/minerals/knowledge_graph.py`, `src/minerals/kg_extractor.py`

A typed, temporal KG that goes beyond a bare DAG:

- **Entity types:** Country, Commodity, Company, Policy, Technology, MarketEvent
- **Relationship types:** CAUSES, PRODUCES, EXPORTS_TO, IMPORTS_FROM, SUBSTITUTES, AFFECTS, PART_OF
- **Properties per edge:** confidence score, mechanism description, temporal scope, evidence source
- **Shock propagation:** breadth-first propagation from an origin node with configurable decay
- **Bridge to inference:** `kg.to_causal_dag()` extracts the CAUSES sub-graph as a `CausalDAG` for L1–L3 queries

LLM extraction (`kg_extractor.py`) retrieves relevant document chunks via RAG, asks the LLM to identify causal triples, and merges them into the live KG.

### 5. RAG Pipeline — `src/minerals/rag_pipeline.py`, `src/minerals/rag_retrieval.py`

Tiered retrieval with automatic backend selection:

```
HippoRAG (graph-based, best)
    ↑ if installed + indexed
IndustrialRAG (ChromaDB + sentence-transformers)
    ↑ if chromadb installed
SimpleRAGRetriever (TF-IDF, always available)
```

The pipeline wraps all backends with:
- **Episodic memory** (`src/llm/memory.py`) — stores past Q&A episodes, injects high-quality ones as few-shot examples
- **Self-evaluation** (`src/minerals/rag_eval.py`) — generates synthetic questions, measures Hit@K / MRR / faithfulness, triggers self-learning
- **Feedback loop** — user thumbs-up/down ratings update episode quality scores

### 6. POMDP Module — `src/pomdp/`

Models sensor reliability and maintenance scheduling as a Partially Observable Markov Decision Process:

- **States:** sensor health levels (good → degraded → failed)
- **Actions:** observe, maintain, replace
- **Observations:** noisy sensor readings
- **Belief updates:** Bayesian filtering over the hidden state
- **Policies:** threshold-based and value-iteration policies
- Integrated with the causal engine to propagate sensor uncertainty into supply chain estimates

### 7. LLM Layer — `src/llm/`

Provider-agnostic LLM backend. Supports:

| Backend | Config |
|---------|--------|
| Anthropic Claude | `ANTHROPIC_API_KEY` |
| OpenAI | `OPENAI_API_KEY` |
| Local vLLM | `LLM_BACKEND=vllm`, `VLLM_BASE_URL`, `VLLM_MODEL` |

### 8. API & Frontend — `api.py`, `main.py`, `frontend/`

- `api.py` — FastAPI, all business logic exposed as typed REST endpoints
- `main.py` — thin re-export for ASGI runners (`from api import app`)
- `frontend/` — Next.js app; all `/api/*` calls rewrite to `BACKEND_URL` via `next.config.ts`

---

## Setup

**Python 3.10–3.12 required.**

```bash
python3.12 -m venv .venv
source .venv/bin/activate

pip install -e ".[dev]"          # core + tests
pip install -e ".[rag]"          # ChromaDB + sentence-transformers
pip install -e ".[ui]"           # Gradio (legacy UI)
pip install -e ".[hipporag]"     # optional graph-based RAG
```

**HippoRAG (optional):** install + index build → **[docs/HIPPORAG.md](docs/HIPPORAG.md)**.

**`.env` in project root:**

```bash
ANTHROPIC_API_KEY=sk-ant-...    # LLM query, RAG, causal discovery
OPENAI_API_KEY=sk-...           # alternative LLM backend
COMTRADE_API_KEY=...            # UN Comtrade data download

# Use a local vLLM server instead:
LLM_BACKEND=vllm
VLLM_BASE_URL=http://localhost:8000
VLLM_MODEL=mistral-7b
```

---

## Quickstart

```bash
# Run a baseline scenario
python -m scripts.run_scenario --scenario scenarios/graphite_baseline_2000_2011.yaml

# Ask a causal question
python -m scripts.llm_query "What happens if China bans graphite exports in 2027?"

# Launch the REST API
uvicorn main:app --reload --port 8000

# Launch the Next.js frontend (separate terminal)
cd frontend && npm install && npm run dev   # → http://localhost:3000

# Run tests
pytest tests/ -q
```

---

## Scenarios

Pre-built scenarios in `scenarios/`:

| Scenario | Description |
|----------|-------------|
| `graphite_baseline_2000_2011.yaml` | Historical baseline calibrated to Comtrade data |
| `graphite_2008_multishock.yaml` | 2008 demand surge + financial crisis + 2010 China export quotas |
| `china_export_restriction_40pct_2025.yaml` | 40% Chinese export reduction from 2025 |
| `china_graphite_export_ban_2027.yaml` | Complete Chinese export ban from 2027 |
| `us_china_trade_stop.yaml` | Full US–China trade halt |

Each run saves `timeseries.csv` and `metrics.json` (total shortage, peak shortage, avg price, inventory cover) to `runs/<scenario>/<timestamp>/`.

**Scenario YAML structure:**

```yaml
name: china_export_restriction_40pct_2025
commodity: graphite
time:
  start_year: 2024
  end_year: 2030
baseline:
  K0: 108.7    # initial capacity (Mt)
  I0: 20.0     # initial inventory
  D0: 100.0    # initial demand
  P0: 1.0      # initial price index
parameters:
  eta_D: -0.25  # demand price elasticity
  tau_K: 3.0    # capacity adjustment lag (years)
shocks:
  - type: export_restriction
    start_year: 2025
    magnitude: 0.4   # 40% reduction
policy:
  stockpile_release: 0.0
  substitution: 0.0
```

---

## Causal identification example

```python
from src.minerals.causal_inference import GraphiteSupplyChainDAG

dag = GraphiteSupplyChainDAG()
result = dag.is_identifiable(treatment="ExportPolicy", outcome="Price")

print(result.identifiable)       # True
print(result.formula)            # "Sum_Z P(Price|ExportPolicy,Z) P(Z)"
print(result.strategy)           # IdentificationStrategy.BACKDOOR
print(result.adjustment_set)     # {"Demand", "Capacity"}
print(result.derivation_steps)   # step-by-step do-calculus trace
```

---

## Deploying to Vercel

The frontend is a standalone Next.js app in `frontend/`. Deploy it to Vercel and point it at your hosted backend:

1. Import `frontend/` as the Vercel root directory
2. Set environment variable `BACKEND_URL=https://your-backend.railway.app`
3. Vercel rewrites `/api/*` → `BACKEND_URL/api/*` — no CORS config needed

The Python backend needs a persistent server (Railway, Fly.io, any VPS). It cannot run as Vercel serverless functions due to heavy ML dependencies.

---

## Documentation

| Doc | Contents |
|-----|----------|
| [docs/CAUSAL_FRAMEWORK.md](docs/CAUSAL_FRAMEWORK.md) | Pearl's ladder, identifiability theory, do-calculus |
| [docs/IMPLEMENTATION_FLOW.md](docs/IMPLEMENTATION_FLOW.md) | End-to-end data flow diagram |
| [docs/KG_TO_CAUSAL.md](docs/KG_TO_CAUSAL.md) | KG → causal DAG pipeline |
| [docs/COMTRADE.md](docs/COMTRADE.md) | UN Comtrade API setup and data ingestion |
| [docs/HIPPORAG.md](docs/HIPPORAG.md) | Graph-based RAG with HippoRAG |
| [docs/VLLM.md](docs/VLLM.md) | Local vLLM server as LLM backend |
| [docs/README_pomdp.md](docs/README_pomdp.md) | POMDP sensor maintenance module |

---

## Dependencies

| Group | Packages |
|-------|----------|
| Causal inference | `dowhy`, `econml`, `networkx`, `pydot` |
| ML / stats | `scikit-learn`, `scipy`, `numpy`, `pandas` |
| LLM | `anthropic`, `openai` (optional) |
| RAG | `sentence-transformers`, `chromadb` (optional), `hipporag` (optional) |
| API | `fastapi`, `uvicorn`, `pydantic` |
| Legacy UI | `gradio` (optional) |

---

## License

MIT
