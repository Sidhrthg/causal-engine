# Implementation Flow

How the Critical Minerals Causal Engine is wired end-to-end: Gradio → handlers → backends → outputs.

---

## 1. Entry point

- **Run:** `python app.py` → starts Gradio on `http://127.0.0.1:7860`.
- **UI:** `app.py` builds a `gr.Blocks` with multiple tabs. Shared state: `run_dir_in` (textbox) is updated by Run Scenario and Unified Workflow so Validate with RAG gets the latest run path.

---

## 2. Tab → handler → backend (by tab)

| Tab | User action | Handler (app.py) | Backend / subprocess |
|-----|-------------|------------------|----------------------|
| **Validate with RAG** | Enter run dir, optional year → Run Validation | `validate_with_rag(run_dir, year)` | `scripts/validate_with_real_rag.py --run-dir <path> [--year N]` |
| **Unified Workflow** | Pick domain (Mineral / Sensor), scenario or paths → Run full pipeline | `run_unified(domain, scenario, sensor_data, sensor_priors)` | Mineral: `run_full_pipeline_mineral(scenario)` → causal_inference, run_scenario, test_synthetic_control. Sensor: `run_full_pipeline_sensor()` → POMDP + sensor causal. Outputs also update `run_dir_in`. |
| **Query Model** | Natural-language question → Run Simulation | `query_model(query)` | `scripts.llm_query` (LLM generates scenario YAML, then `scripts.run_scenario`) |
| **Just RAG** | Upload/save, Reindex, then Search (optional: Include KG context) | `save_uploaded_documents`, `reindex_rag`, `rag_search(query, top_k, use_kg_context)` | `src.minerals.rag_retrieval.SimpleRAGRetriever`; if `use_kg_context`: `get_kg_context_for_rag()` → KG summary + edges prepended to results |
| **Causal Analysis & DAG** | Run Identifiability (default DAG) | `show_causal_analysis()` | `python -m src.minerals.causal_inference` (prints do-calculus rules + derivations, identifiability, param strategies; optional `--plot` → `graphite_causal_dag.png`) |
| **Causal Analysis & DAG** | Run Identifiability (KG DAG) | `run_kg_identifiability()` | In-process: `_get_kg()` → `kg.to_causal_dag()` → `dag.is_identifiable(...)` for fixed queries; uses `do_calculus` for derivation steps; returns markdown with formula + derivation |
| **Run Scenario** | Pick scenario → Run | `run_scenario_tab(scenario_name)` | `scripts.run_scenario --scenario scenarios/<name>` → `src.minerals.simulate.run_scenario(cfg)`; parses "Outputs: <path>" → updates `run_dir_in` |
| **Synthetic Control** | Run Synthetic Control | `run_synthetic_control()` | `scripts/test_synthetic_control.py` (Comtrade data) |
| **POMDP** | Build / POMDP + Causal | `run_pomdp_build`, `run_pomdp_and_causal` | `scripts.build_pomdp`; sensor causal: `src.estimate.estimate_from_dag_path` with `dag_registry/sensor_reliability.dot` |
| **Knowledge Graph** | Build KG, Propagate shock, Run identifiability, Show DAG edges/graph | `get_kg_summary`, `get_kg_shock_sources`, `run_kg_shock_propagation`, `run_kg_identifiability`, `get_kg_dag_edges`, `get_kg_dag_image` | All in-process: `src.minerals.knowledge_graph.build_critical_minerals_kg()`; `kg.to_causal_dag()`, `dag.visualize()` for image |
| **Causal Discovery** | (Instructions only) | — | User runs `python -m src.minerals.causal_discovery` in terminal; writes `dag_registry/discovered_graphite_causal_structure.json` |

---

## 3. Main backend flows

### 3.1 Causal identifiability (do-calculus)

- **Default DAG:** `src.minerals.causal_inference`: `GraphiteSupplyChainDAG()` (hard-coded graph). `demonstrate_identifiability()` prints do-calculus rules (from `do_calculus`), then for each (treatment, outcome): `dag.is_identifiable(treatment, outcome)` → backdoor/frontdoor/d-separation → `IdentificationResult` (formula, strategy, adjustment_set, assumptions, **derivation_steps**). Derivation steps come from `do_calculus.derivation_steps_for_result(...)`.
- **KG DAG:** Same `is_identifiable` API; DAG comes from `build_critical_minerals_kg().to_causal_dag()` (CAUSES-only subgraph). No subprocess; app builds KG, gets DAG, runs same identifiability + derivations.

### 3.2 Scenario run (mineral)

- **Flow:** `scenarios/<name>.yaml` → `src.minerals.schema.load_scenario()` → `ScenarioConfig` → `scripts.run_scenario` → `src.minerals.simulate.run_scenario(cfg)` → `model.step()` in `model.py` (state + shocks) → `timeseries.csv` + `metrics.json` in `runs/<scenario_name>/<timestamp>/`. Script prints "Outputs: <run_dir>"; app parses that and sets `run_dir_in`.

### 3.3 Validate with RAG

- **Flow:** `validate_with_real_rag.py` loads run’s `timeseries.csv`, Comtrade data, compares; `_retrieve_relevant_context()` uses `SimpleRAGRetriever`; builds KG context (summary + causal edges) and injects into prompt; `_generate_rag_analysis_with_retrieval()` sends prompt (model summary, actual data, comparison, **KG context**, retrieved docs) to LLM; report printed to stdout → Gradio shows it.

### 3.4 RAG search (Just RAG)

- **Flow:** `SimpleRAGRetriever(documents_dir, index_path)` loads `data/documents/index.json` and chunks; `retriever.retrieve(query, top_k)`; if `use_kg_context`, `get_kg_context_for_rag()` (KG summary + edges from `to_causal_dag()`) is prepended to the markdown list of chunks.

### 3.5 Knowledge Graph

- **Build:** `build_critical_minerals_kg()` (entities + relations: CAUSES, PRODUCES, etc.). Summary and `get_shock_origin_candidates()` for dropdown.
- **Shock:** `kg.propagate_shock(origin_id)` → BFS with decay → `ShockTrace` (affected, paths) → formatted markdown.
- **Identifiability:** `kg.to_causal_dag()` → `dag.is_identifiable(...)` (same as Causal Analysis KG DAG); derivation from `do_calculus`.
- **DAG edges / image:** `kg.to_causal_dag()` → list of edges or `dag.visualize("kg_causal_dag.png")` → path returned to `gr.Image`.

---

## 4. Key modules and dependencies

```
app.py
├── _run()                    # subprocess helper
├── Query → scripts.llm_query → scripts.run_scenario
├── Just RAG → rag_retrieval.SimpleRAGRetriever; get_kg_context_for_rag() → knowledge_graph
├── Validate with RAG → scripts.validate_with_real_rag (uses RAG + KG in prompt)
├── Causal Analysis → causal_inference (CLI) or run_kg_identifiability() (KG + causal_inference + do_calculus)
├── Run Scenario → scripts.run_scenario → simulate.run_scenario
├── Unified Workflow → run_full_pipeline_mineral / run_full_pipeline_sensor
├── KG tab → knowledge_graph (build_critical_minerals_kg, to_causal_dag, propagate_shock, get_shock_origin_candidates)
├── POMDP → scripts.build_pomdp; sensor causal → src.estimate.estimate_from_dag_path
└── Synthetic Control → scripts.test_synthetic_control

src/minerals/
├── causal_inference.py   # CausalDAG, GraphiteSupplyChainDAG, is_identifiable (uses do_calculus for derivation_steps)
├── do_calculus.py        # Three rules, derivation_steps_backdoor/frontdoor/trivial, derivation_steps_for_result
├── knowledge_graph.py    # CausalKnowledgeGraph, build_critical_minerals_kg, to_causal_dag, propagate_shock
├── simulate.py           # run_scenario(cfg) → model.step
├── model.py              # State, step (dynamics)
├── schema.py             # ScenarioConfig, load_scenario
├── rag_retrieval.py      # SimpleRAGRetriever
└── ...
```

---

## 5. Data flow highlights

- **run_dir:** Produced by `scripts.run_scenario` (printed "Outputs: ..."). Consumed by Validate with RAG (and optionally by user in other tabs). Updated in Gradio via `run_scenario_tab` and `run_unified` outputs → `run_dir_in`.
- **KG:** Built on demand in app (`_get_kg()`). Used by: KG tab (summary, shock, identifiability, DAG edges/image), Just RAG (optional context), Validate with RAG (KG context in prompt), Causal Analysis (KG DAG button).
- **Do-calculus:** `do_calculus.py` defines rules and derivation steps. `causal_inference.is_identifiable()` returns `IdentificationResult` with `derivation_steps` filled by `do_calculus.derivation_steps_for_result()`. CLI and Gradio (KG identifiability output) show formula + derivation.

---

## 6. Summary

- **Gradio** is the single UI; each tab maps to one or more handlers in `app.py`.
- **Subprocesses:** causal_inference (default DAG), run_scenario, validate_with_real_rag, llm_query, build_pomdp, test_synthetic_control, index_rag_documents.
- **In-process:** KG build, KG→DAG identifiability, RAG search, sensor causal estimation; all use `PROJECT_ROOT` and shared helpers (`_get_kg`, `get_kg_context_for_rag`, etc.).
- **Identifiability** is algebra-first: do-calculus rules and derivation steps are in `do_calculus.py`, attached to every identification result, and shown in CLI and in the Causal Analysis / KG identifiability outputs in Gradio.
