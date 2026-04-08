# Repository Map

## Entry Points / Scripts

- `scripts/build_pomdp.py` - POMDP sensor degradation/maintenance pipeline
- `scripts/build_priors.py` - LLM-based priors generation (MockLLM default)
- `scripts/ingest_docs.py` - Document ingestion with TF-IDF indexing
- `scripts/run_scenario.py` - Main CLI entrypoint for minerals modeling scenarios

## Main Modules

### Core Causal Modeling (src/)
- `src/estimate.py` - ATE estimation using DoWhy
- `src/simulate.py` - Intervention simulation
- `src/scm.py` - Causal model construction from DAG
- `src/ingest.py` - Data loading utilities
- `src/config.py` - Configuration management
- `src/api.py` - FastAPI REST API

### POMDP Module (src/pomdp/) - Separate
- Sensor degradation/maintenance modeling
- POMDP learning, belief updates, policies, simulation

### Utilities (src/utils/)
- `data_validation.py` - DataFrame/DAG validation
- `logging_utils.py` - Logging setup

## Data Flow

```
Data (CSV/Parquet/JSON)
  ↓
src/ingest.load_dataset()
  ↓
src/estimate.estimate_ate_drl() or src/simulate.simulate_intervention()
  ↓
Results (EstimationResult/SimulationResult)
```

## Current Pipeline

**Minerals Modeling Pipeline**: Not yet implemented (foundation in progress)
- Planned structure: `data_ingest/ | model/ | simulation/ | scenarios/ | evaluation/`

**Existing Pipelines**:
- Causal estimation: `src/estimate.py` → DoWhy → ATE results
- POMDP sensor: `scripts/build_pomdp.py` → separate module

