# Causal Engine

A causal modeling engine for effect estimation and policy simulation.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

**Note:** `pydot<4` is pinned due to NetworkX DOT parsing incompatibility with pydot 4.x.

## Quickstart (graphite baseline)

```bash
source .venv/bin/activate
python -m pip install -U pip
python -m pip install pydantic pyyaml pandas numpy
python -m scripts.run_scenario --scenario scenarios/graphite_baseline.yaml
python -m pytest -q
```

## Tests

```bash
source .venv/bin/activate
python -m pytest -q
```

