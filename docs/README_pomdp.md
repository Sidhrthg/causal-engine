# POMDP Module for Sensor Degradation/Maintenance

This module implements a Partially Observable Markov Decision Process (POMDP) for modeling sensor degradation and maintenance decisions.

## Overview

The POMDP framework allows us to:
- Model hidden states (healthy, degrading, failed) of sensors
- Make decisions (ignore, calibrate, repair) based on partial observations
- Learn transition and emission probabilities from historical data
- Simulate maintenance policies and evaluate their performance

## Data Format

### sensor_test_data.csv

The `data/sensor_test_data.csv` file contains synthetic sensor time-series data used to test the POMDP pipeline. The data includes:

- **MaterialType**: Material type identifier (0 or 1)
- **Temperature**: Temperature readings
- **CalibrationInterval**: Calibration interval in days
- **Drift**: Sensor drift measurement
- **Failure**: Binary failure indicator (0 = healthy, 1 = failed)

This is synthetic data generated to simulate sensor behavior over time. Each row represents a sensor reading at a particular point in time.

## Data Processing

### Episode Inference

The system automatically infers episodes from the data:

1. If an `episode_id` column exists, it is used directly
2. Otherwise, episodes are created by:
   - Sorting by timestamp (if available)
   - Splitting when time gap > threshold (default 3600 seconds)
   - Splitting when `sensor_id` changes (if available)

### Observation Discretization

Continuous sensor measurements (temperature, pressure, vibration, etc.) are discretized into a single observation token:

- **Method**: Quantile-based binning (default) or uniform binning
- **Bins**: Default 10 bins per numeric column
- **Deterministic**: Uses a seed for reproducibility
- **Artifacts**: Bin edges are saved to `artifacts/pomdp/bin_edges.json` for reproducibility

The discretization combines multiple numeric columns into a single observation token like `"obs_3_5_2"` representing the bin indices for each feature.

## Hidden States

The system learns or infers hidden states representing sensor health:

- **healthy**: Sensor is functioning normally
- **degrading**: Sensor is showing signs of degradation
- **failed**: Sensor has failed and requires repair

States are inferred from the data if not explicitly provided:
- If a `Failure` column exists, states are mapped directly (0 → healthy, 1 → failed)
- Otherwise, states are inferred heuristically from observation patterns

## Transition and Emission Learning

### Transition Probabilities T[a][s, s']

The transition matrix `T[a]` defines the probability of transitioning from state `s` to state `s'` when action `a` is taken:

- Learned from episode sequences in the data
- Count matrices are smoothed using Dirichlet priors (Laplace smoothing by default)
- Optional priors can be provided via JSON (see Priors section)

### Emission Probabilities Z[a][s, o]

The emission matrix `Z[a]` defines the probability of observing `o` given the next state `s'` and action `a`:

- Learned from observation sequences in the data
- Also smoothed using Dirichlet priors
- Each row sums to 1 (row-stochastic)

## DOT Graph Visualization

The system exports two types of DOT graphs for visualization:

### Transition Graphs (`transitions_{action}.dot`)

These graphs show state-to-state transitions for each action:
- Nodes represent states
- Edges represent transitions with probability labels
- Top-k edges per state are shown (default k=5)
- Minimum probability threshold (default 0.01)

Visualize with: `dot -Tpng transitions_ignore.dot -o transitions_ignore.png`

### Emission Graphs (`emissions_{action}.dot`)

These graphs show state-to-observation emissions for each action:
- Nodes represent states and observations
- Edges show observation likelihoods
- Top-k observations per state are shown

Visualize with: `dot -Tpng emissions_ignore.dot -o emissions_ignore.png`

## Optional LLM-Based Learning

The system supports an optional workflow for incorporating domain knowledge from documents:

### Workflow

1. **Document Ingestion** (`scripts/ingest_docs.py`):
   - Parses PDF/text files (supports PDF, TXT, MD)
   - Chunks text into overlapping segments
   - Builds TF-IDF index for retrieval
   - Stores index in `artifacts/knowledge/index.jsonl`

2. **Priors Generation** (`scripts/build_priors.py`):
   - Retrieves top-k relevant chunks using TF-IDF similarity
   - Uses LLM (MockLLM by default) to extract priors
   - Generates structured priors JSON
   - Validates output against Pydantic schema

3. **Constrained Fitting**:
   - Priors JSON specifies Dirichlet alpha parameters for T/Z matrices
   - Can include action costs and failure penalties
   - Hard constraints can be enforced (e.g., "no direct healthy→failed transition unless action=ignore")

### MockLLM

By default, the system uses a **MockLLM** that:
- Returns valid priors JSON templates
- Populates default values (action costs, failure penalties)
- Requires no network calls or external APIs
- Is fully deterministic

### OpenAI Integration (Optional)

If `OPENAI_API_KEY` is set and `--enable-openai` is passed:
- Uses OpenAI API client (currently a stub)
- Would parse documents to extract actual priors
- In production, would use GPT-4 or similar models

### Auditable Logs

The priors JSON includes metadata:
- Which question/prompt was used
- Which source documents contributed
- Generation method (MockLLM or OpenAI)

This enables full auditability of how priors were generated.

## Usage

### Basic Pipeline

```bash
# Build POMDP from sensor data
python -m scripts.build_pomdp

# Build with custom priors
python -m scripts.build_pomdp --priors configs/pomdp_priors.json
```

### Document Ingestion

```bash
# Ingest documents from a directory
python -m scripts.ingest_docs --path materials/
```

### Generate Priors from Documents

```bash
# Generate priors using MockLLM (default)
python -m scripts.build_priors \
    --question "extract action costs + failure penalties" \
    --out configs/pomdp_priors.json

# Use OpenAI (if API key is set)
python -m scripts.build_priors \
    --question "extract action costs + failure penalties" \
    --out configs/pomdp_priors.json \
    --enable-openai
```

### Full Workflow

```bash
# 1. Ingest documents
python -m scripts.ingest_docs --path materials/

# 2. Generate priors
python -m scripts.build_priors \
    --question "extract action costs + failure penalties" \
    --out configs/pomdp_priors.json

# 3. Build POMDP with priors
python -m scripts.build_pomdp --priors configs/pomdp_priors.json
```

## Output

The pipeline produces:

- **POMDP Model**: Learned transition and emission matrices
- **DOT Graphs**: Visualization files in `artifacts/pomdp/graphs/`
- **Bin Edges**: Discretization parameters in `artifacts/pomdp/bin_edges.json`
- **Summary**: Console output with:
  - Number of states, actions, observations
  - Checksums of T/Z matrices
  - Rollout reward and final belief

## Constraints

- **No Network Calls by Default**: MockLLM requires no external APIs
- **Deterministic**: Uses seeds for reproducibility
- **Clean Typing**: Full type hints throughout
- **Robust Logging**: Structured logging with clear messages

## Testing

Run tests with:

```bash
pytest tests/test_pomdp.py
```

Tests cover:
- Belief update normalization and non-negativity
- Row-stochastic matrix generation
- DOT file export functionality
- Prior integration


