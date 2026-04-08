# Data Directory

This directory is for storing datasets used in causal analysis.

## Supported Formats

- **CSV** (`.csv`): Comma-separated values
- **Parquet** (`.parquet`): Columnar storage format
- **JSON** (`.json`): JSON format (arrays of objects)

## Usage

Datasets can be loaded using the `load_dataset()` function in `src/ingest.py`:

```python
from src.ingest import load_dataset

# Load CSV
data = load_dataset("data/my_dataset.csv")

# Load Parquet
data = load_dataset("data/my_dataset.parquet", file_type="parquet")

# Load JSON
data = load_dataset("data/my_dataset.json", file_type="json")
```

## Data Requirements

For causal analysis, datasets should contain:
- Treatment variable(s)
- Outcome variable(s)
- Control/confounding variables (if applicable)
- Sufficient sample size for reliable estimation

## Example Structure

```
data/
  ├── experiment_data.csv
  ├── observational_study.parquet
  └── synthetic_data.json
```

