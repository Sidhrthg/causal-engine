# DAG Registry

This directory contains registered Directed Acyclic Graphs (DAGs) representing causal structures.

## File Formats

- **`.dot`**: Graphviz DOT format for visualization
- **`.json`**: NetworkX node-link JSON format for programmatic loading

## Example DAGs

- `sensor_reliability.dot/json`: Example DAG for sensor reliability analysis
- `site_uptime.dot/json`: Example DAG for site uptime analysis

## Usage

DAGs can be loaded using the `load_dag_from_file()` function in `src/scm.py`:

```python
from src.scm import load_dag_from_file

# Load from JSON
graph = load_dag_from_file("dag_registry/sensor_reliability.json", format="json")

# Load from DOT
graph = load_dag_from_file("dag_registry/sensor_reliability.dot", format="dot")
```

