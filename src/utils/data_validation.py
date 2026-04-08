"""
Data validation utilities for ensuring data quality and structure.
"""

from typing import List, Dict, Any, Optional
import pandas as pd
import networkx as nx
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def validate_dataframe(
    df: pd.DataFrame,
    required_columns: Optional[List[str]] = None,
    min_rows: int = 1
) -> bool:
    """
    Validate a pandas DataFrame meets basic requirements.
    
    Args:
        df: DataFrame to validate
        required_columns: List of column names that must be present
        min_rows: Minimum number of rows required
    
    Returns:
        True if validation passes, raises ValueError otherwise
    
    Raises:
        ValueError: If validation fails
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame")
    
    if len(df) < min_rows:
        raise ValueError(f"DataFrame must have at least {min_rows} rows")
    
    if required_columns:
        missing = set(required_columns) - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
    
    logger.debug(f"DataFrame validation passed: {len(df)} rows, {len(df.columns)} columns")
    return True


def validate_dag_structure(
    graph: nx.DiGraph,
    required_nodes: Optional[List[str]] = None
) -> bool:
    """
    Validate that a graph is a valid DAG (no cycles).
    
    Args:
        graph: NetworkX directed graph to validate
        required_nodes: List of node names that must be present
    
    Returns:
        True if validation passes, raises ValueError otherwise
    
    Raises:
        ValueError: If validation fails or graph contains cycles
    """
    if not isinstance(graph, nx.DiGraph):
        raise ValueError("Input must be a NetworkX DiGraph")
    
    if not nx.is_directed_acyclic_graph(graph):
        raise ValueError("Graph contains cycles and is not a valid DAG")
    
    if required_nodes:
        missing = set(required_nodes) - set(graph.nodes())
        if missing:
            raise ValueError(f"Missing required nodes: {missing}")
    
    logger.debug(f"DAG validation passed: {len(graph.nodes())} nodes, {len(graph.edges())} edges")
    return True

