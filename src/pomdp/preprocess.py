"""
Data preprocessing for POMDP learning.
"""

from typing import List, Tuple, Optional, Dict, Any
import pandas as pd
import numpy as np
from pathlib import Path

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def load_sensor_csv(path: str) -> pd.DataFrame:
    """
    Load sensor data from CSV file.
    
    Args:
        path: Path to CSV file
        
    Returns:
        DataFrame with sensor data
    """
    logger.info(f"Loading sensor data from {path}")
    df = pd.read_csv(path)
    logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")
    return df


def infer_episodes(
    df: pd.DataFrame,
    episode_id_col: Optional[str] = None,
    timestamp_col: Optional[str] = None,
    sensor_id_col: Optional[str] = None,
    time_gap_threshold: float = 3600.0,  # seconds
    rng: Optional[np.random.Generator] = None,
) -> pd.DataFrame:
    """
    Infer episode boundaries from dataframe.
    
    If episode_id_col exists, use it directly.
    Otherwise, create episodes by:
        - Sorting by timestamp (if available)
        - Splitting when time gap > threshold OR sensor_id changes
    
    Args:
        df: Input dataframe
        episode_id_col: Name of episode ID column (if exists)
        timestamp_col: Name of timestamp column (optional)
        sensor_id_col: Name of sensor ID column (optional)
        time_gap_threshold: Maximum time gap within an episode (seconds)
        rng: Random number generator for reproducibility
        
    Returns:
        DataFrame with added 'episode_id' column
    """
    df = df.copy()
    
    # If episode_id already exists, use it
    if episode_id_col and episode_id_col in df.columns:
        logger.info(f"Using existing episode_id column: {episode_id_col}")
        df["episode_id"] = df[episode_id_col]
        return df
    
    # Infer episodes
    logger.info("Inferring episode boundaries")
    
    # Try to find timestamp column
    if timestamp_col and timestamp_col in df.columns:
        df = df.sort_values(timestamp_col)
        timestamps = pd.to_datetime(df[timestamp_col])
        time_diffs = timestamps.diff().dt.total_seconds()
    else:
        # No timestamp, use row index as proxy
        logger.warning("No timestamp column found, using row order")
        timestamps = None
        time_diffs = None
    
    # Initialize episode IDs
    episode_ids = np.zeros(len(df), dtype=int)
    current_episode = 0
    
    for i in range(len(df)):
        if i == 0:
            episode_ids[i] = current_episode
            continue
        
        # Check for sensor_id change
        if sensor_id_col and sensor_id_col in df.columns:
            if df.iloc[i][sensor_id_col] != df.iloc[i-1][sensor_id_col]:
                current_episode += 1
                episode_ids[i] = current_episode
                continue
        
        # Check for time gap
        if time_diffs is not None:
            if time_diffs.iloc[i] > time_gap_threshold:
                current_episode += 1
                episode_ids[i] = current_episode
                continue
        
        # Same episode
        episode_ids[i] = current_episode
    
    df["episode_id"] = episode_ids
    n_episodes = len(df["episode_id"].unique())
    logger.info(f"Created {n_episodes} episodes")
    
    return df


def discretize_observations(
    df: pd.DataFrame,
    numeric_cols: Optional[List[str]] = None,
    n_bins: int = 10,
    method: str = "quantile",
    seed: Optional[int] = None,
    bin_edges_path: Optional[str] = None,
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    """
    Discretize numeric columns into a single observation token.
    
    Uses quantile-based binning by default for reproducibility.
    
    Args:
        df: Input dataframe
        numeric_cols: List of numeric column names to use (None = auto-detect)
        n_bins: Number of bins per column
        method: Binning method ("quantile" or "uniform")
        seed: Random seed for reproducibility (affects tie-breaking)
        bin_edges_path: Path to save/load bin edges JSON
        
    Returns:
        Tuple of (dataframe with 'observation' column, bin_edges dict)
    """
    df = df.copy()
    
    # Auto-detect numeric columns if not specified
    if numeric_cols is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        # Exclude common non-feature columns
        exclude = ["episode_id", "Failure", "failure", "label", "target"]
        numeric_cols = [c for c in numeric_cols if c not in exclude]
        logger.info(f"Auto-detected numeric columns: {numeric_cols}")
    
    if not numeric_cols:
        raise ValueError("No numeric columns found for discretization")
    
    # Load existing bin edges if provided
    bin_edges = {}
    if bin_edges_path and Path(bin_edges_path).exists():
        import json
        logger.info(f"Loading bin edges from {bin_edges_path}")
        with open(bin_edges_path, "r") as f:
            bin_edges_data = json.load(f)
            bin_edges = {k: np.array(v) for k, v in bin_edges_data.items()}
    
    # Compute bin edges for each column
    for col in numeric_cols:
        if col not in bin_edges:
            values = df[col].dropna()
            if method == "quantile":
                # Use quantiles
                quantiles = np.linspace(0, 100, n_bins + 1)
                edges = np.percentile(values, quantiles)
                # Ensure unique edges
                edges = np.unique(edges)
                if len(edges) < 2:
                    edges = np.array([values.min() - 1e-6, values.max() + 1e-6])
                bin_edges[col] = edges
            else:  # uniform
                edges = np.linspace(values.min(), values.max(), n_bins + 1)
                bin_edges[col] = edges
            logger.info(f"Computed {len(bin_edges[col])-1} bins for {col}")
    
    # Discretize each column
    digitized_cols = []
    for col in numeric_cols:
        digitized = np.digitize(df[col].fillna(df[col].median()), bin_edges[col])
        digitized_cols.append(digitized)
    
    # Combine into single observation token
    # Use a simple encoding: (bin1, bin2, ...) -> "obs_{bin1}_{bin2}_..."
    obs_tokens = []
    for i in range(len(df)):
        tokens = [str(int(d[i])) for d in digitized_cols]
        obs_tokens.append("_".join(tokens))
    
    df["observation"] = obs_tokens
    
    # Save bin edges if path provided
    if bin_edges_path:
        import json
        Path(bin_edges_path).parent.mkdir(parents=True, exist_ok=True)
        bin_edges_serializable = {k: v.tolist() for k, v in bin_edges.items()}
        with open(bin_edges_path, "w") as f:
            json.dump(bin_edges_serializable, f, indent=2)
        logger.info(f"Saved bin edges to {bin_edges_path}")
    
    n_unique_obs = df["observation"].nunique()
    logger.info(f"Created {n_unique_obs} unique observations")
    
    return df, bin_edges


