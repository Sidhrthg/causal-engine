"""Data ingestion and normalization utilities."""

from pathlib import Path
from typing import Optional, Dict, Any, List
import numpy as np
import pandas as pd
from src.config import Config
from src.utils.logging_utils import get_logger
from src.utils.data_validation import validate_dataframe

logger = get_logger(__name__)


def load_dataset(
    file_path: str,
    file_type: Optional[str] = None,
    **kwargs
) -> pd.DataFrame:
    """
    Load a dataset from a file path.

    Supports CSV, Parquet, and JSON formats.

    Args:
        file_path: Path to the dataset file
        file_type: File type ('csv', 'parquet', 'json'). Auto-detected if None
        **kwargs: Additional arguments passed to pandas read functions

    Returns:
        Loaded DataFrame

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file type is unsupported
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {file_path}")

    if file_type is None:
        file_type = path.suffix.lower().lstrip('.')

    logger.info(f"Loading dataset from {file_path} (type: {file_type})")

    if file_type == 'csv':
        df = pd.read_csv(path, **kwargs)
    elif file_type == 'parquet':
        df = pd.read_parquet(path, **kwargs)
    elif file_type == 'json':
        df = pd.read_json(path, **kwargs)
    else:
        raise ValueError(f"Unsupported file type: {file_type}")

    logger.info(f"Loaded dataset with {len(df)} rows and {len(df.columns)} columns")
    validate_dataframe(df)
    return df


def preprocess_dataset(
    df: pd.DataFrame,
    drop_na: bool = True,
    normalize: bool = False,
    normalize_method: str = "zscore",
    exclude_columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Preprocess a dataset for causal analysis.

    Args:
        df: Input DataFrame
        drop_na: Whether to drop rows with missing values
        normalize: Whether to normalize numeric columns
        normalize_method: ``"zscore"`` (subtract mean, divide by std) or
            ``"minmax"`` (scale to [0, 1]).  Columns with zero variance are
            left unchanged.
        exclude_columns: Column names to skip during normalization (e.g.
            binary treatment indicators or ID columns).

    Returns:
        Preprocessed DataFrame
    """
    logger.info("Preprocessing dataset")
    df_processed = df.copy()

    if drop_na:
        initial_rows = len(df_processed)
        df_processed = df_processed.dropna()
        dropped = initial_rows - len(df_processed)
        if dropped > 0:
            logger.warning(f"Dropped {dropped} rows with missing values")

    if normalize:
        skip = set(exclude_columns or [])
        numeric_cols = [
            c for c in df_processed.select_dtypes(include=[np.number]).columns
            if c not in skip
        ]
        if not numeric_cols:
            logger.warning("normalize=True but no numeric columns found to normalize")
        else:
            if normalize_method == "zscore":
                means = df_processed[numeric_cols].mean()
                stds = df_processed[numeric_cols].std(ddof=0)
                non_constant = stds[stds > 0].index.tolist()
                df_processed[non_constant] = (
                    df_processed[non_constant] - means[non_constant]
                ) / stds[non_constant]
                logger.info(
                    f"Z-score normalized {len(non_constant)} columns "
                    f"(skipped {len(numeric_cols) - len(non_constant)} constant)"
                )
            elif normalize_method == "minmax":
                mins = df_processed[numeric_cols].min()
                maxs = df_processed[numeric_cols].max()
                ranges = maxs - mins
                non_constant = ranges[ranges > 0].index.tolist()
                df_processed[non_constant] = (
                    df_processed[non_constant] - mins[non_constant]
                ) / ranges[non_constant]
                logger.info(
                    f"Min-max normalized {len(non_constant)} columns "
                    f"(skipped {len(numeric_cols) - len(non_constant)} constant)"
                )
            else:
                raise ValueError(
                    f"Unknown normalize_method '{normalize_method}'. "
                    "Use 'zscore' or 'minmax'."
                )

    return df_processed
