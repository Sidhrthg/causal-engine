"""
Data ingestion utilities for loading and preprocessing datasets.
"""

from pathlib import Path
from typing import Optional, Dict, Any
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
    
    # Auto-detect file type if not provided
    if file_type is None:
        file_type = path.suffix.lower().lstrip('.')
    
    logger.info(f"Loading dataset from {file_path} (type: {file_type})")
    
    # Load based on file type
    if file_type == 'csv':
        df = pd.read_csv(path, **kwargs)
    elif file_type == 'parquet':
        df = pd.read_parquet(path, **kwargs)
    elif file_type == 'json':
        df = pd.read_json(path, **kwargs)
    else:
        raise ValueError(f"Unsupported file type: {file_type}")
    
    logger.info(f"Loaded dataset with {len(df)} rows and {len(df.columns)} columns")
    
    # Basic validation
    validate_dataframe(df)
    
    return df


def preprocess_dataset(
    df: pd.DataFrame,
    drop_na: bool = True,
    normalize: bool = False
) -> pd.DataFrame:
    """
    Preprocess a dataset for causal analysis.
    
    Args:
        df: Input DataFrame
        drop_na: Whether to drop rows with missing values
        normalize: Whether to normalize numeric columns (TODO: implement)
    
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
    
    # TODO: Implement normalization logic
    if normalize:
        logger.warning("Normalization not yet implemented")
    
    return df_processed

