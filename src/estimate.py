"""
Causal effect estimation using DoWhy only.
Clean, warning-free, and with robust confidence interval extraction.
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional, TYPE_CHECKING
import numpy as np
import warnings

# Silence warnings from pandas/dowhy internals
warnings.filterwarnings("ignore")
warnings.simplefilter("ignore", FutureWarning)

if TYPE_CHECKING:
    import pandas as pd

from src.utils.logging_utils import get_logger
from src.utils.data_validation import validate_dataframe
from src.scm import causal_model_from_dag, load_dag_dot

logger = get_logger(__name__)


@dataclass
class EstimationResult:
    ate: float
    ate_ci: Tuple[float, float]
    method: str
    model_summary: Optional[str] = None
    ci_reason: Optional[str] = None


# -------------------------------------------------------------
# CI Extraction (robust against DoWhy formats)
# -------------------------------------------------------------
def _extract_ci(estimate) -> Tuple[float, float, Optional[str]]:
    """
    Supports CI formats:
        - (lower, upper)
        - pandas.DataFrame
        - dict with lower_bound / upper_bound
        - numpy arrays
        - weird nested tuples (DoWhy sometimes does this)
    Returns (lower, upper, reason) where reason is None if successful,
    or a string explaining why CI is unavailable.
    """
    try:
        ci_raw = estimate.get_confidence_intervals()

        # Handle None/empty
        if ci_raw is None:
            return (np.nan, np.nan, "CI unavailable: estimator doesn't provide CI")

        # Already a clean tuple
        if isinstance(ci_raw, tuple) and len(ci_raw) == 2:
            try:
                lower = float(ci_raw[0])
                upper = float(ci_raw[1])
                if np.isnan(lower) or np.isnan(upper):
                    return (np.nan, np.nan, "CI unavailable: CI contains NaN values")
                return (lower, upper, None)
            except (ValueError, TypeError) as e:
                return (np.nan, np.nan, f"CI unavailable: tuple conversion failed: {e}")

        # Pandas DataFrame
        if hasattr(ci_raw, "iloc"):
            try:
                # Use iloc instead of [0] to avoid FutureWarning
                lower = float(ci_raw.iloc[0, 0])
                upper = float(ci_raw.iloc[0, 1])
                if np.isnan(lower) or np.isnan(upper):
                    return (np.nan, np.nan, "CI unavailable: CI DataFrame contains NaN values")
                return (lower, upper, None)
            except (ValueError, TypeError, IndexError) as e:
                return (np.nan, np.nan, f"CI unavailable: DataFrame extraction failed: {e}")

        # Dictionary
        if isinstance(ci_raw, dict):
            try:
                lower = float(ci_raw.get("lower_bound", np.nan))
                upper = float(ci_raw.get("upper_bound", np.nan))
                if np.isnan(lower) or np.isnan(upper):
                    return (np.nan, np.nan, "CI unavailable: CI dict contains NaN values")
                return (lower, upper, None)
            except (ValueError, TypeError) as e:
                return (np.nan, np.nan, f"CI unavailable: dict conversion failed: {e}")

        # Some estimators return ((lower, upper), misc)
        if (
            isinstance(ci_raw, tuple)
            and len(ci_raw) > 0
            and isinstance(ci_raw[0], tuple)
            and len(ci_raw[0]) == 2
        ):
            try:
                lower = float(ci_raw[0][0])
                upper = float(ci_raw[0][1])
                if np.isnan(lower) or np.isnan(upper):
                    return (np.nan, np.nan, "CI unavailable: nested tuple CI contains NaN values")
                return (lower, upper, None)
            except (ValueError, TypeError, IndexError) as e:
                return (np.nan, np.nan, f"CI unavailable: nested tuple extraction failed: {e}")

        # NumPy array
        if isinstance(ci_raw, np.ndarray):
            try:
                if ci_raw.size >= 2:
                    lower = float(ci_raw.flat[0])
                    upper = float(ci_raw.flat[1])
                    if np.isnan(lower) or np.isnan(upper):
                        return (np.nan, np.nan, "CI unavailable: CI array contains NaN values")
                    return (lower, upper, None)
                else:
                    return (np.nan, np.nan, "CI unavailable: CI array has insufficient elements")
            except (ValueError, TypeError) as e:
                return (np.nan, np.nan, f"CI unavailable: array conversion failed: {e}")

        return (np.nan, np.nan, f"CI unavailable: format not recognized ({type(ci_raw)})")

    except AttributeError:
        return (np.nan, np.nan, "CI unavailable: estimator doesn't provide CI method")
    except Exception as e:
        return (np.nan, np.nan, f"CI unavailable: extraction error: {e}")


# -------------------------------------------------------------
# Main ATE estimation function
# -------------------------------------------------------------
def estimate_ate_drl(
    df,
    treatment: str,
    outcome: str,
    controls: List[str],
    graph_dot: str,
) -> EstimationResult:
    """
    Compute ATE using DoWhy's regression estimator.
    No econml, no sklearn encoders, no grouping warnings.
    """

    logger.info(f"Estimating ATE: {treatment} → {outcome}")
    validate_dataframe(df, required_columns=[treatment, outcome] + controls)
    logger.info(f"Data: {len(df)} rows, {len(df.columns)} columns")

    # Build DoWhy model
    causal_model = causal_model_from_dag(
        df=df,
        treatment=treatment,
        outcome=outcome,
        graph_dot=graph_dot,
    )

    # Identify effect
    estimand = causal_model.identify_effect(
        proceed_when_unidentifiable=True
    )

    # Estimate using linear regression
    estimate = causal_model.estimate_effect(
        estimand,
        method_name="backdoor.linear_regression",
        confidence_intervals=True,
        test_significance=True,
    )

    # Extract estimate value
    try:
        ate_value = float(estimate.value)
    except Exception:
        ate_value = np.nan

    # Extract CI
    ate_ci_lower, ate_ci_upper, ci_reason = _extract_ci(estimate)
    ate_ci = (ate_ci_lower, ate_ci_upper)

    if ci_reason is None:
        logger.info(f"ATE: {ate_value:.6f}, CI = [{ate_ci[0]:.6f}, {ate_ci[1]:.6f}]")
    else:
        logger.warning(ci_reason)
        logger.info(f"ATE: {ate_value:.6f}, {ci_reason}")

    summary = (
        f"DoWhy regression | Treatment={treatment}, Outcome={outcome}, "
        f"Controls={controls}, N={len(df)}"
    )

    return EstimationResult(
        ate=ate_value,
        ate_ci=ate_ci,
        method="dowhy_backdoor.linear_regression",
        model_summary=summary,
        ci_reason=ci_reason,
    )


# -------------------------------------------------------------
# Wrapper — load DAG file then estimate
# -------------------------------------------------------------
def estimate_from_dag_path(
    df,
    treatment: str,
    outcome: str,
    controls: List[str],
    dag_path: str,
) -> EstimationResult:

    graph_dot = load_dag_dot(dag_path)

    return estimate_ate_drl(
        df=df,
        treatment=treatment,
        outcome=outcome,
        controls=controls,
        graph_dot=graph_dot,
    )
