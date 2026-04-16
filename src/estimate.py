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
# Bootstrap CI via plain OLS (pandas-version-independent)
# -------------------------------------------------------------
def _bootstrap_ci_ols(
    df,
    treatment: str,
    outcome: str,
    controls: List[str],
    n_boot: int = 500,
    alpha: float = 0.05,
) -> Tuple[float, float, Optional[str]]:
    """
    Compute 95% CI for the ATE using bootstrap OLS, bypassing DoWhy's
    CI machinery which breaks on pandas 3.0.

    Model: outcome ~ 1 + treatment + controls (OLS).
    Coefficient on treatment = ATE under backdoor adjustment.
    Returns (lower, upper, reason) matching _extract_ci convention.
    """
    regressors = [treatment] + [c for c in controls if c in df.columns and c != treatment]
    needed = [outcome] + regressors
    missing = [c for c in needed if c not in df.columns]
    if missing:
        return (np.nan, np.nan, f"CI unavailable: missing columns {missing}")

    sub = df[needed].dropna()
    if len(sub) < len(regressors) + 2:
        return (np.nan, np.nan, "CI unavailable: too few observations")

    def _ols_ate(sample: "pd.DataFrame") -> float:
        y = sample[outcome].values.astype(float)
        X = np.column_stack([np.ones(len(sample))] + [sample[c].values.astype(float) for c in regressors])
        try:
            beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
            return float(beta[1])  # coefficient on treatment
        except Exception:
            return np.nan

    rng = np.random.default_rng(42)
    boot_ates = []
    n = len(sub)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot_ates.append(_ols_ate(sub.iloc[idx]))

    boot_arr = np.array([v for v in boot_ates if not np.isnan(v)])
    if len(boot_arr) < 10:
        return (np.nan, np.nan, "CI unavailable: too many failed bootstrap resamples")

    lo = float(np.percentile(boot_arr, 100 * alpha / 2))
    hi = float(np.percentile(boot_arr, 100 * (1 - alpha / 2)))
    return (lo, hi, None)


# -------------------------------------------------------------
# Main ATE estimation function
# -------------------------------------------------------------
def estimate_ate_drl(
    df,
    treatment: str,
    outcome: str,
    controls: List[str],
    graph_dot: str | None = None,
    graph_path: str | None = None,
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
        graph_path=graph_path,
    )

    # Identify the causal effect to get the backdoor adjustment set from the DAG.
    estimand = causal_model.identify_effect(
        proceed_when_unidentifiable=True
    )

    # DoWhy's estimate_effect / backdoor.linear_regression triggers an internal
    # pandas groupby.apply() that crashes on pandas 3.0 (KeyError: 0 from
    # index.get_loc inside LinearRegressionEstimator._estimate_effect_fn).
    # Workaround: extract the DoWhy-identified adjustment set, then run our own
    # OLS.  This is numerically identical to DoWhy's regression estimator but
    # entirely pandas-version-independent.
    try:
        adj_vars = list(estimand.backdoor_variables or [])
    except Exception:
        adj_vars = []

    # Fall back to the caller-supplied controls if DoWhy's identification
    # didn't produce an adjustment set (unidentifiable graph, empty DAG, etc.)
    adjustment_controls = adj_vars if adj_vars else controls

    # ATE via OLS: outcome ~ 1 + treatment + adjustment_controls
    # Coefficient on treatment = backdoor-adjusted ATE.
    regressors = [treatment] + [c for c in adjustment_controls if c in df.columns and c != treatment]
    needed = [outcome] + regressors
    sub = df[[c for c in needed if c in df.columns]].dropna()

    if len(sub) < len(regressors) + 2:
        ate_value = np.nan
        logger.warning("Too few observations for OLS ATE estimation")
    else:
        y = sub[outcome].values.astype(float)
        X = np.column_stack([np.ones(len(sub))] + [sub[c].values.astype(float) for c in regressors])
        try:
            beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
            ate_value = float(beta[1])
        except Exception as e:
            ate_value = np.nan
            logger.warning(f"OLS ATE failed: {e}")

    # Bootstrap 95% CI using the same OLS model.
    ate_ci_lower, ate_ci_upper, ci_reason = _bootstrap_ci_ols(
        df=df,
        treatment=treatment,
        outcome=outcome,
        controls=adjustment_controls,
        n_boot=500,
    )
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
    return estimate_ate_drl(
        df=df,
        treatment=treatment,
        outcome=outcome,
        controls=controls,
        graph_path=dag_path,
    )
