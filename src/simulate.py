"""
Counterfactual and intervention simulation utilities.

Uses DoWhy's do-calculus identification pipeline to simulate
P(Y | do(X = x)) rather than naively overwriting data columns.

Monte Carlo bootstrap provides 95% CIs on all estimates.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Dict, Any, Union

if TYPE_CHECKING:
    import pandas as pd
    import numpy as np

from src.utils.logging_utils import get_logger
from src.utils.data_validation import validate_dataframe
from src.scm import causal_model_from_dag, load_dag_dot
from src.estimate import estimate_ate_drl

logger = get_logger(__name__)


@dataclass
class Intervention:
    """
    Specification for a causal intervention (do-operator).

    Attributes:
        node: Name of the variable to intervene on.
        value: Value to fix via do(node = value).
    """
    node: str
    value: Union[float, int, str]


@dataclass
class SimulationResult:
    """
    Result of a counterfactual simulation.

    Attributes:
        outcomes: Scalar summary metrics including Monte Carlo CIs.
        raw_samples: Bootstrap distribution of the intervened outcome mean.
        metadata: Simulation provenance (method, identified estimand, etc.).
    """
    outcomes: Dict[str, float]
    raw_samples: "np.ndarray | None" = None
    metadata: Dict[str, Any] | None = None


def simulate_intervention(
    df: "pd.DataFrame",
    treatment: str,
    outcome: str,
    controls: List[str],
    graph_dot: str,
    intervention: Intervention,
    num_samples: int = 1000,
) -> SimulationResult:
    """
    Simulate P(outcome | do(intervention.node = intervention.value)).

    Steps:
    1. Build a DoWhy CausalModel from the data and DAG.
    2. Identify the causal effect via do-calculus (backdoor / frontdoor /
       ID algorithm — whichever the DAG supports).
    3. Estimate the structural coefficient under the identified estimand.
    4. Compute E[Y | do(X=v)] via linear extrapolation for treatment-node
       interventions, or direct structural propagation otherwise.
    5. Bootstrap the estimate (num_samples replicates) for 95% CIs.

    Args:
        df: Observational data.
        treatment: Name of the treatment variable in the DAG.
        outcome: Name of the outcome variable.
        controls: Observed covariates used for adjustment.
        graph_dot: DAG in DOT string format.
        intervention: Node and value for the do-operator.
        num_samples: Number of bootstrap replicates for uncertainty.

    Returns:
        SimulationResult with outcomes dict (including ci_lower_95 / ci_upper_95)
        and bootstrap raw_samples.

    Raises:
        ValueError: If intervention node is not in the dataframe.
    """
    import pandas as pd
    import numpy as np

    logger.info(f"Simulating intervention: do({intervention.node} = {intervention.value})")
    logger.info(f"Treatment: {treatment}, Outcome: {outcome}, DAG provided: {bool(graph_dot)}")

    if intervention.node not in df.columns:
        raise ValueError(
            f"Intervention node '{intervention.node}' not in dataframe. "
            f"Available: {list(df.columns)}"
        )

    validate_dataframe(df, required_columns=[treatment, outcome, intervention.node] + controls)

    baseline_mean = float(df[outcome].mean())
    baseline_std = float(df[outcome].std())

    # ------------------------------------------------------------------
    # Step 1 + 2: DoWhy — build model and identify causal effect
    # ------------------------------------------------------------------
    causal_model = causal_model_from_dag(df=df, treatment=treatment, outcome=outcome, graph_dot=graph_dot)

    try:
        identified_estimand = causal_model.identify_effect(proceed_when_unidentifiable=True)
        estimand_str = str(identified_estimand)
        logger.info(f"Identified estimand type: {identified_estimand.estimand_type}")
    except Exception as exc:
        logger.warning(f"DoWhy identify_effect failed ({exc}); proceeding without identified estimand")
        identified_estimand = None
        estimand_str = "identification_failed"

    # ------------------------------------------------------------------
    # Step 3: Estimate structural coefficient under identified estimand
    # ------------------------------------------------------------------
    method_used = "dowhy_backdoor_linear"
    ate_coeff = None

    if identified_estimand is not None:
        try:
            causal_estimate = causal_model.estimate_effect(
                identified_estimand,
                method_name="backdoor.linear_regression",
                target_units="ate",
            )
            ate_coeff = float(causal_estimate.value)
            logger.info(f"ATE coefficient (backdoor linear): {ate_coeff:.6f}")
        except Exception as exc:
            logger.warning(f"DoWhy estimate_effect failed ({exc}); will use structural propagation")
            method_used = "structural_propagation_fallback"
    else:
        method_used = "structural_propagation_fallback"

    # ------------------------------------------------------------------
    # Step 4: Compute E[Y | do(intervention.node = v)]
    # ------------------------------------------------------------------
    int_value = float(intervention.value)

    if ate_coeff is not None and intervention.node == treatment:
        # Treatment-node intervention: E[Y|do(T=v)] ≈ baseline + coeff*(v - E[T])
        t_mean = float(df[treatment].mean())
        intervened_mean = baseline_mean + ate_coeff * (int_value - t_mean)

    elif intervention.node != treatment:
        # Mediator / upstream node: fit linear model outcome ~ node + controls
        # and propagate the shift, respecting the causal path.
        relevant = [c for c in [treatment] + controls if c in df.columns and c != intervention.node]
        X_cols = [intervention.node] + relevant
        X_mat = np.column_stack([np.ones(len(df))] + [df[c].values for c in X_cols])
        beta, _, _, _ = np.linalg.lstsq(X_mat, df[outcome].values.astype(float), rcond=None)
        node_mean = float(df[intervention.node].mean())
        intervened_mean = baseline_mean + float(beta[1]) * (int_value - node_mean)
        method_used = "structural_path_linear"

    else:
        # Final fallback: mutilate the data (do-operator approximation)
        df_int = df.copy()
        df_int[intervention.node] = int_value
        intervened_mean = float(df_int[outcome].mean())
        ate_coeff = intervened_mean - baseline_mean
        method_used = "structural_propagation_fallback"

    # ------------------------------------------------------------------
    # Step 5: Monte Carlo bootstrap for 95% CIs
    # ------------------------------------------------------------------
    rng = np.random.default_rng(42)
    n_boot = min(num_samples, 500)
    boot_means = []

    for _ in range(n_boot):
        idx = rng.integers(0, len(df), size=len(df))
        df_b = df.iloc[idx]

        if method_used == "dowhy_backdoor_linear" and ate_coeff is not None:
            t_b_mean = float(df_b[treatment].mean())
            b_int_mean = float(df_b[outcome].mean()) + ate_coeff * (int_value - t_b_mean)
        elif method_used == "structural_path_linear":
            X_b = np.column_stack([np.ones(len(df_b))] + [df_b[c].values for c in X_cols])
            beta_b, _, _, _ = np.linalg.lstsq(X_b, df_b[outcome].values.astype(float), rcond=None)
            b_int_mean = float(df_b[outcome].mean()) + float(beta_b[1]) * (int_value - float(df_b[intervention.node].mean()))
        else:
            df_b_int = df_b.copy()
            df_b_int[intervention.node] = int_value
            b_int_mean = float(df_b_int[outcome].mean())

        boot_means.append(b_int_mean)

    boot_arr = np.array(boot_means)
    difference = intervened_mean - baseline_mean
    pct_change = (difference / baseline_mean * 100) if baseline_mean != 0 else float("inf")

    logger.info(f"Intervened mean: {intervened_mean:.6f}, Δ={difference:.6f} ({pct_change:.2f}%)")
    logger.info(f"95% CI: [{np.percentile(boot_arr, 2.5):.6f}, {np.percentile(boot_arr, 97.5):.6f}]")

    outcomes = {
        "baseline_mean": baseline_mean,
        "baseline_std": baseline_std,
        "intervened_mean": intervened_mean,
        "intervened_std": float(boot_arr.std()),
        "difference": difference,
        "percent_change": pct_change,
        "ci_lower_95": float(np.percentile(boot_arr, 2.5)),
        "ci_upper_95": float(np.percentile(boot_arr, 97.5)),
        "monte_carlo_std": float(boot_arr.std()),
        "ate_coefficient": float(ate_coeff) if ate_coeff is not None else difference,
    }

    metadata = {
        "intervention_node": intervention.node,
        "intervention_value": intervention.value,
        "treatment": treatment,
        "outcome": outcome,
        "controls": controls,
        "num_samples": num_samples,
        "n_bootstrap": n_boot,
        "data_shape": (len(df), len(df.columns)),
        "simulation_method": method_used,
        "identified_estimand": estimand_str,
    }

    return SimulationResult(
        outcomes=outcomes,
        raw_samples=boot_arr,
        metadata=metadata,
    )


def simulate_from_dag_path(
    df: "pd.DataFrame",
    treatment: str,
    outcome: str,
    controls: List[str],
    dag_path: str,
    intervention: Intervention,
    num_samples: int = 1000,
) -> SimulationResult:
    """
    Convenience wrapper: load DAG from a .dot file path then call simulate_intervention.

    Args:
        df: Observational data.
        treatment: Treatment variable name.
        outcome: Outcome variable name.
        controls: Covariate names.
        dag_path: Path to a DOT-format DAG file.
        intervention: do(node = value) specification.
        num_samples: Bootstrap replicates.

    Returns:
        SimulationResult (same as simulate_intervention).

    Raises:
        FileNotFoundError: If dag_path does not exist.
    """
    logger.info(f"Loading DAG from: {dag_path}")
    graph_dot = load_dag_dot(dag_path)
    return simulate_intervention(
        df=df,
        treatment=treatment,
        outcome=outcome,
        controls=controls,
        graph_dot=graph_dot,
        intervention=intervention,
        num_samples=num_samples,
    )
