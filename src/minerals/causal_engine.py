"""
Three-layer causal inference engine implementing Pearl's Ladder of Causation.

Regular probability only covers Layer 1 (observation/association). Causal
reasoning requires two additional layers that probability alone cannot reach:

  Layer 1 — Association (Seeing):      P(Y|X)          "What if I see X?"
  Layer 2 — Intervention (Doing):      P(Y|do(X))      "What if I do X?"
  Layer 3 — Counterfactual (Imagining): P(Y_x | X',Y') "What if X had been x?"

The Causal Hierarchy Theorem (Bareinboim et al., 2022) proves these form a
strict hierarchy: each layer can answer questions the layer below cannot.

Reference: Pearl, J. (2009). Causality: Models, Reasoning, and Inference.
"""

from __future__ import annotations

import copy
import dataclasses
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from .causal_inference import CausalDAG, IdentificationResult
from .model import State, StepResult, step
from .schema import ScenarioConfig, ShockConfig
from .shocks import ShockSignals, shocks_for_year
from .simulate import run_scenario


# ============================================================================
# Result dataclasses
# ============================================================================


@dataclass(frozen=True)
class CorrelationMatrix:
    """Pairwise correlation matrix with significance tests."""

    variables: List[str]
    pearson: np.ndarray  # (n, n) correlation coefficients
    p_values: np.ndarray  # (n, n) two-sided p-values
    spearman: Optional[np.ndarray] = field(default=None, repr=False)
    spearman_p_values: Optional[np.ndarray] = field(default=None, repr=False)


@dataclass(frozen=True)
class ConditionalDistribution:
    """P(Y|X=x) estimated from observational data."""

    outcome: str
    conditioning: Dict[str, Any]
    mean: float
    std: float
    count: int
    values: np.ndarray = field(repr=False)
    quantiles: Dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class IndependenceTestResult:
    """Result of conditional independence test X _||_ Y | Z."""

    x: str
    y: str
    z: List[str]
    test_statistic: float
    p_value: float
    independent: bool  # True if p_value > alpha
    method: str
    alpha: float


@dataclass(frozen=True)
class AssociationResult:
    """Regression-based association E[Y|X, controls] (Layer 1 only — not causal)."""

    outcome: str
    predictors: List[str]
    coefficients: Dict[str, float]
    std_errors: Dict[str, float]
    r_squared: float
    residuals: np.ndarray = field(repr=False)


@dataclass(frozen=True)
class BackdoorEstimate:
    """P(Y|do(X)) estimated via backdoor adjustment from data."""

    treatment: str
    outcome: str
    adjustment_set: Set[str]
    ate: float  # Average Treatment Effect
    ate_std_error: float
    ate_ci: Tuple[float, float]  # 95% confidence interval
    identification: Optional[IdentificationResult] = None


@dataclass(frozen=True)
class SimulationDoResult:
    """P(Y|do(X=x)) estimated by running the structural model with intervention."""

    treatment_var: str
    treatment_value: float
    baseline_trajectory: pd.DataFrame = field(repr=False)
    intervention_trajectory: pd.DataFrame = field(repr=False)
    effect_on_outcome: Dict[str, float]
    metrics_baseline: Dict[str, float]
    metrics_intervention: Dict[str, float]


@dataclass(frozen=True)
class AbductionResult:
    """Inferred exogenous noise from observed trajectory (Pearl Step 1)."""

    years: List[int]
    inferred_noise: Dict[int, float]  # year -> epsilon that rationalizes observed P
    fit_error: float  # RMSE of reproduced vs observed prices


@dataclass(frozen=True)
class CounterfactualResult:
    """Full 3-step counterfactual: abduction -> action -> prediction."""

    factual_trajectory: pd.DataFrame = field(repr=False)
    counterfactual_trajectory: pd.DataFrame = field(repr=False)
    effect: pd.DataFrame = field(repr=False)  # per-year delta for each variable
    abduction: AbductionResult
    do_overrides: Dict[int, Dict[str, float]]
    summary: Dict[str, float]


@dataclass(frozen=True)
class CounterfactualTrajectoryResult:
    """Multi-step counterfactual trajectory comparison."""

    factual_states: List[State]
    factual_results: List[StepResult]
    counterfactual_states: List[State]
    counterfactual_results: List[StepResult]
    years: List[int]
    deltas: Dict[str, List[float]]


# ============================================================================
# Helpers
# ============================================================================

# Mapping from DAG node names to shock config fields for the do() operator.
# Each entry maps a causal DAG node to (shock_type, magnitude_field).
#
# Supports both canonical GraphiteSupplyChainDAG names and KG entity names
# (which come from CausalDiscoveryAgent or build_critical_minerals_kg()).
# Fuzzy matching via _resolve_treatment_node() handles free-text LLM-extracted names.
_NODE_TO_SHOCK: Dict[str, Tuple[str, str]] = {
    # ---- Canonical GraphiteSupplyChainDAG variables ----
    "ExportPolicy":        ("export_restriction", "magnitude"),
    "Demand":              ("demand_surge", "magnitude"),
    "Capacity":            ("capacity_reduction", "magnitude"),
    "Supply":              ("policy_shock", "magnitude"),
    "Inventory":           ("stockpile_release", "magnitude"),
    "GlobalDemand":        ("macro_demand_shock", "magnitude"),
    "TradeValue":          ("export_restriction", "magnitude"),

    # ---- Supply-side interventions ----
    "ExportQuota":              ("export_restriction", "magnitude"),
    "ExportRestriction":        ("export_restriction", "magnitude"),
    "ExportControl":            ("export_restriction", "magnitude"),
    "TradeRestriction":         ("export_restriction", "magnitude"),
    "LicensingRequirement":     ("export_restriction", "magnitude"),
    "ExportLicense":            ("export_restriction", "magnitude"),
    "ExportBan":                ("export_restriction", "magnitude"),
    "Tariff":                   ("export_restriction", "magnitude"),
    "Sanctions":                ("export_restriction", "magnitude"),
    "TradeBarrier":             ("export_restriction", "magnitude"),

    # ---- Production / capacity interventions ----
    "ProductionCapacity":       ("capacity_reduction", "magnitude"),
    "MiningCapacity":           ("capacity_reduction", "magnitude"),
    "CapacityConstraint":       ("capacity_reduction", "magnitude"),
    "EnvironmentalRegulation":  ("capacity_reduction", "magnitude"),
    "EnvironmentalCompliance":  ("capacity_reduction", "magnitude"),
    "MineShutdown":             ("capacity_reduction", "magnitude"),
    "ProductionCutback":        ("capacity_reduction", "magnitude"),
    "CapacityReduction":        ("capacity_reduction", "magnitude"),
    "ProcessingCapacity":       ("capacity_reduction", "magnitude"),
    "RefiningCapacity":         ("capacity_reduction", "magnitude"),

    # ---- Policy / quota interventions ----
    "PolicyRestriction":        ("policy_shock", "magnitude"),
    "ExportQuotaPolicy":        ("policy_shock", "magnitude"),
    "TradePolicy":              ("policy_shock", "magnitude"),
    "GovernmentPolicy":         ("policy_shock", "magnitude"),
    "QuotaReduction":           ("policy_shock", "magnitude"),

    # ---- Demand-side interventions ----
    "DemandGrowth":             ("demand_surge", "magnitude"),
    "IndustrialDemand":         ("demand_surge", "magnitude"),
    "BatteryDemand":            ("demand_surge", "magnitude"),
    "EVDemand":                 ("demand_surge", "magnitude"),
    "SteelDemand":              ("demand_surge", "magnitude"),
    "DemandShock":              ("demand_surge", "magnitude"),
    "ConsumptionGrowth":        ("demand_surge", "magnitude"),
    "TechDemand":               ("demand_surge", "magnitude"),

    # ---- Macro / systemic shocks ----
    "MacroShock":               ("macro_demand_shock", "magnitude"),
    "FinancialCrisis":          ("macro_demand_shock", "magnitude"),
    "EconomicRecession":        ("macro_demand_shock", "magnitude"),
    "GlobalGrowth":             ("macro_demand_shock", "magnitude"),
    "DemandDestruction":        ("macro_demand_shock", "magnitude"),

    # ---- Stockpile / inventory interventions ----
    "StrategicStockpile":       ("stockpile_release", "magnitude"),
    "Stockpile":                ("stockpile_release", "magnitude"),
    "InventoryRelease":         ("stockpile_release", "magnitude"),
    "StockpileRelease":         ("stockpile_release", "magnitude"),
    "InventoryBuild":           ("stockpile_release", "magnitude"),
    "StrategicReserve":         ("stockpile_release", "magnitude"),

    # ---- KG entity IDs from build_critical_minerals_kg() ----
    "china_export_controls":    ("export_restriction", "magnitude"),
    "indonesia_nickel_ore_ban": ("export_restriction", "magnitude"),
    "drc_mining_code_reform":   ("capacity_reduction", "magnitude"),
    "global_ev_demand":         ("demand_surge", "magnitude"),
}

# Keyword patterns for fuzzy resolution of LLM-extracted / free-text node names.
# Each tuple: (list_of_keywords_any_of_which_triggers, shock_type)
# Checked in order; first match wins. All matches are case-insensitive substring.
_FUZZY_SHOCK_RULES: List[Tuple[List[str], str]] = [
    # Export / trade restrictions (check before generic "policy")
    (["export quota", "export restrict", "export control", "export ban",
      "export licens", "trade restrict", "trade barrier", "trade sanction",
      "tariff", "sanction", "embargo",
      "licens", "access to export", "market access restrict"],  "export_restriction"),
    # Environmental → capacity
    (["environmental compli", "environmental regulat",
      "mining regulat", "compliance standard"],           "capacity_reduction"),
    # Production capacity (check "supply capacity" / "new supply capacity" before generic "supply")
    (["production capacity", "mining capacity", "refin", "processing cap",
      "mine shutdown", "capacity cut", "capacity reduction",
      "supply capacity", "new capacity", "adjustment period",
      "operational cost"],                                "capacity_reduction"),
    # Policy supply quota (catch-all for quota language)
    (["quota", "allocation restrict", "supply restrict",
      "access to export", "market access"],               "policy_shock"),
    # Stockpile / inventory
    (["stockpile", "strategic reserve", "inventory release",
      "inventory build", "buffer stock"],                 "stockpile_release"),
    # Demand surge / specific industries
    (["battery demand", "ev demand", "electric vehicle",
      "steel demand", "automotive demand", "tech demand",
      "industrial demand", "consumption growth"],         "demand_surge"),
    # Macro demand shocks
    (["financial crisis", "recession", "economic contraction",
      "demand destruction", "global growth", "macro shock"],  "macro_demand_shock"),
    # Generic demand
    (["demand"],                                          "demand_surge"),
    # Generic supply / policy (last resort)
    (["supply", "policy", "regulat", "govern"],          "policy_shock"),
]


def _resolve_treatment_node(name: str) -> Optional[str]:
    """
    Resolve a treatment node name to a shock type string.

    First tries exact lookup in _NODE_TO_SHOCK. Then tries case-insensitive
    exact match. Then applies keyword-based fuzzy matching for LLM-extracted
    free-text node names (e.g., "China's export quotas reduced by 30-40%").

    Returns the shock_type string (e.g., "export_restriction"), or None if
    no match is found.
    """
    # 1. Exact match
    if name in _NODE_TO_SHOCK:
        return _NODE_TO_SHOCK[name][0]

    # 2. Case-insensitive exact match
    name_lower = name.lower()
    for key, (shock_type, _) in _NODE_TO_SHOCK.items():
        if key.lower() == name_lower:
            return shock_type

    # 3. Fuzzy keyword matching
    for keywords, shock_type in _FUZZY_SHOCK_RULES:
        if any(kw in name_lower for kw in keywords):
            return shock_type

    return None


class _DeterministicNoiseRNG:
    """
    Numpy RNG adapter that yields predetermined noise values.

    Used in Layer 3 counterfactual prediction: replay the structural model
    with the *same* exogenous noise recovered during abduction, but under
    modified (counterfactual) shock values.

    Compatible with model.step() which calls rng.normal(0.0, 1.0).
    """

    def __init__(self, noise_sequence: List[float]) -> None:
        self._seq = list(noise_sequence)
        self._idx = 0

    def normal(self, loc: float = 0.0, scale: float = 1.0) -> float:
        if self._idx >= len(self._seq):
            return loc  # fallback: no noise beyond sequence
        val = self._seq[self._idx]
        self._idx += 1
        return loc + scale * val


# ============================================================================
# CausalInferenceEngine
# ============================================================================


class CausalInferenceEngine:
    """
    Unified three-layer causal inference engine (Pearl's Ladder of Causation).

    Layer 1 — Association (Seeing):  P(Y|X)
        Statistical relationships from observational data.
        No causal interpretation.  All of ML/regression lives here.

    Layer 2 — Intervention (Doing):  P(Y|do(X))
        Causal effects via do-calculus (graph surgery).
        Requires a causal DAG.  No amount of Layer-1 data suffices.

    Layer 3 — Counterfactual (Imagining):  P(Y_x | X', Y')
        "What would have happened?" via abduction-action-prediction.
        Requires the full SCM with exogenous noise.  No amount of
        Layer-2 interventional data suffices.

    Args:
        dag:  A CausalDAG encoding the qualitative causal structure.
        cfg:  Optional ScenarioConfig for simulation-based methods
              (Layer 2 ``do()`` and all Layer 3 methods).
        seed: Random seed for bootstrapping and simulation.
    """

    def __init__(
        self,
        dag: CausalDAG,
        cfg: Optional[ScenarioConfig] = None,
        seed: int = 42,
    ) -> None:
        self.dag = dag
        self.cfg = cfg
        self._rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------ #
    #  Layer 1 — Association (Seeing):  P(Y|X)                            #
    # ------------------------------------------------------------------ #

    def correlate(
        self,
        data: pd.DataFrame,
        variables: Optional[List[str]] = None,
        method: str = "pearson",
    ) -> CorrelationMatrix:
        """
        Layer 1 — Pairwise correlation matrix with p-values.

        This is purely associational.  A high correlation does NOT imply
        causation (confounding, collider bias, etc. can produce spurious
        associations).

        Args:
            data:      Observational dataset.
            variables: Columns to include (default: all numeric).
            method:    ``"pearson"`` or ``"spearman"``.
        """
        if data.empty:
            raise ValueError("Data is empty")
        if variables is None:
            variables = list(data.select_dtypes(include=[np.number]).columns)
        for v in variables:
            if v not in data.columns:
                raise ValueError(f"Column '{v}' not in data")

        n = len(variables)
        pearson_mat = np.ones((n, n))
        p_mat = np.zeros((n, n))
        spearman_mat = np.ones((n, n)) if method == "spearman" else None
        spearman_p = np.zeros((n, n)) if method == "spearman" else None

        for i in range(n):
            for j in range(i + 1, n):
                xi = data[variables[i]].dropna()
                xj = data[variables[j]].dropna()
                idx = xi.index.intersection(xj.index)
                xi, xj = xi.loc[idx].values, xj.loc[idx].values

                r, p = stats.pearsonr(xi, xj)
                pearson_mat[i, j] = pearson_mat[j, i] = r
                p_mat[i, j] = p_mat[j, i] = p

                if method == "spearman":
                    rs, ps = stats.spearmanr(xi, xj)
                    spearman_mat[i, j] = spearman_mat[j, i] = rs
                    spearman_p[i, j] = spearman_p[j, i] = ps

        return CorrelationMatrix(
            variables=variables,
            pearson=pearson_mat,
            p_values=p_mat,
            spearman=spearman_mat,
            spearman_p_values=spearman_p,
        )

    def conditional_distribution(
        self,
        data: pd.DataFrame,
        outcome: str,
        conditioning: Optional[Dict[str, Any]] = None,
    ) -> ConditionalDistribution:
        """
        Layer 1 — Estimate P(Y|X=x) from data.

        Purely observational: filters data by conditioning values and
        computes summary statistics of the outcome.

        Args:
            data:         Observational dataset.
            outcome:      Column name for outcome variable.
            conditioning: Dict of {column: value} to condition on.
        """
        if outcome not in data.columns:
            raise ValueError(f"Outcome column '{outcome}' not in data")

        df = data.copy()
        cond = conditioning or {}
        for col, val in cond.items():
            if col not in df.columns:
                raise ValueError(f"Conditioning column '{col}' not in data")
            df = df[df[col] == val]

        vals = df[outcome].dropna().values
        if len(vals) == 0:
            raise ValueError("No data after conditioning")

        quantiles = {
            "25%": float(np.percentile(vals, 25)),
            "50%": float(np.percentile(vals, 50)),
            "75%": float(np.percentile(vals, 75)),
        }
        return ConditionalDistribution(
            outcome=outcome,
            conditioning=cond,
            mean=float(np.mean(vals)),
            std=float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
            count=len(vals),
            values=vals,
            quantiles=quantiles,
        )

    def test_independence(
        self,
        data: pd.DataFrame,
        x: str,
        y: str,
        z: Optional[List[str]] = None,
        alpha: float = 0.05,
    ) -> IndependenceTestResult:
        """
        Layer 1 — Test conditional independence X _||_ Y | Z.

        Marginal independence (z=None): Pearson r test.
        Conditional independence: partial correlation (regress out Z).

        This can be compared with the DAG's d-separation predictions
        (``dag.d_separated({x}, {y}, set(z))``).

        Args:
            data:  Observational dataset.
            x, y:  Variable names.
            z:     Conditioning set (empty = marginal test).
            alpha: Significance level.
        """
        for col in [x, y] + (z or []):
            if col not in data.columns:
                raise ValueError(f"Column '{col}' not in data")

        df = data[[x, y] + (z or [])].dropna()
        x_vals = df[x].values
        y_vals = df[y].values

        if z:
            # Partial correlation: regress X and Y on Z, test residuals
            Z_mat = df[z].values
            Z_aug = np.column_stack([np.ones(len(Z_mat)), Z_mat])
            # Residualize X
            beta_x, _, _, _ = np.linalg.lstsq(Z_aug, x_vals, rcond=None)
            resid_x = x_vals - Z_aug @ beta_x
            # Residualize Y
            beta_y, _, _, _ = np.linalg.lstsq(Z_aug, y_vals, rcond=None)
            resid_y = y_vals - Z_aug @ beta_y
            # Test correlation of residuals
            stat, p_val = stats.pearsonr(resid_x, resid_y)
            method = "partial_correlation"
        else:
            stat, p_val = stats.pearsonr(x_vals, y_vals)
            method = "pearson"

        return IndependenceTestResult(
            x=x,
            y=y,
            z=z or [],
            test_statistic=float(stat),
            p_value=float(p_val),
            independent=p_val > alpha,
            method=method,
            alpha=alpha,
        )

    def regression_association(
        self,
        data: pd.DataFrame,
        outcome: str,
        predictors: List[str],
    ) -> AssociationResult:
        """
        Layer 1 — OLS regression E[Y|X] (associational, NOT causal).

        Fits Y = X @ beta + epsilon via least squares.
        The coefficients are associations, not causal effects — omitted
        variable bias, reverse causation, and collider bias can all
        distort the estimates.

        Args:
            data:       Observational dataset.
            outcome:    Column for the dependent variable Y.
            predictors: Columns for the independent variables X.
        """
        for col in [outcome] + predictors:
            if col not in data.columns:
                raise ValueError(f"Column '{col}' not in data")

        df = data[[outcome] + predictors].dropna()
        y = df[outcome].values
        X = df[predictors].values
        # Add intercept
        X_aug = np.column_stack([np.ones(len(X)), X])

        beta, residuals_ss, _, _ = np.linalg.lstsq(X_aug, y, rcond=None)

        y_hat = X_aug @ beta
        resid = y - y_hat
        n, k = X_aug.shape

        # R-squared
        ss_res = float(np.sum(resid**2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        r_sq = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

        # Standard errors: sqrt(diag((X'X)^-1 * sigma^2))
        sigma2 = ss_res / max(n - k, 1)
        try:
            cov = np.linalg.inv(X_aug.T @ X_aug) * sigma2
            se = np.sqrt(np.diag(cov))
        except np.linalg.LinAlgError:
            se = np.full(k, np.nan)

        # Skip intercept (index 0) in output
        coefficients = {predictors[i]: float(beta[i + 1]) for i in range(len(predictors))}
        std_errors = {predictors[i]: float(se[i + 1]) for i in range(len(predictors))}

        return AssociationResult(
            outcome=outcome,
            predictors=predictors,
            coefficients=coefficients,
            std_errors=std_errors,
            r_squared=r_sq,
            residuals=resid,
        )

    # ------------------------------------------------------------------ #
    #  Layer 2 — Intervention (Doing):  P(Y|do(X))                        #
    # ------------------------------------------------------------------ #

    def identify(self, treatment: str, outcome: str) -> IdentificationResult:
        """
        Layer 2 — Check if P(Y|do(X)) is identifiable from observational data.

        Delegates to the DAG's do-calculus (backdoor/frontdoor criteria).
        This tells you WHETHER you can compute the causal effect, not what
        the effect is.  Use ``backdoor_estimate()`` or ``do()`` for that.
        """
        return self.dag.is_identifiable(treatment, outcome)

    def backdoor_estimate(
        self,
        data: pd.DataFrame,
        treatment: str,
        outcome: str,
        n_bootstrap: int = 200,
    ) -> BackdoorEstimate:
        """
        Layer 2 — Estimate P(Y|do(X)) via the backdoor adjustment formula.

        This is the KEY step that crosses from Layer 1 to Layer 2: the
        backdoor formula uses the causal graph to identify which variables
        to adjust for, then computes the causal effect from observational
        data.

        P(Y|do(X)) = Sum_z P(Y|X,Z=z) P(Z=z)

        For continuous X: OLS regression Y ~ X + Z, where Z is the
        backdoor adjustment set.  The coefficient on X is the average
        causal effect (under linearity).

        Args:
            data:         Observational dataset.
            treatment:    Treatment variable name.
            outcome:      Outcome variable name.
            n_bootstrap:  Number of bootstrap resamples for CI.

        Raises:
            ValueError: If effect is not identifiable via backdoor.
        """
        identification = self.dag.is_identifiable(treatment, outcome)
        adj_set = self.dag.find_backdoor_adjustment_set(treatment, outcome)

        if adj_set is None:
            raise ValueError(
                f"P({outcome}|do({treatment})) is not identifiable via "
                f"backdoor adjustment.  The causal effect cannot be "
                f"estimated from observational data without additional "
                f"assumptions (e.g., instrumental variables)."
            )

        adj_list = sorted(adj_set)
        all_cols = [treatment, outcome] + adj_list
        for col in all_cols:
            if col not in data.columns:
                raise ValueError(f"Column '{col}' not in data")

        df = data[all_cols].dropna()

        def _estimate_ate(sample: pd.DataFrame) -> float:
            """OLS: Y ~ X + Z; return coefficient on X."""
            y = sample[outcome].values
            predictors = [treatment] + adj_list
            X = sample[predictors].values
            X_aug = np.column_stack([np.ones(len(X)), X])
            beta, _, _, _ = np.linalg.lstsq(X_aug, y, rcond=None)
            return float(beta[1])  # coefficient on treatment

        ate = _estimate_ate(df)

        # Bootstrap for standard error and CI
        boot_ates = []
        for _ in range(n_bootstrap):
            idx = self._rng.choice(len(df), size=len(df), replace=True)
            boot_sample = df.iloc[idx]
            boot_ates.append(_estimate_ate(boot_sample))

        boot_ates = np.array(boot_ates)
        ate_se = float(np.std(boot_ates, ddof=1))
        ci_lo = float(np.percentile(boot_ates, 2.5))
        ci_hi = float(np.percentile(boot_ates, 97.5))

        return BackdoorEstimate(
            treatment=treatment,
            outcome=outcome,
            adjustment_set=adj_set,
            ate=ate,
            ate_std_error=ate_se,
            ate_ci=(ci_lo, ci_hi),
            identification=identification,
        )

    def do(
        self,
        treatment_var: str,
        treatment_value: float,
        outcome_vars: Optional[List[str]] = None,
    ) -> SimulationDoResult:
        """
        Layer 2 — Estimate P(Y|do(X=x)) by running the structural model.

        This is "graph surgery" made concrete: the simulation engine
        implements the structural equations, and we intervene by injecting
        a shock that forces the treatment variable to the desired value.

        Requires ``self.cfg`` (a ScenarioConfig) to be set.

        Args:
            treatment_var:   DAG node name to intervene on.
            treatment_value: Value to force (e.g., 0.4 for 40% restriction).
            outcome_vars:    Outcome columns to report (default: P, shortage, D).
        """
        if self.cfg is None:
            raise ValueError(
                "ScenarioConfig required for simulation-based do(). "
                "Pass cfg= to the CausalInferenceEngine constructor."
            )

        if outcome_vars is None:
            outcome_vars = ["P", "shortage", "D", "Q_eff", "cover"]

        # Baseline run (no additional shock)
        df_base, metrics_base = run_scenario(self.cfg)

        # Build intervention config
        cfg_do = self.cfg.model_copy(deep=True)

        shock_type = _resolve_treatment_node(treatment_var)
        if shock_type is not None:
            new_shock = ShockConfig(
                type=shock_type,
                start_year=cfg_do.time.start_year,
                end_year=cfg_do.time.end_year,
                magnitude=treatment_value,
            )
            cfg_do.shocks = list(cfg_do.shocks) + [new_shock]
        else:
            raise ValueError(
                f"No shock mapping for DAG node '{treatment_var}'. "
                f"Supported canonical nodes: {sorted(_NODE_TO_SHOCK.keys())}. "
                f"Or use a descriptive name that contains keywords like "
                f"'export restriction', 'capacity', 'demand', 'stockpile', 'quota'."
            )

        df_do, metrics_do = run_scenario(cfg_do)

        # Compute effect on each outcome variable
        effect = {}
        for var in outcome_vars:
            if var in df_base.columns and var in df_do.columns:
                effect[var] = float(df_do[var].mean() - df_base[var].mean())

        return SimulationDoResult(
            treatment_var=treatment_var,
            treatment_value=treatment_value,
            baseline_trajectory=df_base,
            intervention_trajectory=df_do,
            effect_on_outcome=effect,
            metrics_baseline=metrics_base,
            metrics_intervention=metrics_do,
        )

    def average_treatment_effect(
        self,
        data: pd.DataFrame,
        treatment: str,
        outcome: str,
        method: str = "backdoor",
    ) -> BackdoorEstimate:
        """
        Layer 2 — Convenience method for average treatment effect.

        Args:
            method: ``"backdoor"`` for backdoor adjustment from data.
        """
        if method == "backdoor":
            return self.backdoor_estimate(data, treatment, outcome)
        raise ValueError(f"Unknown method '{method}'. Use 'backdoor'.")

    # ------------------------------------------------------------------ #
    #  Layer 3 — Counterfactual (Imagining):  P(Y_x | X', Y')             #
    # ------------------------------------------------------------------ #

    def abduct(
        self,
        observed_data: pd.DataFrame,
        cfg: Optional[ScenarioConfig] = None,
    ) -> AbductionResult:
        """
        Layer 3, Step 1 — Abduction: infer exogenous noise U from evidence.

        Given an observed trajectory (from a scenario run), recover the
        exogenous noise epsilon_t that rationalizes the observed prices
        under the structural equations in model.py.

        From the price equation:
            log(P_{t+1}) = log(P_t)
                + dt * alpha_P * (tight - lambda * (cover - cover*))
                + sigma_P * sqrt(dt) * epsilon_t

        Solving for epsilon_t:
            epsilon_t = [log(P_{t+1}) - log(P_t)
                - dt * alpha_P * (tight - lambda * (cover - cover*))]
                / (sigma_P * sqrt(dt))

        If sigma_P == 0 (deterministic model), epsilon_t = 0 and any
        gap between model and data is a structural residual.

        Args:
            observed_data: DataFrame from run_scenario() with columns
                           year, P, tight, cover.
            cfg:           ScenarioConfig (defaults to self.cfg).
        """
        cfg = cfg or self.cfg
        if cfg is None:
            raise ValueError("ScenarioConfig required for abduction.")

        p = cfg.parameters
        dt = cfg.time.dt
        eps = p.eps
        sigma = p.sigma_P

        years = observed_data["year"].tolist()
        prices = observed_data["P"].values
        tights = observed_data["tight"].values
        covers = observed_data["cover"].values

        inferred_noise: Dict[int, float] = {}
        squared_errors = []

        for i in range(len(years) - 1):
            yr = years[i]
            log_p = np.log(max(prices[i], eps))
            log_p_next_obs = np.log(max(prices[i + 1], eps))

            # Deterministic drift from the structural equation
            drift = dt * p.alpha_P * (
                tights[i] - p.lambda_cover * (covers[i] - p.cover_star)
            )
            log_p_next_pred = log_p + drift

            if sigma > 0:
                noise_t = (log_p_next_obs - log_p_next_pred) / (
                    sigma * np.sqrt(dt)
                )
            else:
                noise_t = 0.0

            inferred_noise[yr] = float(noise_t)
            squared_errors.append((log_p_next_obs - log_p_next_pred) ** 2)

        # Last year: no next price to invert, set noise to 0
        if years:
            inferred_noise[years[-1]] = 0.0

        rmse = float(np.sqrt(np.mean(squared_errors))) if squared_errors else 0.0

        return AbductionResult(
            years=years,
            inferred_noise=inferred_noise,
            fit_error=rmse,
        )

    def counterfactual(
        self,
        observed_data: pd.DataFrame,
        do_overrides: Dict[int, Dict[str, float]],
        cfg: Optional[ScenarioConfig] = None,
    ) -> CounterfactualResult:
        """
        Layer 3 — Full counterfactual via Pearl's 3-step algorithm.

        Step 1 (Abduction):  Infer exogenous noise U from the observed
            trajectory — "what randomness was realized in the actual world?"

        Step 2 (Action):  Modify the structural equations by applying
            do-interventions (shock overrides) for specified years.

        Step 3 (Prediction):  Propagate through the modified SCM with
            the SAME noise U — "in a world where X had been different
            but all other randomness was the same, what would Y have been?"

        This is fundamentally different from simply re-running the
        simulation with different shocks (which uses fresh randomness).
        The key insight is that counterfactuals hold exogenous noise
        fixed, which preserves the "individual" identity of the run.

        Args:
            observed_data:  DataFrame from run_scenario().
            do_overrides:   {year: {shock_field: value}} specifying the
                            counterfactual intervention.  E.g.,
                            {2010: {"policy_supply_mult": 1.0}} means
                            "what if there had been no policy shock in 2010?"
            cfg:            ScenarioConfig (defaults to self.cfg).
        """
        cfg = cfg or self.cfg
        if cfg is None:
            raise ValueError("ScenarioConfig required for counterfactual.")

        # ------ Step 1: Abduction ------
        abduction = self.abduct(observed_data, cfg)

        # ------ Step 2 & 3: Action + Prediction ------
        # Build noise sequence for deterministic replay
        noise_seq = [abduction.inferred_noise.get(yr, 0.0) for yr in cfg.years]
        det_rng = _DeterministicNoiseRNG(noise_seq)

        # Initialize state
        s = State(
            year=cfg.time.start_year,
            t_index=0,
            K=cfg.baseline.K0,
            I=cfg.baseline.I0,
            P=cfg.baseline.P0,
        )

        cf_rows = []
        for year in cfg.years:
            # Get factual shock
            factual_shock = shocks_for_year(cfg.shocks, year)

            # Apply counterfactual overrides (Action step)
            overrides = do_overrides.get(year)
            if overrides:
                kwargs = dataclasses.asdict(factual_shock)
                for key, value in overrides.items():
                    if key in kwargs:
                        kwargs[key] = value
                cf_shock = ShockSignals(**kwargs)
            else:
                cf_shock = factual_shock

            # Predict next state with same noise (Prediction step)
            s_next, res = step(cfg, s, cf_shock, det_rng)

            cf_rows.append({
                "year": year,
                "K": s.K,
                "I": s.I,
                "P": s.P,
                "Q": res.Q,
                "Q_eff": res.Q_eff,
                "D": res.D,
                "shortage": res.shortage,
                "tight": res.tight,
                "cover": res.cover,
            })
            s = s_next

        df_cf = pd.DataFrame(cf_rows)

        # Compute effect (delta)
        outcome_cols = ["P", "K", "I", "Q", "Q_eff", "D", "shortage", "tight", "cover"]
        factual_cols = [c for c in outcome_cols if c in observed_data.columns]
        df_effect = pd.DataFrame({"year": cfg.years})
        for col in factual_cols:
            if col in df_cf.columns:
                f_vals = observed_data[col].values[: len(cfg.years)]
                cf_vals = df_cf[col].values[: len(cfg.years)]
                df_effect[col] = cf_vals - f_vals

        summary = {}
        for col in factual_cols:
            if col in df_effect.columns:
                summary[f"mean_delta_{col}"] = float(df_effect[col].mean())
                summary[f"max_abs_delta_{col}"] = float(df_effect[col].abs().max())

        return CounterfactualResult(
            factual_trajectory=observed_data,
            counterfactual_trajectory=df_cf,
            effect=df_effect,
            abduction=abduction,
            do_overrides=do_overrides,
            summary=summary,
        )

    def counterfactual_trajectory(
        self,
        state_0: State,
        years: List[int],
        do_overrides_by_year: Dict[int, Dict[str, float]],
    ) -> CounterfactualTrajectoryResult:
        """
        Layer 3 — Multi-step counterfactual trajectory from a given state.

        Runs both the factual and counterfactual trajectories from the
        same initial state and computes per-year deltas.

        Args:
            state_0:              Initial state.
            years:                Years to simulate.
            do_overrides_by_year: {year: {shock_field: value}} overrides.
        """
        if self.cfg is None:
            raise ValueError("ScenarioConfig required for counterfactual trajectory.")

        cfg = self.cfg

        # Use the SAME seed for both trajectories so exogenous noise is
        # identical — this is what makes it a true counterfactual (holding
        # the "individual" identity fixed) rather than just two simulations.
        seed = int(self._rng.integers(0, 2**31))
        rng_f = np.random.default_rng(seed)
        rng_cf = np.random.default_rng(seed)

        # Factual trajectory
        f_states: List[State] = [state_0]
        f_results: List[StepResult] = []
        s = state_0
        for year in years:
            shock = shocks_for_year(cfg.shocks, year)
            s_next, res = step(cfg, s, shock, rng_f)
            f_states.append(s_next)
            f_results.append(res)
            s = s_next

        # Counterfactual trajectory (same initial state, same noise sequence)
        cf_states: List[State] = [state_0]
        cf_results: List[StepResult] = []
        s = state_0
        for year in years:
            factual_shock = shocks_for_year(cfg.shocks, year)
            overrides = do_overrides_by_year.get(year)
            if overrides:
                kwargs = dataclasses.asdict(factual_shock)
                for key, value in overrides.items():
                    if key in kwargs:
                        kwargs[key] = value
                shock = ShockSignals(**kwargs)
            else:
                shock = factual_shock
            s_next, res = step(cfg, s, shock, rng_cf)
            cf_states.append(s_next)
            cf_results.append(res)
            s = s_next

        # Compute deltas
        deltas: Dict[str, List[float]] = {
            "P": [], "K": [], "I": [],
            "shortage": [], "Q_eff": [], "D": [],
        }
        for i in range(len(years)):
            deltas["P"].append(cf_states[i + 1].P - f_states[i + 1].P)
            deltas["K"].append(cf_states[i + 1].K - f_states[i + 1].K)
            deltas["I"].append(cf_states[i + 1].I - f_states[i + 1].I)
            deltas["shortage"].append(cf_results[i].shortage - f_results[i].shortage)
            deltas["Q_eff"].append(cf_results[i].Q_eff - f_results[i].Q_eff)
            deltas["D"].append(cf_results[i].D - f_results[i].D)

        return CounterfactualTrajectoryResult(
            factual_states=f_states,
            factual_results=f_results,
            counterfactual_states=cf_states,
            counterfactual_results=cf_results,
            years=years,
            deltas=deltas,
        )

    def counterfactual_contrast(
        self,
        cfg_factual: ScenarioConfig,
        cfg_counterfactual: ScenarioConfig,
    ) -> CounterfactualTrajectoryResult:
        """
        Layer 3 — Compare two complete scenario runs (factual vs counterfactual).

        A simpler form of counterfactual reasoning that doesn't require
        the full abduction step — useful for "what if we had used policy A
        instead of policy B?" comparisons.

        Args:
            cfg_factual:        The scenario that actually happened.
            cfg_counterfactual: The alternative scenario.
        """
        df_f, _ = run_scenario(cfg_factual)
        df_cf, _ = run_scenario(cfg_counterfactual)

        # Build State/StepResult lists from DataFrames
        f_states = [
            State(year=int(r["year"]), t_index=i, K=r["K"], I=r["I"], P=r["P"])
            for i, r in df_f.iterrows()
        ]
        f_results = [
            StepResult(
                Q=r["Q"], Q_eff=r["Q_eff"],
                Q_sub=r.get("Q_sub", 0.0), Q_fringe=r.get("Q_fringe", 0.0),
                Q_total=r.get("Q_total", r["Q"]),
                D=r["D"], shortage=r["shortage"], tight=r["tight"], cover=r["cover"],
            )
            for _, r in df_f.iterrows()
        ]
        cf_states = [
            State(year=int(r["year"]), t_index=i, K=r["K"], I=r["I"], P=r["P"])
            for i, r in df_cf.iterrows()
        ]
        cf_results = [
            StepResult(
                Q=r["Q"], Q_eff=r["Q_eff"],
                Q_sub=r.get("Q_sub", 0.0), Q_fringe=r.get("Q_fringe", 0.0),
                Q_total=r.get("Q_total", r["Q"]),
                D=r["D"], shortage=r["shortage"], tight=r["tight"], cover=r["cover"],
            )
            for _, r in df_cf.iterrows()
        ]

        n = min(len(f_states), len(cf_states))
        years = [f_states[i].year for i in range(n)]

        deltas: Dict[str, List[float]] = {
            "P": [], "K": [], "I": [],
            "shortage": [], "Q_eff": [], "D": [],
        }
        for i in range(n):
            deltas["P"].append(cf_states[i].P - f_states[i].P)
            deltas["K"].append(cf_states[i].K - f_states[i].K)
            deltas["I"].append(cf_states[i].I - f_states[i].I)
            deltas["shortage"].append(cf_results[i].shortage - f_results[i].shortage)
            deltas["Q_eff"].append(cf_results[i].Q_eff - f_results[i].Q_eff)
            deltas["D"].append(cf_results[i].D - f_results[i].D)

        return CounterfactualTrajectoryResult(
            factual_states=f_states,
            factual_results=f_results,
            counterfactual_states=cf_states,
            counterfactual_results=cf_results,
            years=years,
            deltas=deltas,
        )
