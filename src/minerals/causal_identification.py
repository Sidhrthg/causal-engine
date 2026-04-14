"""
Causal parameter identification: synthetic control, IV, RD, and DiD.

Implements identification of treatment effects as specified in causal_inference.py:
- tau_K  : Synthetic Control (Abadie et al. 2010)
- eta_D  : Instrumental Variables / 2SLS (supply shocks as instruments)
- alpha_P: Regression Discontinuity (local linear at policy threshold)
- policy_shock: Difference-in-Differences (parallel trends, panel TWFE)

Reference:
- Pearl, J. (2009). Causality: Models, Reasoning, and Inference
- Abadie et al. (2010). Synthetic Control Methods for Comparative Case Studies
- Angrist & Pischke (2009). Mostly Harmless Econometrics
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class TreatmentEffect:
    """Results from causal identification."""
    treatment_effect: pd.Series  # Actual - Counterfactual
    counterfactual: pd.Series
    actual: pd.Series
    weights: Dict[str, float]
    pre_treatment_rmse: float
    post_treatment_years: List[int]


class SyntheticControl:
    """
    Synthetic control method for causal inference.
    
    Estimates treatment effects by constructing a synthetic control
    from weighted combination of untreated units.
    """
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
    
    def estimate_treatment_effect(
        self,
        data: pd.DataFrame,
        treated_unit: str,
        control_units: List[str],
        treatment_time: int,
        outcome_var: str,
        unit_col: str = "country",
        time_col: str = "year"
    ) -> TreatmentEffect:
        """
        Estimate treatment effect using synthetic control.
        
        Args:
            data: Panel data with units and time
            treated_unit: Name of treated unit (e.g., "USA")
            control_units: List of control unit names
            treatment_time: Year treatment started
            outcome_var: Variable to analyze (e.g., "trade_value_usd")
            unit_col: Column name for unit identifier
            time_col: Column name for time identifier
            
        Returns:
            TreatmentEffect with counterfactual and treatment effect
        """
        
        # Split data into pre and post treatment
        pre_data = data[data[time_col] < treatment_time].copy()
        post_data = data[data[time_col] >= treatment_time].copy()
        
        # Get treated unit outcomes
        treated_pre = pre_data[pre_data[unit_col] == treated_unit]
        treated_post = post_data[post_data[unit_col] == treated_unit]
        
        if len(treated_pre) == 0:
            raise ValueError(f"No pre-treatment data for {treated_unit}")
        
        # Get control units outcomes
        controls_pre = pre_data[pre_data[unit_col].isin(control_units)]
        controls_post = post_data[post_data[unit_col].isin(control_units)]
        
        # Optimize weights to match pre-treatment
        weights = self._optimize_weights(
            treated_outcome=treated_pre[outcome_var].values,
            control_outcomes=self._pivot_controls(controls_pre, unit_col, time_col, outcome_var),
            control_units=control_units
        )
        
        # Construct synthetic control for post-treatment
        synthetic_post = self._construct_synthetic(
            controls_post, weights, unit_col, time_col, outcome_var
        )
        
        # Calculate treatment effect
        actual = treated_post.set_index(time_col)[outcome_var]
        counterfactual = synthetic_post.set_index(time_col)[outcome_var]
        treatment_effect = actual - counterfactual
        
        # Calculate pre-treatment fit quality
        synthetic_pre = self._construct_synthetic(
            controls_pre, weights, unit_col, time_col, outcome_var
        )
        pre_rmse = np.sqrt(np.mean(
            (treated_pre[outcome_var].values - synthetic_pre[outcome_var].values) ** 2
        ))
        
        if self.verbose:
            print(f"Pre-treatment RMSE: {pre_rmse:.4f}")
            print(f"Weights: {weights}")
            print(f"Average treatment effect: {treatment_effect.mean():.4f}")
        
        return TreatmentEffect(
            treatment_effect=treatment_effect,
            counterfactual=counterfactual,
            actual=actual,
            weights=weights,
            pre_treatment_rmse=pre_rmse,
            post_treatment_years=post_data[time_col].unique().tolist()
        )
    
    def _pivot_controls(
        self, 
        controls_data: pd.DataFrame,
        unit_col: str,
        time_col: str,
        outcome_var: str
    ) -> np.ndarray:
        """Pivot control units into matrix: time x units."""
        pivoted = controls_data.pivot(
            index=time_col,
            columns=unit_col,
            values=outcome_var
        )
        return pivoted.values
    
    def _optimize_weights(
        self,
        treated_outcome: np.ndarray,
        control_outcomes: np.ndarray,
        control_units: List[str]
    ) -> Dict[str, float]:
        """
        Find optimal weights to minimize pre-treatment fit.
        
        Solves: min ||Y_treated - W * Y_controls||^2
        subject to: W >= 0, sum(W) = 1
        """
        n_controls = len(control_units)
        
        def objective(w):
            synthetic = control_outcomes @ w
            return np.sum((treated_outcome - synthetic) ** 2)
        
        # Constraints: weights sum to 1, weights >= 0
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        ]
        bounds = [(0, 1) for _ in range(n_controls)]
        
        # Initial guess: equal weights
        w0 = np.ones(n_controls) / n_controls
        
        result = minimize(
            objective,
            w0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        if not result.success:
            print(f"Warning: Optimization did not converge: {result.message}")
        
        return {unit: weight for unit, weight in zip(control_units, result.x)}
    
    def _construct_synthetic(
        self,
        controls_data: pd.DataFrame,
        weights: Dict[str, float],
        unit_col: str,
        time_col: str,
        outcome_var: str
    ) -> pd.DataFrame:
        """Construct synthetic control using weights."""
        
        synthetic = []
        for time in controls_data[time_col].unique():
            time_data = controls_data[controls_data[time_col] == time]
            
            synthetic_value = sum(
                time_data[time_data[unit_col] == unit][outcome_var].iloc[0] * weight
                for unit, weight in weights.items()
                if unit in time_data[unit_col].values
            )
            
            synthetic.append({
                time_col: time,
                outcome_var: synthetic_value
            })
        
        return pd.DataFrame(synthetic)
    
    def placebo_test(
        self,
        data: pd.DataFrame,
        treated_unit: str,
        control_units: List[str],
        treatment_time: int,
        outcome_var: str,
        n_placebos: int = 10,
        **kwargs
    ) -> Dict:
        """
        Run placebo tests by applying method to control units.
        
        If treatment effect is real, should be larger than placebo effects.
        """
        
        # Get actual treatment effect
        actual_effect = self.estimate_treatment_effect(
            data, treated_unit, control_units, treatment_time, outcome_var, **kwargs
        )
        
        # Run placebo tests on control units
        placebo_effects = []
        for placebo_unit in control_units[:n_placebos]:
            try:
                placebo_controls = [u for u in control_units if u != placebo_unit]
                placebo_result = self.estimate_treatment_effect(
                    data, placebo_unit, placebo_controls, 
                    treatment_time, outcome_var, **kwargs
                )
                placebo_effects.append(placebo_result.treatment_effect.mean())
            except Exception as e:
                if self.verbose:
                    print(f"Placebo {placebo_unit} failed: {e}")
                continue
        
        # Compare actual to distribution of placebos
        actual_mean = actual_effect.treatment_effect.mean()
        p_value = np.mean([abs(p) >= abs(actual_mean) for p in placebo_effects])
        
        return {
            'actual_effect': actual_mean,
            'placebo_effects': placebo_effects,
            'p_value': p_value,
            'significant': p_value < 0.05
        }


def example_usage():
    """Example of how to use SyntheticControl."""
    
    # Create synthetic data
    np.random.seed(42)
    years = list(range(2000, 2015))
    countries = ["USA", "EU", "Japan", "India", "Brazil"]
    
    data = []
    for country in countries:
        for year in years:
            # Baseline trend
            value = 100 + year * 2 + np.random.normal(0, 5)
            
            # Treatment effect for USA after 2010
            if country == "USA" and year >= 2010:
                value -= 30  # 30% drop
            
            data.append({
                'country': country,
                'year': year,
                'trade_value': value
            })
    
    df = pd.DataFrame(data)
    
    # Run synthetic control
    sc = SyntheticControl(verbose=True)
    result = sc.estimate_treatment_effect(
        data=df,
        treated_unit="USA",
        control_units=["EU", "Japan", "India", "Brazil"],
        treatment_time=2010,
        outcome_var="trade_value"
    )
    
    print("\n=== Results ===")
    print(f"Treatment Effect (post-2010):\n{result.treatment_effect}")
    print(f"\nWeights: {result.weights}")
    print(f"Pre-treatment RMSE: {result.pre_treatment_rmse:.2f}")
    
    # Placebo test
    placebo = sc.placebo_test(
        data=df,
        treated_unit="USA",
        control_units=["EU", "Japan", "India", "Brazil"],
        treatment_time=2010,
        outcome_var="trade_value",
        n_placebos=3
    )
    print(f"\nPlacebo test p-value: {placebo['p_value']:.3f}")


if __name__ == "__main__":
    example_usage()


# ---------------------------------------------------------------------------
# eta_D: Instrumental Variables (2SLS)
# Estimand: ∂log(Demand)/∂log(Price), instrumented by exogenous supply shocks
# ---------------------------------------------------------------------------

@dataclass
class IVResult:
    """Results from two-stage least squares estimation."""
    estimate: float               # 2SLS coefficient on endogenous treatment
    se: float                     # Asymptotic standard error
    first_stage_f_stat: float     # First-stage F-statistic (weak instrument test)
    weak_instrument: bool         # True if F < 10 (Stock-Yogo threshold)
    n_obs: int
    confidence_interval: Tuple[float, float]  # 95% CI


class InstrumentalVariable:
    """
    Two-stage least squares (2SLS) estimator for demand elasticity (eta_D).

    Identification: supply shocks (e.g. mine closures, export quotas) shift
    Price but affect Demand only through Price — satisfying the exclusion
    restriction. This breaks the Price ↔ Demand simultaneity and identifies
    ∂Demand/∂Price causally.

    Stage 1: Price ~ Instrument + Controls   (relevance check via F-stat)
    Stage 2: Demand ~ Price_hat + Controls   (structural equation)
    """

    def __init__(self, verbose: bool = False):
        self.verbose = verbose

    def estimate(
        self,
        data: pd.DataFrame,
        outcome_var: str,
        treatment_var: str,
        instrument_var: str,
        controls: Optional[List[str]] = None,
    ) -> IVResult:
        """
        Estimate via 2SLS.

        Args:
            data: Panel or cross-sectional DataFrame.
            outcome_var: Endogenous outcome (e.g. "demand").
            treatment_var: Endogenous treatment (e.g. "price").
            instrument_var: Exogenous instrument (e.g. "supply_shock").
            controls: Exogenous covariates included in both stages.

        Returns:
            IVResult with estimate, SE, first-stage diagnostics.
        """
        controls = controls or []
        n = len(data)

        # --- Stage 1: T = a + b*Z + c*X ---
        z_cols = [instrument_var] + controls
        X1 = np.column_stack([np.ones(n)] + [data[c].values for c in z_cols])
        T = data[treatment_var].values.astype(float)
        beta1, _, _, _ = np.linalg.lstsq(X1, T, rcond=None)
        T_hat = X1 @ beta1
        resid1 = T - T_hat

        # Partial F-statistic for the excluded instrument (Stock-Yogo weak instrument test).
        # Compare full model (instrument + controls) vs restricted model (controls only).
        # This correctly isolates the instrument's predictive power regardless of controls.
        ss_resid1 = np.sum(resid1 ** 2)
        k1 = X1.shape[1]
        if controls:
            X1_restr = np.column_stack([np.ones(n)] + [data[c].values for c in controls])
            beta1_restr, _, _, _ = np.linalg.lstsq(X1_restr, T, rcond=None)
            ss_restr = np.sum((T - X1_restr @ beta1_restr) ** 2)
            f_stat = ((ss_restr - ss_resid1) / 1.0) / (ss_resid1 / max(n - k1, 1))
        else:
            ss_total = np.sum((T - T.mean()) ** 2)
            f_stat = ((ss_total - ss_resid1) / 1.0) / (ss_resid1 / max(n - k1, 1))

        # --- Stage 2: Y = a + b*T_hat + c*X ---
        X2 = np.column_stack([np.ones(n), T_hat] + [data[c].values for c in controls])
        Y = data[outcome_var].values.astype(float)
        beta2, _, _, _ = np.linalg.lstsq(X2, Y, rcond=None)
        estimate = float(beta2[1])

        # Asymptotic SE: use structural residuals with original T (not T_hat) to get
        # correct sigma^2. Using T_hat inflates sigma^2 by Var(first-stage noise)*beta^2,
        # causing SE overestimation (~1.26x for typical instruments).
        X2_orig = np.column_stack([np.ones(n), T] + [data[c].values for c in controls])
        resid2 = Y - X2_orig @ beta2
        k2 = X2.shape[1]
        sigma2 = np.sum(resid2 ** 2) / max(n - k2, 1)
        xtx_inv = np.linalg.pinv(X2.T @ X2)
        se = float(np.sqrt(sigma2 * xtx_inv[1, 1]))

        ci = (estimate - 1.96 * se, estimate + 1.96 * se)

        if self.verbose:
            print(f"2SLS estimate: {estimate:.4f} (SE={se:.4f})")
            print(f"First-stage F: {f_stat:.2f} {'⚠️ weak' if f_stat < 10 else '✅'}")

        return IVResult(
            estimate=estimate,
            se=se,
            first_stage_f_stat=float(f_stat),
            weak_instrument=f_stat < 10,
            n_obs=n,
            confidence_interval=ci,
        )


# ---------------------------------------------------------------------------
# alpha_P: Regression Discontinuity (local linear at policy threshold)
# Estimand: ∂Price/∂Shortage at a discrete policy event cutoff
# ---------------------------------------------------------------------------

@dataclass
class RDResult:
    """Results from regression discontinuity estimation."""
    discontinuity: float          # Jump in outcome at threshold (LATE)
    se: float
    bandwidth: float              # Window used around threshold
    n_left: int                   # Observations just left of threshold
    n_right: int                  # Observations just right of threshold
    confidence_interval: Tuple[float, float]


class RegressionDiscontinuity:
    """
    Sharp regression discontinuity for price-adjustment speed (alpha_P).

    Estimates the causal jump in outcome (e.g. Price) at a known policy
    threshold in a running variable (e.g. Shortage index, time).  Within a
    bandwidth around the threshold, units are as-good-as-randomly assigned
    to treatment (above/below), satisfying local randomisation.

    Estimator: local linear regression on each side of the threshold;
    discontinuity = right-limit minus left-limit at the cutoff.
    """

    def __init__(self, verbose: bool = False):
        self.verbose = verbose

    def estimate(
        self,
        data: pd.DataFrame,
        running_var: str,
        outcome_var: str,
        threshold: float,
        bandwidth: Optional[float] = None,
    ) -> RDResult:
        """
        Estimate the discontinuity at threshold.

        Args:
            data: DataFrame with running_var and outcome_var columns.
            running_var: Continuous variable that crosses the threshold
                         (e.g. shortage index, calendar time).
            outcome_var: Outcome of interest (e.g. price, supply).
            threshold: The known cutoff value.
            bandwidth: Half-width of local window (default: 1 SD of running_var).

        Returns:
            RDResult with discontinuity estimate and diagnostics.
        """
        rv = data[running_var].values.astype(float)
        if bandwidth is None:
            bandwidth = float(np.std(rv))

        mask = np.abs(rv - threshold) <= bandwidth
        local = data[mask].copy()
        left = local[local[running_var] < threshold]
        right = local[local[running_var] >= threshold]

        if len(left) < 2 or len(right) < 2:
            raise ValueError(
                f"Too few observations within bandwidth {bandwidth:.3f}: "
                f"left={len(left)}, right={len(right)}. Increase bandwidth."
            )

        # Fit interaction model on local window for pooled SE:
        # Y = a + b*D + c*(X-c0) + d*D*(X-c0)   where b = discontinuity
        rv_c = (local[running_var].values - threshold).astype(float)
        D = (rv_c >= 0).astype(float)
        Y = local[outcome_var].values.astype(float)
        X_mat = np.column_stack([np.ones(len(Y)), D, rv_c, D * rv_c])
        beta, _, _, _ = np.linalg.lstsq(X_mat, Y, rcond=None)
        disc = float(beta[1])

        resid = Y - X_mat @ beta
        n, k = len(Y), 4
        sigma2 = np.sum(resid ** 2) / max(n - k, 1)
        xtx_inv = np.linalg.pinv(X_mat.T @ X_mat)
        se = float(np.sqrt(sigma2 * xtx_inv[1, 1]))

        ci = (disc - 1.96 * se, disc + 1.96 * se)

        if self.verbose:
            print(f"RD discontinuity: {disc:.4f} (SE={se:.4f})")
            print(f"Bandwidth: {bandwidth:.3f}, n_left={len(left)}, n_right={len(right)}")

        return RDResult(
            discontinuity=disc,
            se=se,
            bandwidth=bandwidth,
            n_left=len(left),
            n_right=len(right),
            confidence_interval=ci,
        )


# ---------------------------------------------------------------------------
# policy_shock_magnitude: Difference-in-Differences (TWFE)
# Estimand: ATT of export quota on supply / trade value, panel countries
# ---------------------------------------------------------------------------

@dataclass
class DIDResult:
    """Results from difference-in-differences estimation."""
    att: float                    # Average treatment effect on the treated
    se: float
    pre_trend_pvalue: float       # p-value for parallel pre-trends test (want > 0.05)
    n_treated: int                # Number of treated units
    n_control: int
    post_treatment_years: List[int]
    confidence_interval: Tuple[float, float]


class DifferenceInDifferences:
    """
    Difference-in-differences (2x2 DiD / panel TWFE) for policy_shock_magnitude.

    Estimates ATT = (Y_treated_post - Y_treated_pre) - (Y_control_post - Y_control_pre).
    Identification relies on the parallel trends assumption: absent the policy,
    treated and control units would have followed the same time trend.

    Pre-trend test: applies the DiD estimator to a placebo split within the
    pre-treatment period. If the coefficient is near zero (p > 0.05) trends
    are parallel.
    """

    def __init__(self, verbose: bool = False):
        self.verbose = verbose

    def estimate(
        self,
        data: pd.DataFrame,
        outcome_var: str,
        treatment_col: str,
        time_col: str,
        treatment_time: int,
        unit_col: Optional[str] = None,
    ) -> DIDResult:
        """
        Estimate ATT via OLS with interaction term (equivalent to 2x2 DiD).

        Args:
            data: Panel DataFrame.
            outcome_var: Outcome variable (e.g. "trade_value_usd").
            treatment_col: Binary column: 1 = treated unit, 0 = control.
            time_col: Time period column (int year).
            treatment_time: First period of treatment.
            unit_col: Unit identifier column (used to count distinct units).

        Returns:
            DIDResult with ATT, SE, parallel-trends p-value, and unit counts.
        """
        D = data[treatment_col].values.astype(float)
        Post = (data[time_col] >= treatment_time).astype(float).values
        Y = data[outcome_var].values.astype(float)

        # OLS: Y = a + b*D + c*Post + d*(D*Post)  → d = ATT
        X = np.column_stack([np.ones(len(Y)), D, Post, D * Post])
        beta, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)
        att = float(beta[3])

        resid = Y - X @ beta
        n, k = len(Y), 4
        sigma2 = np.sum(resid ** 2) / max(n - k, 1)
        xtx_inv = np.linalg.pinv(X.T @ X)
        se = float(np.sqrt(sigma2 * xtx_inv[3, 3]))

        # --- Parallel pre-trends test ---
        pre_trend_pvalue = self._pre_trend_test(data, outcome_var, treatment_col, time_col, treatment_time)

        ci = (att - 1.96 * se, att + 1.96 * se)

        if unit_col and unit_col in data.columns:
            n_treated = int(data[data[treatment_col] == 1][unit_col].nunique())
            n_control = int(data[data[treatment_col] == 0][unit_col].nunique())
        else:
            n_treated = int((data[treatment_col] == 1).sum())
            n_control = int((data[treatment_col] == 0).sum())

        post_years = sorted(data[data[time_col] >= treatment_time][time_col].unique().tolist())

        if self.verbose:
            print(f"DiD ATT: {att:.4f} (SE={se:.4f})")
            print(f"Pre-trend p-value: {pre_trend_pvalue:.3f} {'✅ parallel' if pre_trend_pvalue > 0.05 else '⚠️ non-parallel'}")

        return DIDResult(
            att=att,
            se=se,
            pre_trend_pvalue=pre_trend_pvalue,
            n_treated=n_treated,
            n_control=n_control,
            post_treatment_years=post_years,
            confidence_interval=ci,
        )

    def _pre_trend_test(
        self,
        data: pd.DataFrame,
        outcome_var: str,
        treatment_col: str,
        time_col: str,
        treatment_time: int,
    ) -> float:
        """
        Placebo DiD on pre-period only: split pre-period at midpoint and test
        whether the DiD coefficient is zero (no differential pre-trend).
        Returns p-value; want p > 0.05 for parallel trends.
        """
        pre = data[data[time_col] < treatment_time].copy()
        pre_times = sorted(pre[time_col].unique())
        if len(pre_times) < 4:
            return 1.0  # cannot test with fewer than 4 pre-periods

        mid = pre_times[len(pre_times) // 2]
        D_pre = pre[treatment_col].values.astype(float)
        Post_pre = (pre[time_col] >= mid).astype(float).values
        Y_pre = pre[outcome_var].values.astype(float)
        X_pre = np.column_stack([np.ones(len(Y_pre)), D_pre, Post_pre, D_pre * Post_pre])
        beta_pre, _, _, _ = np.linalg.lstsq(X_pre, Y_pre, rcond=None)

        resid_pre = Y_pre - X_pre @ beta_pre
        n_pre = len(Y_pre)
        sigma2_pre = np.sum(resid_pre ** 2) / max(n_pre - 4, 1)
        xtx_inv_pre = np.linalg.pinv(X_pre.T @ X_pre)
        se_pre = float(np.sqrt(sigma2_pre * xtx_inv_pre[3, 3]))

        t_stat = abs(float(beta_pre[3])) / (se_pre + 1e-12)
        return float(2 * stats.t.sf(t_stat, df=max(n_pre - 4, 1)))

    def placebo_test(
        self,
        data: pd.DataFrame,
        outcome_var: str,
        treatment_col: str,
        time_col: str,
        treatment_time: int,
        n_placebos: int = 10,
        unit_col: Optional[str] = None,
    ) -> Dict:
        """
        Placebo policy times: reassign treatment_time to random pre-period years
        and check if DiD ATT is smaller than the actual estimate.
        Returns empirical p-value (fraction of placebos with |ATT| >= actual).
        """
        actual = self.estimate(data, outcome_var, treatment_col, time_col, treatment_time, unit_col)
        pre_times = sorted(data[data[time_col] < treatment_time][time_col].unique())
        if len(pre_times) < 2:
            return {"actual_att": actual.att, "placebo_atts": [], "p_value": 1.0, "significant": False}

        rng = np.random.default_rng(42)
        placebo_atts = []
        for t_placebo in rng.choice(pre_times, size=min(n_placebos, len(pre_times)), replace=False):
            try:
                r = self.estimate(data, outcome_var, treatment_col, time_col, int(t_placebo), unit_col)
                placebo_atts.append(r.att)
            except Exception:
                continue

        p_value = float(np.mean([abs(p) >= abs(actual.att) for p in placebo_atts])) if placebo_atts else 1.0
        return {
            "actual_att": actual.att,
            "placebo_atts": placebo_atts,
            "p_value": p_value,
            "significant": p_value < 0.05,
        }
