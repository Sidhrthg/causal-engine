"""
Causal parameter identification using synthetic control method.

This implements identification of treatment effects as specified in
causal_inference.py, specifically for the tau_K (capacity adjustment time) parameter.

Identification Strategy:
- Estimand: P(Capacity_t | do(PriceShock_{t-k}))
- Method: Synthetic Control (Abadie et al. 2010)
- Assumptions:
  * Parallel trends (control units track treated absent intervention)
  * No spillovers between units
  * SUTVA (Stable Unit Treatment Value Assumption)

See GraphiteSupplyChainDAG.get_parameter_identifications() in causal_inference.py
for the formal identifiability proof.

Reference:
- Pearl, J. (2009). Causality: Models, Reasoning, and Inference
- Abadie et al. (2010). Synthetic Control Methods for Comparative Case Studies
"""

import numpy as np
import pandas as pd
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
