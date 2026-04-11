"""
Pearl's three layers of causal reasoning (Ladder of Causation).

Regular probability only covers Layer 1 (observation/association). Causal reasoning
requires two additional layers: intervention (do) and counterfactual (imagine).

  Layer 1 — Association (Seeing):  P(Y|X)     "What if I see X?"
  Layer 2 — Intervention (Doing):  P(Y|do(X)) "What if I do X?"
  Layer 3 — Counterfactual (Imagine): P(Y_x | X',Y') "What if X had been x, given what I saw?"

Reference: Pearl, J. (2009). Causality: Models, Reasoning, and Inference. Cambridge.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from .causal_inference import CausalDAG, GraphiteSupplyChainDAG, IdentificationResult
from .model import State, StepResult, step
from .schema import ScenarioConfig
from .shocks import ShockSignals, shocks_for_year


# ---------------------------------------------------------------------------
# Layer 1: Association (Seeing) — P(Y|X)
# ---------------------------------------------------------------------------


def observational_conditional(
    data: pd.DataFrame,
    outcome: str,
    conditioning: Optional[Dict[str, Any]] = None,
) -> pd.Series:
    """
    Layer 1 — Association: Compute E[outcome | conditioning] from observational data.

    This is pure probability: no intervention, no causation. "What if I see X?"
    Use this for descriptive statistics and associations only.

    Args:
        data: Observational dataset (e.g. Comtrade, run outputs).
        outcome: Column name for outcome variable.
        conditioning: Optional dict of {column: value} to condition on.

    Returns:
        Series of outcome values (or summary) given conditioning.
    """
    if outcome not in data.columns:
        raise ValueError(f"Outcome column '{outcome}' not in data")
    df = data.copy()
    if conditioning:
        for col, val in conditioning.items():
            if col not in df.columns:
                raise ValueError(f"Conditioning column '{col}' not in data")
            df = df[df[col] == val]
    return df[outcome]


def observational_summary(data: pd.DataFrame, outcome: str, group_by: Optional[str] = None) -> pd.DataFrame:
    """
    Layer 1 — Association: Summary statistics of outcome (mean, std, count).

    Observational only. No do(·).
    """
    if group_by:
        return data.groupby(group_by)[outcome].agg(["mean", "std", "count"]).reset_index()
    return pd.DataFrame({"mean": [data[outcome].mean()], "std": [data[outcome].std()], "count": [len(data)]})


# ---------------------------------------------------------------------------
# Layer 2: Intervention (Doing) — P(Y|do(X))
# ---------------------------------------------------------------------------


def interventional_identifiability(
    treatment: str,
    outcome: str,
    dag: Optional[CausalDAG] = None,
) -> IdentificationResult:
    """
    Layer 2 — Intervention: Is P(outcome | do(treatment)) identifiable from data?

    Uses do-calculus (backdoor/frontdoor). Does not compute the effect;
    use scenario runs (run_scenario with different treatments) to estimate it.
    """
    if dag is None:
        dag = GraphiteSupplyChainDAG()
    return dag.is_identifiable(treatment, outcome)


def mutilated_graph_for_do(dag: CausalDAG, node: str):
    """
    Layer 2 — Intervention: Graph after do(node); removes incoming edges to node.
    """
    return dag.remove_incoming_edges(node)


# ---------------------------------------------------------------------------
# Layer 3: Counterfactual (Imagining) — P(Y_x | X', Y')
# ---------------------------------------------------------------------------


def counterfactual_step(
    state: State,
    cfg: ScenarioConfig,
    year: int,
    do_shock_overrides: Dict[str, float],
    rng: np.random.Generator,
) -> Tuple[State, StepResult]:
    """
    Layer 3 — Counterfactual: One-step outcome if we had applied do(shock overrides).

    "Given we were in state at year, what would have happened if we had set
    shock = counterfactual values?" Uses the structural model (step) with
    the counterfactual shock.

    Args:
        state: Current state (e.g. at year t).
        cfg: Scenario config (used for parameters and shock list).
        year: Year for which to override shocks.
        do_shock_overrides: Keys are ShockSignals field names (e.g. policy_supply_mult),
            values are the counterfactual values (e.g. 1.0 for "no policy shock").
        rng: Random generator (for sigma_P if present).

    Returns:
        (state_next, step_result) under the counterfactual shock.
    """
    factual_shock = shocks_for_year(cfg.shocks, year)
    # Override with do(·) values
    kwargs = dataclasses.asdict(factual_shock)
    for key, value in do_shock_overrides.items():
        if key in kwargs:
            kwargs[key] = value
    counterfactual_shock = ShockSignals(**kwargs)
    return step(cfg, state, counterfactual_shock, rng)


def counterfactual_trajectory(
    state_0: State,
    cfg: ScenarioConfig,
    years: list[int],
    do_shock_overrides_by_year: Dict[int, Dict[str, float]],
    rng: np.random.Generator,
) -> Tuple[list[State], list[StepResult]]:
    """
    Layer 3 — Counterfactual: Multi-step trajectory under do(interventions) by year.

    For each year in years, if the year is in do_shock_overrides_by_year, use that
    counterfactual shock; otherwise use the scenario's factual shock.
    """
    states = [state_0]
    results: list[StepResult] = []
    s = state_0
    for year in years:
        overrides = do_shock_overrides_by_year.get(year)
        if overrides:
            s, res = counterfactual_step(s, cfg, year, overrides, rng)
        else:
            shock = shocks_for_year(cfg.shocks, year)
            s, res = step(cfg, s, shock, rng)
        states.append(s)
        results.append(res)
    return states, results


# ---------------------------------------------------------------------------
# Summary / demo
# ---------------------------------------------------------------------------


def three_layers_summary() -> str:
    """Return a short summary of where each layer is implemented."""
    return """
Pearl's three layers in this codebase:

  Layer 1 — Association (Seeing):  P(Y|X)
    • observational_conditional(), observational_summary() in pearl_layers.py
    • Any analysis on raw data (Comtrade, run CSV) without do(·)
    • Validation metrics that compare observed to predicted (association only)

  Layer 2 — Intervention (Doing):  P(Y|do(X))
    • causal_inference.py: CausalDAG, backdoor criterion, is_identifiable()
    • Scenario runs: run_scenario(cfg) with different shocks = do(policy)
    • interventional_identifiability() in pearl_layers.py

  Layer 3 — Counterfactual (Imagining): P(Y_x | X', Y')
    • counterfactual_step(), counterfactual_trajectory() in pearl_layers.py
    • "What would have happened if we had set X=x?" from a given state
    • Requires structural model (model.step) and state; no observation-only equivalent
"""
