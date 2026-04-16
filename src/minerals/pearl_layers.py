"""
Pearl's three layers of causal reasoning (Ladder of Causation).

Regular probability only covers Layer 1 (observation/association). Causal reasoning
requires two additional layers: intervention (do) and counterfactual (imagine).

Formal definitions
------------------
  Layer 1 — Association (Seeing):
      P(Y | X=x)
      "Among observations where X=x, what is Y?"
      Tool: filter data, compute statistics. No claim of causation.

  Layer 2 — Intervention (Doing):
      P(Y | do(X=x))
      "If I forcibly set X=x — severing whatever normally determines X — what is Y?"
      Tool: graph surgery in the SCM (remove all incoming edges to X, fix X=x, run forward).
      Differs from L1 when confounders exist: P(Y|X=x) ≠ P(Y|do(X=x)).

  Layer 3 — Counterfactual (Imagining):
      P(Y_x | X=x', Y=y')
      "Given this specific individual trajectory where X=x' and Y=y' actually occurred,
       what would Y have been if X had instead been x?"
      Tool: Abduction-Action-Prediction (twin network):
        1. Abduction  — infer latent noise U from observed (X=x', Y=y')
        2. Action     — surgically set X=x in the structural equations (same as L2)
        3. Prediction — propagate with the abduced U through the modified model

Hierarchy: L3 > L2 > L1.  L2 cannot be answered from L1 alone (requires a causal model).
           L3 cannot be answered from L2 alone (requires specific noise abduction).

Structural mechanisms in this codebase
---------------------------------------

  SubstitutionSupply  (Q_sub in model.py)
    Structural equation:
        Q_sub = export_restriction · Q · clamp(0, cap, elasticity · max(0, P/P_ref − 1))
    Exogenous noise:   SubstitutionCapacity (latent market structure)
    Observable parents: ExportPolicy (shock.export_restriction), Price (P)

    L1: observe_substitution_association(data)
        Correlation table: Q_sub ~ export_restriction, price_premium
    L2: do_substitution(cfg, elasticity, cap)
        Graph surgery on the Price→SubstitutionSupply edge weight.
        Severs the confounded path (market_conditions→elasticity→Q_sub).
        Returns P(Q_total | do(substitution_elasticity=e)).
    L3: counterfactual_substitution(cfg, cf_elasticity, cf_cap)
        "Given the factual trajectory where elasticity=0 (no substitution),
         what would prices have been if elasticity had been cf_elasticity?"
        Abduction: fix the price-noise seed (exogenous U = noise draws from factual run).
        Action: set elasticity=cf_elasticity.
        Prediction: replay with same U, modified equation.

  FringeSupply  (Q_fringe in model.py)
    Structural equation:
        Q_fringe = fringe_K · clamp(0, 1, max(0, P/P_ref − entry) / entry)
        where fringe_K = fringe_capacity_share · K0
    Exogenous noise:   FringeCapacity (latent, set by fringe_capacity_share parameter)
    Observable parents: Price (P)

    L1: observe_fringe_association(data)
        Correlation table: Q_fringe ~ price_ratio
    L2: do_fringe_supply(cfg, capacity_share, entry_price)
        Graph surgery on FringeCapacity→FringeSupply edge.
        Returns P(Q_total | do(fringe_capacity_share=f)).
    L3: counterfactual_fringe(cfg, cf_share, cf_entry)
        "Given factual trajectory with no fringe, what would prices have been
         if fringe producers had existed with capacity_share=cf_share?"
        Abduction: fix noise seed. Action: set fringe params. Prediction: replay.

  Noise abduction in this model
    The stochastic driver is Euler-Maruyama price noise:
        logP_{t+1} = logP_t + dt·alpha_P·(tight − lambda_cover·(cover − cover_star))
                   + sigma_P·√dt·ε_t,   ε_t ~ N(0,1)
    U = {ε_0, ε_1, ..., ε_T} is the complete exogenous noise sequence.
    Abduction = fixing the RNG seed to the factual run's seed. Because step() calls
    rng.normal() exactly once per timestep regardless of parameters, replaying with
    the same seed and same number of steps reproduces the identical noise sequence ε_t
    in both factual and counterfactual worlds — the correct twin-network coupling.

Reference: Pearl, J. (2009). Causality: Models, Reasoning, and Inference. Cambridge.
           Pearl, J. & Mackenzie, D. (2018). The Book of Why. Basic Books.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .causal_inference import CausalDAG, GraphiteSupplyChainDAG, IdentificationResult
from .model import State, StepResult, step
from .schema import ScenarioConfig
from .shocks import ShockSignals, shocks_for_year


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _override_params(cfg: ScenarioConfig, **param_overrides) -> ScenarioConfig:
    """
    Graph-surgery helper: return a new ScenarioConfig with structural parameters
    surgically overridden.  This is the computational realisation of Pearl's
    do(·) operator — we cut all incoming edges to the target parameter node and
    pin it to the supplied value.

    Uses Pydantic model_copy so the original cfg is never mutated.
    """
    new_params = cfg.parameters.model_copy(update=param_overrides)
    return cfg.model_copy(update={"parameters": new_params})


def _run_scenario_inner(
    cfg: ScenarioConfig,
    noise_sequence: Optional[List[float]] = None,
) -> Tuple[pd.DataFrame, List[float]]:
    """
    Run a scenario, optionally with a pre-supplied noise sequence (for L3 abduction).

    When noise_sequence is None the RNG is seeded from cfg.seed (normal run).
    When noise_sequence is provided those exact values are used for the price
    noise draws in order, bypassing the RNG.  This is the Prediction step of
    Abduction-Action-Prediction.

    Returns (df, noise_drawn) where noise_drawn is the list of ε_t values
    consumed (useful to capture during the factual run for later replay).
    """
    from .metrics import compute_metrics  # local import to avoid circular deps

    rng = np.random.default_rng(cfg.seed)
    noise_drawn: List[float] = []

    s = State(
        year=cfg.time.start_year,
        t_index=0,
        K=cfg.baseline.K0,
        I=cfg.baseline.I0,
        P=cfg.baseline.P0,
    )

    rows = []
    years = cfg.years

    for idx, year in enumerate(years):
        shock = shocks_for_year(cfg.shocks, year)
        p = cfg.parameters

        # --- Euler-Maruyama noise (one draw per step) ---
        if noise_sequence is not None:
            # Prediction step: use pre-abduced noise; still advance rng to stay in sync
            # if sigma_P > 0 so that any downstream calls remain consistent
            _rng_draw = rng.normal(0.0, 1.0) if p.sigma_P > 0 else 0.0
            z = noise_sequence[idx] if idx < len(noise_sequence) else 0.0
        else:
            z = rng.normal(0.0, 1.0) if p.sigma_P > 0 else 0.0

        noise_drawn.append(z)

        # step with explicit noise (avoids double-draw)
        s_next, res = _step_explicit_noise(cfg, s, shock, z)

        row = {
            "year": year,
            "K": s.K, "I": s.I, "P": s.P,
            "Q": res.Q, "Q_eff": res.Q_eff,
            "Q_sub": res.Q_sub, "Q_fringe": res.Q_fringe, "Q_total": res.Q_total,
            "D": res.D, "shortage": res.shortage, "tight": res.tight, "cover": res.cover,
            "shock_export_restriction": shock.export_restriction,
            "shock_demand_surge": shock.demand_surge,
            "shock_capex_shock": shock.capex_shock,
            "shock_stockpile_release": shock.stockpile_release,
            "shock_policy_supply_mult": shock.policy_supply_mult,
            "shock_capacity_supply_mult": shock.capacity_supply_mult,
            "shock_demand_destruction_mult": shock.demand_destruction_mult,
        }
        rows.append(row)
        s = s_next

    df = pd.DataFrame(rows)
    return df, noise_drawn


def _step_explicit_noise(
    cfg: ScenarioConfig, s: State, shock: ShockSignals, z: float
) -> Tuple[State, StepResult]:
    """
    Identical to model.step() but accepts a pre-drawn noise value z instead of
    sampling from an RNG.  Used by _run_scenario_inner for L3 noise replay.
    """
    from .model import _clip  # reuse clip helper

    p = cfg.parameters
    b = cfg.baseline
    pol = cfg.policy
    dt = cfg.time.dt
    eps = p.eps

    u_val = _clip(p.u0 + p.beta_u * float(np.log(max(s.P, eps) / b.P_ref)), p.u_min, p.u_max)
    Q = min(s.K, s.K * u_val)

    g_t = cfg.parameters.demand_growth.g ** s.t_index

    D = (
        b.D0
        * g_t
        * (max(s.P, eps) / b.P_ref) ** p.eta_D
        * (1.0 - pol.substitution)
        * (1.0 - pol.efficiency)
        * (1.0 + shock.demand_surge)
        * shock.demand_destruction_mult
    )
    D = max(D, eps)

    supply_mult = (1.0 - shock.export_restriction) * shock.policy_supply_mult * shock.capacity_supply_mult
    Q_eff = Q * supply_mult

    if p.substitution_elasticity > 0.0 and shock.export_restriction > 0.0:
        price_premium = max(0.0, s.P / b.P_ref - 1.0)
        sub_rate = min(p.substitution_cap, p.substitution_elasticity * price_premium)
        Q_sub = shock.export_restriction * Q * sub_rate
    else:
        Q_sub = 0.0

    if p.fringe_capacity_share > 0.0:
        fringe_K = p.fringe_capacity_share * b.K0
        price_ratio = s.P / max(b.P_ref, eps)
        fringe_premium = max(0.0, price_ratio - p.fringe_entry_price)
        Q_fringe = min(fringe_K, fringe_K * fringe_premium / max(p.fringe_entry_price, eps))
    else:
        Q_fringe = 0.0

    Q_total = Q_eff + Q_sub + Q_fringe

    I_next = max(0.0, s.I + dt * (Q_total - D) + shock.stockpile_release + pol.stockpile_release)

    tight = (D - Q_total) / max(D, eps)
    cover = I_next / max(D, eps)
    shortage = max(0.0, D - Q_total)

    # Euler-Maruyama with explicit noise z
    logP_next = (
        np.log(max(s.P, eps))
        + dt * p.alpha_P * (tight - p.lambda_cover * (cover - p.cover_star))
        + p.sigma_P * np.sqrt(dt) * z
    )
    P_next = max(float(np.exp(logP_next)), eps)

    K_star = max(b.K0 * (max(s.P, eps) / b.P_ref) ** p.eta_K * (1.0 + pol.subsidy) * (1.0 - shock.capex_shock), eps)
    build = max(0.0, K_star - s.K) / p.tau_K
    retire = p.retire_rate * s.K
    K_next = max(eps, s.K + dt * (build - retire))

    s_next = State(
        year=s.year + int(dt),
        t_index=s.t_index + 1,
        K=float(K_next),
        I=float(I_next),
        P=float(P_next),
    )
    res = StepResult(
        Q=float(Q), Q_eff=float(Q_eff), Q_sub=float(Q_sub),
        Q_fringe=float(Q_fringe), Q_total=float(Q_total),
        D=float(D), shortage=float(shortage), tight=float(tight), cover=float(cover),
    )
    return s_next, res


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

    Pure probability — no intervention, no causation.  "What if I see X=x?"
    Use for descriptive statistics and associations only.

    Note: P(Q_sub | export_restriction=high) from this function is NOT the causal
    effect of export restrictions on substitution supply.  It may be confounded by
    latent market structure (SubstitutionCapacity).  For causal effect use L2.

    Args:
        data: Observational dataset (e.g. Comtrade, run outputs).
        outcome: Column name for outcome variable.
        conditioning: Optional {column: value} equality filters.

    Returns:
        Series of outcome values in the filtered rows.
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


def observational_summary(
    data: pd.DataFrame,
    outcome: str,
    group_by: Optional[str] = None,
) -> pd.DataFrame:
    """
    Layer 1 — Association: Summary statistics of outcome (mean, std, count).

    Observational only.  No do(·).
    """
    if group_by:
        return data.groupby(group_by)[outcome].agg(["mean", "std", "count"]).reset_index()
    return pd.DataFrame({
        "mean": [data[outcome].mean()],
        "std": [data[outcome].std()],
        "count": [len(data)],
    })


def observe_substitution_association(data: pd.DataFrame) -> pd.DataFrame:
    """
    Layer 1 — Association: Summarise the observed relationship between
    Q_sub, export_restriction, and price_premium.

    Returns a DataFrame with columns:
        export_restricted (bool), mean_Q_sub, std_Q_sub,
        mean_price_premium, correlation_Q_sub_price_premium, count

    This is purely observational — correlation, not causation.
    To estimate the causal effect use do_substitution() (L2).
    """
    required = {"Q_sub", "shock_export_restriction", "P", "D"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Missing columns for substitution association: {missing}")

    df = data.copy()

    # Proxy for price premium: use P (absolute level; caller should normalise vs P_ref)
    # We include raw Q_sub correlation with shock_export_restriction
    df["export_restricted"] = df["shock_export_restriction"] > 0.0

    summary = (
        df.groupby("export_restricted")
        .agg(
            mean_Q_sub=("Q_sub", "mean"),
            std_Q_sub=("Q_sub", "std"),
            mean_shock_export_restriction=("shock_export_restriction", "mean"),
            count=("Q_sub", "count"),
        )
        .reset_index()
    )

    # Spearman correlation: Q_sub ~ export_restriction (observational association)
    if len(df) > 2:
        from scipy.stats import spearmanr
        rho, pval = spearmanr(df["shock_export_restriction"], df["Q_sub"])
        summary["spearman_rho_Q_sub_x_restriction"] = rho
        summary["spearman_pval"] = pval
    else:
        summary["spearman_rho_Q_sub_x_restriction"] = float("nan")
        summary["spearman_pval"] = float("nan")

    return summary


def observe_fringe_association(data: pd.DataFrame) -> pd.DataFrame:
    """
    Layer 1 — Association: Summarise the observed relationship between
    Q_fringe and the price level.

    Returns a DataFrame with Spearman ρ between P and Q_fringe, plus quartile
    binned means.  Purely observational — for causal effect use do_fringe_supply() (L2).
    """
    required = {"Q_fringe", "P"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Missing columns for fringe association: {missing}")

    df = data.copy()
    df["price_quartile"] = pd.qcut(df["P"], q=4, labels=["Q1", "Q2", "Q3", "Q4"], duplicates="drop")

    summary = (
        df.groupby("price_quartile", observed=True)
        .agg(
            mean_P=("P", "mean"),
            mean_Q_fringe=("Q_fringe", "mean"),
            std_Q_fringe=("Q_fringe", "std"),
            count=("Q_fringe", "count"),
        )
        .reset_index()
    )

    if len(df) > 2:
        from scipy.stats import spearmanr
        rho, pval = spearmanr(df["P"], df["Q_fringe"])
        summary["spearman_rho_Q_fringe_x_price"] = rho
        summary["spearman_pval"] = pval
    else:
        summary["spearman_rho_Q_fringe_x_price"] = float("nan")
        summary["spearman_pval"] = float("nan")

    return summary


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

    Uses do-calculus (backdoor/frontdoor criterion). Returns identification status
    and adjustment set if identifiable.  Does not compute the effect — use
    do_substitution() or do_fringe_supply() for that.
    """
    if dag is None:
        dag = GraphiteSupplyChainDAG()
    return dag.is_identifiable(treatment, outcome)


def mutilated_graph_for_do(dag: CausalDAG, node: str) -> CausalDAG:
    """
    Layer 2 — Intervention: Return the DAG after do(node).

    Graph surgery: remove all incoming edges to `node`.  The resulting
    mutilated graph has `node` as a root (no parents), representing the
    forced intervention.
    """
    return dag.remove_incoming_edges(node)


def do_substitution(
    cfg: ScenarioConfig,
    elasticity: float,
    cap: Optional[float] = None,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Layer 2 — Intervention: P(Y | do(substitution_elasticity=elasticity)).

    Graph surgery on the Price→SubstitutionSupply structural edge.
    Severing that edge means: elasticity is no longer determined by market
    conditions — it is forcibly pinned to the supplied value regardless of
    what supply-chain dynamics would ordinarily produce.

    Concretely: creates a mutilated ScenarioConfig with
        parameters.substitution_elasticity = elasticity
        parameters.substitution_cap       = cap (if provided)
    and runs the full simulation forward.

    The interventional distribution P(Q_total|do(elasticity=e)) differs from
    the observational P(Q_total|elasticity=e) whenever confounders exist
    (e.g. high-restriction periods also happen to have low substitution capacity).

    Args:
        cfg:        Base scenario config (factual parameter values).
        elasticity: Counterfactual elasticity to impose via do(·).
        cap:        Counterfactual substitution cap (None = keep factual cap).

    Returns:
        (df, metrics) under the interventional distribution.
    """
    from .simulate import run_scenario

    overrides: Dict[str, Any] = {"substitution_elasticity": elasticity}
    if cap is not None:
        overrides["substitution_cap"] = cap

    intervened_cfg = _override_params(cfg, **overrides)
    return run_scenario(intervened_cfg)


def do_fringe_supply(
    cfg: ScenarioConfig,
    capacity_share: float,
    entry_price: Optional[float] = None,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Layer 2 — Intervention: P(Y | do(fringe_capacity_share=capacity_share)).

    Graph surgery on the FringeCapacity→FringeSupply structural edge.
    Pins the fringe capacity and/or entry threshold to the given values,
    severing dependence on underlying market structure.

    Args:
        cfg:            Base scenario config.
        capacity_share: Counterfactual fringe capacity share to impose via do(·).
                        E.g. 0.3 = fringe can supply up to 30% of dominant capacity.
        entry_price:    Counterfactual normalised entry price (None = keep factual).
                        E.g. 1.5 = fringe enters at P > 1.5·P_ref.

    Returns:
        (df, metrics) under the interventional distribution.
    """
    from .simulate import run_scenario

    overrides: Dict[str, Any] = {"fringe_capacity_share": capacity_share}
    if entry_price is not None:
        overrides["fringe_entry_price"] = entry_price

    intervened_cfg = _override_params(cfg, **overrides)
    return run_scenario(intervened_cfg)


def do_compare(
    cfg: ScenarioConfig,
    intervention: Dict[str, float],
    outcomes: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Layer 2 — Intervention: Compare factual vs interventional trajectories.

    Runs both the factual scenario (cfg as-is) and the intervened scenario
    (cfg with parameters overridden by `intervention`) and returns a merged
    DataFrame with columns:
        year, <outcome>_factual, <outcome>_intervention, <outcome>_ate
    for each outcome in `outcomes`.

    The Average Treatment Effect (ATE) is the per-year difference
        E[Y|do(X=x)] − E[Y|do(X=x')]   (here just the single-trajectory difference).

    Args:
        cfg:          Factual scenario config.
        intervention: Dict of parameter overrides for do(·), e.g.
                      {"substitution_elasticity": 0.8}.
        outcomes:     Columns to compare (default: P, Q_total, shortage).

    Returns:
        Wide DataFrame with factual, interventional, and ATE columns.
    """
    from .simulate import run_scenario

    if outcomes is None:
        outcomes = ["P", "Q_total", "shortage"]

    factual_df, _ = run_scenario(cfg)
    intervened_cfg = _override_params(cfg, **intervention)
    intervened_df, _ = run_scenario(intervened_cfg)

    result = factual_df[["year"]].copy()
    for col in outcomes:
        if col in factual_df.columns and col in intervened_df.columns:
            result[f"{col}_factual"] = factual_df[col].values
            result[f"{col}_intervention"] = intervened_df[col].values
            result[f"{col}_ate"] = intervened_df[col].values - factual_df[col].values
    return result


# ---------------------------------------------------------------------------
# Layer 3: Counterfactual (Imagining) — P(Y_x | X', Y')
# ---------------------------------------------------------------------------


@dataclass
class CounterfactualResult:
    """
    Result of a Layer 3 counterfactual query.

    Attributes
    ----------
    factual : DataFrame
        Trajectory under the factual structural equations (observed world).
    counterfactual : DataFrame
        Trajectory under the counterfactual structural equations, with the
        SAME exogenous noise sequence U (twin network coupling).
    ate : dict
        Average Treatment Effect for key outcomes:
            ATE[outcome] = mean(cf[outcome]) - mean(factual[outcome])
        This is the average over years in the shared trajectory.
    description : str
        Human-readable description of the counterfactual query.
    noise_sequence : list[float]
        The abduced noise sequence ε_t used in both trajectories.
    """
    factual: pd.DataFrame
    counterfactual: pd.DataFrame
    ate: Dict[str, float]
    description: str
    noise_sequence: List[float]


def _compute_ate(factual: pd.DataFrame, counterfactual: pd.DataFrame, outcomes: List[str]) -> Dict[str, float]:
    """Compute average treatment effect E[Y_cf] - E[Y_factual] for each outcome."""
    ate = {}
    for col in outcomes:
        if col in factual.columns and col in counterfactual.columns:
            ate[col] = float((counterfactual[col] - factual[col]).mean())
    return ate


def counterfactual_substitution(
    cfg: ScenarioConfig,
    cf_elasticity: float,
    cf_cap: Optional[float] = None,
) -> CounterfactualResult:
    """
    Layer 3 — Counterfactual: P(Y_e | factual trajectory with elasticity=cfg.parameters.substitution_elasticity).

    "Given the trajectory that actually occurred under the current substitution_elasticity,
     what would prices / supply / shortage have been if elasticity had instead been cf_elasticity?"

    Implements Abduction-Action-Prediction:

    1. Abduction  — Run the factual scenario with cfg.seed, capturing the full
                    noise sequence ε_t = {ε_0, ..., ε_T}.  This is the exogenous
                    variable U inferred from the observed trajectory.

    2. Action     — Apply do(substitution_elasticity=cf_elasticity): sever the
                    Price→SubstitutionSupply edge and pin elasticity to cf_elasticity.
                    Also override cap if cf_cap is supplied.

    3. Prediction — Replay the simulation from the same initial state with the
                    SAME noise sequence ε_t and the modified structural equation.
                    This is the counterfactual trajectory Y_{e}.

    The twin-network coupling (shared ε_t) ensures that observed price noise is
    identical in both worlds — only the structural mechanism changes.

    Args:
        cfg:            Factual scenario config.  cfg.parameters.substitution_elasticity
                        is the FACTUAL elasticity (often 0.0).
        cf_elasticity:  Counterfactual elasticity to impose via do(·).
        cf_cap:         Counterfactual cap (None = keep factual cap).

    Returns:
        CounterfactualResult with factual and counterfactual DataFrames, ATE, and noise.
    """
    # Step 1: Abduction — run factual, capture noise sequence
    factual_df, noise_seq = _run_scenario_inner(cfg)

    # Step 2: Action — graph surgery on substitution parameters
    overrides: Dict[str, Any] = {"substitution_elasticity": cf_elasticity}
    if cf_cap is not None:
        overrides["substitution_cap"] = cf_cap
    cf_cfg = _override_params(cfg, **overrides)

    # Step 3: Prediction — replay with same noise ε_t, modified structural equation
    cf_df, _ = _run_scenario_inner(cf_cfg, noise_sequence=noise_seq)

    outcomes = ["P", "Q_total", "Q_sub", "shortage", "tight"]
    ate = _compute_ate(factual_df, cf_df, outcomes)

    factual_e = cfg.parameters.substitution_elasticity
    desc = (
        f"L3 counterfactual — SubstitutionSupply: "
        f"do(substitution_elasticity={cf_elasticity}) "
        f"vs factual elasticity={factual_e}. "
        f"Noise sequence abduced from seed={cfg.seed}."
    )

    return CounterfactualResult(
        factual=factual_df,
        counterfactual=cf_df,
        ate=ate,
        description=desc,
        noise_sequence=noise_seq,
    )


def counterfactual_fringe(
    cfg: ScenarioConfig,
    cf_capacity_share: float,
    cf_entry_price: Optional[float] = None,
) -> CounterfactualResult:
    """
    Layer 3 — Counterfactual: P(Y_f | factual trajectory with fringe_capacity_share=cfg.parameters.fringe_capacity_share).

    "Given the trajectory that actually occurred with no fringe supply,
     what would prices / supply / shortage have been if fringe producers
     with capacity_share=cf_capacity_share had been present?"

    Implements Abduction-Action-Prediction (twin network):

    1. Abduction  — Capture noise ε_t from factual run (same seed).
    2. Action     — do(fringe_capacity_share=cf_capacity_share), optionally
                    do(fringe_entry_price=cf_entry_price).
    3. Prediction — Replay with same ε_t, new fringe structural equation.

    Args:
        cfg:               Factual scenario config.
        cf_capacity_share: Counterfactual fringe capacity share (e.g. 0.3).
        cf_entry_price:    Counterfactual entry price threshold (e.g. 1.5).

    Returns:
        CounterfactualResult with factual and counterfactual DataFrames, ATE, and noise.
    """
    # Step 1: Abduction
    factual_df, noise_seq = _run_scenario_inner(cfg)

    # Step 2: Action
    overrides: Dict[str, Any] = {"fringe_capacity_share": cf_capacity_share}
    if cf_entry_price is not None:
        overrides["fringe_entry_price"] = cf_entry_price
    cf_cfg = _override_params(cfg, **overrides)

    # Step 3: Prediction
    cf_df, _ = _run_scenario_inner(cf_cfg, noise_sequence=noise_seq)

    outcomes = ["P", "Q_total", "Q_fringe", "shortage", "tight"]
    ate = _compute_ate(factual_df, cf_df, outcomes)

    factual_share = cfg.parameters.fringe_capacity_share
    desc = (
        f"L3 counterfactual — FringeSupply: "
        f"do(fringe_capacity_share={cf_capacity_share}) "
        f"vs factual share={factual_share}. "
        f"Noise sequence abduced from seed={cfg.seed}."
    )

    return CounterfactualResult(
        factual=factual_df,
        counterfactual=cf_df,
        ate=ate,
        description=desc,
        noise_sequence=noise_seq,
    )


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
# Summary
# ---------------------------------------------------------------------------


def three_layers_summary() -> str:
    """Return a structured summary of where each layer is implemented."""
    return """
Pearl's Three Layers (Ladder of Causation) — implementation map
================================================================

Layer 1 — Association (Seeing):  P(Y|X)
  "Among observations where X=x, what is Y?"
  Cannot answer causal questions — correlation only.

  Functions:
    observational_conditional(data, outcome, conditioning)
    observational_summary(data, outcome, group_by)
    observe_substitution_association(data)   → Spearman ρ(Q_sub, export_restriction)
    observe_fringe_association(data)         → Spearman ρ(Q_fringe, P) by price quartile

Layer 2 — Intervention (Doing):  P(Y|do(X))
  "If I surgically force X=x — severing all upstream causes — what is Y?"
  Implemented via graph surgery: mutilate DAG (cut incoming edges to X), run forward.
  P(Y|do(X=x)) ≠ P(Y|X=x) when confounders exist.

  Functions:
    interventional_identifiability(treatment, outcome, dag)  → backdoor/frontdoor check
    mutilated_graph_for_do(dag, node)                        → DAG after surgery
    do_substitution(cfg, elasticity, cap)                    → P(Y|do(substitution_elasticity=e))
    do_fringe_supply(cfg, capacity_share, entry_price)       → P(Y|do(fringe_capacity_share=f))
    do_compare(cfg, intervention, outcomes)                  → factual vs interventional table + ATE

Layer 3 — Counterfactual (Imagining):  P(Y_x | X'=x', Y'=y')
  "Given this specific observed trajectory (X=x', Y=y'), what would Y have been
   if X had been x?"
  Requires Abduction-Action-Prediction:
    1. Abduction:  infer exogenous noise U from observed trajectory (fix RNG seed)
    2. Action:     apply do(X=x) — same graph surgery as L2
    3. Prediction: replay simulation with same U, modified structural equation

  Functions:
    counterfactual_substitution(cfg, cf_elasticity, cf_cap)
        → CounterfactualResult: factual vs cf trajectory, ATE, noise_sequence
    counterfactual_fringe(cfg, cf_capacity_share, cf_entry_price)
        → CounterfactualResult: factual vs cf trajectory, ATE, noise_sequence
    counterfactual_step(state, cfg, year, do_shock_overrides, rng)
        → single-step counterfactual with shock override
    counterfactual_trajectory(state_0, cfg, years, overrides_by_year, rng)
        → multi-step counterfactual trajectory

Noise abduction:
  Exogenous U = price noise sequence {ε_t} from Euler-Maruyama.
  step() draws exactly one ε_t = rng.normal() per timestep.
  Fixing cfg.seed ensures identical ε_t in factual and counterfactual worlds.
  _run_scenario_inner() captures and replays this sequence explicitly.
"""
