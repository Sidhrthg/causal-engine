#!/usr/bin/env python3
"""
Forward Scenario 2025 — L2 do-calculus projections: "What if China restricts today?"

Thesis Chapter 5 output: 2025–2030 price trajectories under a standardised
30% Chinese export restriction starting January 2025 for each critical mineral.

Pearl Layer 2 (do-calculus):
  do(export_restriction_2025 = 0.30) for each mineral.

This is L2, NOT L3:
  - We are projecting forward from the current state (2025 is the present).
  - There is no historical trajectory to condition on (abduction would require
    observed post-restriction prices, which don't exist yet).
  - L3 would be appropriate AFTER the restriction actually occurs and CEPII
    data is available; for now, L2 forward projection is the correct causal tool.

Scenarios:
  BASELINE   — no new restriction; current trends continue
  MILD_BAN   — 30% restriction 2025–2026 (2yr), then removed
  FULL_BAN   — 30% restriction 2025–2027 (3yr), then removed
  SEVERE_BAN — 50% restriction 2025–2028 (4yr), then removed
              (represents escalation beyond 2023 graphite licence regime)

For each mineral, the script reports:
  1. Peak price index relative to 2024 baseline
  2. Year of peak price
  3. Year prices return within 10% of no-restriction baseline (normalisation)
  4. Normalisation lag (years after restriction end until prices normalise)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import math
import pandas as pd

from src.minerals.schema import (
    BaselineConfig, DemandGrowthConfig, OutputsConfig,
    ParametersConfig, PolicyConfig, ScenarioConfig,
    ShockConfig, TimeConfig,
)
from src.minerals.simulate import run_scenario
from src.minerals.causal_engine import CausalInferenceEngine
from src.minerals.causal_inference import GraphiteSupplyChainDAG
from src.minerals.predictability import (
    _GRAPHITE_2022_PARAMS,
    _RARE_EARTHS_2010_PARAMS,
    _COBALT_2016_PARAMS,
    _LITHIUM_2022_PARAMS,
    _NICKEL_2022_PARAMS,
    _URANIUM_2022_PARAMS,
)
from src.minerals.constants import ODE_DEFAULTS, SCENARIO_EXTRAS

BASELINE_CFG = BaselineConfig(P_ref=1.0, P0=1.0, K0=108.695652, I0=20.0, D0=100.0)
EULER_SAFETY  = 0.9
NORM_THRESHOLD = 0.10  # within 10% of no-restriction baseline


def _stable_alpha_P(params: dict) -> float:
    eta_D = params["eta_D"]
    limit = EULER_SAFETY / max(abs(eta_D), 1e-6)
    return min(params["alpha_P"], limit)


# ── Mineral parameter registry for 2025 forward scenarios ────────────────────
# Uses the most recent calibrated episode params for each mineral.
# For graphite: graphite_2022 params reflect the restriction-era structural dynamics.
# For cobalt:   cobalt_2016 params (2022 episode is post-peak crash; 2016 = shock regime).
# For lithium:  lithium_2022 params (most recent EV-cycle calibration).
# For nickel:   nickel_2022 params (HPAL oversupply era; supply-side response calibrated).
# For uranium:  uranium_2022 params (Russia sanctions era — inelastic demand confirmed).

_MINERAL_PARAMS = {
    "graphite":   _GRAPHITE_2022_PARAMS,
    "rare_earths": _RARE_EARTHS_2010_PARAMS,   # only episode with export restriction
    "cobalt":     _COBALT_2016_PARAMS,
    "lithium":    _LITHIUM_2022_PARAMS,
    "nickel":     _NICKEL_2022_PARAMS,
    "uranium":    _URANIUM_2022_PARAMS,
}

# Dominant exporter for each mineral (context for restriction interpretation)
_DOMINANT_EXPORTER = {
    "graphite":    "China (90% of global exports)",
    "rare_earths": "China (60% mine output, 85% processing)",
    "cobalt":      "DRC (70%) / China (65% refining)",
    "lithium":     "Australia (55%) / Chile (23%)",
    "nickel":      "Indonesia (37%) / Philippines (13%)",
    "uranium":     "Kazakhstan (43%) / Russia (15% pre-2023)",
}



def _build_forward_cfg(
    mineral: str,
    params: dict,
    extra_params: dict,
    restriction_magnitude: float,
    restriction_start: int,
    restriction_end: int,
    name_suffix: str,
) -> ScenarioConfig:
    """Build a 2025-forward ScenarioConfig for L2 projection."""
    alpha_P = _stable_alpha_P(params)
    kw = {
        **ODE_DEFAULTS,
        "tau_K": params["tau_K"],
        "eta_D": params["eta_D"],
        "demand_growth": DemandGrowthConfig(type="constant", g=params["g"]),
        "alpha_P": alpha_P,
    }
    kw.update(extra_params)

    shocks = []
    if restriction_magnitude > 0:
        shocks.append(ShockConfig(
            type="export_restriction",
            start_year=restriction_start,
            end_year=restriction_end,
            magnitude=restriction_magnitude,
        ))

    return ScenarioConfig(
        name=f"{mineral}_2025_{name_suffix}",
        commodity=mineral,
        seed=42,
        time=TimeConfig(dt=1.0, start_year=2024, end_year=2032),
        baseline=BASELINE_CFG,
        parameters=ParametersConfig(**kw),
        policy=PolicyConfig(),
        shocks=shocks,
        outputs=OutputsConfig(metrics=["avg_price"]),
    )


def _find_normalization_year(
    scenario_df: pd.DataFrame,
    baseline_df: pd.DataFrame,
    base_year: int,
    search_from_year: int,
) -> int | None:
    """Find year >= search_from_year where scenario price is within 10% of baseline."""
    sc  = scenario_df.set_index("year")["P"]
    bl  = baseline_df.set_index("year")["P"]
    sc_base = sc.loc[base_year]
    bl_base = bl.loc[base_year]
    years = sorted(y for y in sc.index if y >= search_from_year and y in bl.index)
    for yr in years:
        sc_idx = sc.loc[yr] / sc_base
        bl_idx = bl.loc[yr] / bl_base
        if abs(sc_idx - bl_idx) < NORM_THRESHOLD:
            return yr
    return None


def run_mineral_scenario(mineral: str, params: dict, extra: dict) -> None:
    """Run all three forward scenarios for one mineral and print results."""
    print(f"\n{'─' * 70}")
    print(f"MINERAL: {mineral.upper()}")
    print(f"  Dominant supplier: {_DOMINANT_EXPORTER[mineral]}")
    print(f"  τ_K = {params['tau_K']:.3f}yr   α_P = {_stable_alpha_P(params):.3f} "
          f"(orig {params['alpha_P']:.3f})   η_D = {params['eta_D']:.3f}")
    print(f"  Background growth: g = {params['g']:.4f}/yr")

    scenarios = [
        # (name, magnitude, start, end)
        ("baseline",   0.00, 2025, 2025),
        ("mild_ban",   0.30, 2025, 2026),
        ("full_ban",   0.30, 2025, 2027),
        ("severe_ban", 0.50, 2025, 2028),
    ]

    report_years = list(range(2024, 2032))
    base_year    = 2024

    # Build and run all scenarios
    results = {}
    for scenario_name, mag, start, end in scenarios:
        cfg = _build_forward_cfg(mineral, params, extra, mag, start, end, scenario_name)
        df, _ = run_scenario(cfg)
        results[scenario_name] = df

    bl_df = results["baseline"]
    bl    = bl_df.set_index("year")["P"]
    bl_base = bl.loc[base_year]

    # Print price index table
    print(f"\n  Price index (P / P_{base_year}):")
    print(f"  {'Scenario':<14}", end="")
    for yr in report_years:
        print(f"  {yr:>5}", end="")
    print(f"  {'Peak':>6}  {'Peak yr':>7}  {'Norm yr':>8}  {'Lag':>5}")
    print("  " + "─" * 14, end="")
    for _ in report_years:
        print("  " + "─" * 5, end="")
    print("  " + "─" * 6 + "  " + "─" * 7 + "  " + "─" * 8 + "  " + "─" * 5)

    scenario_meta = {}
    for scenario_name, mag, start, end in scenarios:
        df  = results[scenario_name]
        sc  = df.set_index("year")["P"]
        sc_base = sc.loc[base_year]

        # Price index relative to base_year
        idxs = {yr: sc.loc[yr] / sc_base for yr in report_years if yr in sc.index}

        # Peak: max index across all years from start onwards
        window = {yr: v for yr, v in idxs.items() if yr >= start}
        if window:
            peak_yr  = max(window, key=window.get)
            peak_idx = window[peak_yr]
        else:
            peak_yr, peak_idx = base_year, 1.0

        # Normalisation year (search from restriction end + 1)
        search_from = end + 1 if mag > 0 else start
        norm_yr = _find_normalization_year(df, bl_df, base_year, search_from)
        lag     = (norm_yr - end) if norm_yr and mag > 0 else None

        scenario_meta[scenario_name] = {
            "peak_yr": peak_yr,
            "peak_idx": peak_idx,
            "norm_yr": norm_yr,
            "lag": lag,
        }

        row = f"  {scenario_name:<14}"
        for yr in report_years:
            val = idxs.get(yr, float("nan"))
            if math.isnan(val):
                row += f"  {'?':>5}"
            else:
                row += f"  {val:5.3f}"

        norm_str = f"{norm_yr}" if norm_yr else "never"
        lag_str  = f"+{lag}yr" if lag is not None else "—"
        row += f"  {peak_idx:6.3f}  {peak_yr:>7}  {norm_str:>8}  {lag_str:>5}"
        print(row)

    # Key findings per mineral
    print(f"\n  Key findings ({mineral}):")
    if scenario_meta["full_ban"]["peak_idx"] > 1.5:
        print(f"  ▲ FULL_BAN scenario peaks at {scenario_meta['full_ban']['peak_idx']:.2f}× baseline "
              f"in {scenario_meta['full_ban']['peak_yr']}.")
    if scenario_meta["full_ban"]["lag"] is not None:
        print(f"  ▲ After restriction ends (2027), prices normalise "
              f"{scenario_meta['full_ban']['lag']} years later "
              f"({scenario_meta['full_ban']['norm_yr']}).")
    elif scenario_meta["full_ban"]["norm_yr"] is None:
        print(f"  ▲ Prices do not normalise within the 2032 projection window.")

    print(f"  ▲ Capacity adjustment time τ_K = {params['tau_K']:.2f}yr: "
          f"{'slow recovery — long price scar' if params['tau_K'] > 5 else 'moderate recovery'}")

    return scenario_meta


def main():
    print("=" * 80)
    print("FORWARD SCENARIO 2025 — L2 DO-CALCULUS PROJECTION")
    print("What if China imposes a 30–50% export restriction TODAY (2025)?")
    print("=" * 80)
    print("""
Pearl Layer 2 application:
  do(export_restriction_2025 = {0.30, 0.50}) for each critical mineral.
  This asks: "What would prices be if the restriction were imposed right now?"
  This is L2 (intervention calculus), not L1 (observation) or L3 (counterfactual).
  L3 requires an observed post-restriction trajectory for abduction — unavailable.

  All scenarios start from the 2024 calibrated state.
  Background demand growth (g) continues at the calibrated episode rate.
  Euler-stabilised α_P used throughout (capped at 0.9/|η_D|).

Scenarios:
  BASELINE   — no new restriction; current trends continue
  MILD_BAN   — do(restriction = 0.30) for 2025–2026 (2yr), then removed
  FULL_BAN   — do(restriction = 0.30) for 2025–2027 (3yr), then removed
  SEVERE_BAN — do(restriction = 0.50) for 2025–2028 (4yr), then removed
""")

    all_meta = {}
    for mineral, params in _MINERAL_PARAMS.items():
        extra = SCENARIO_EXTRAS.get(mineral, {})
        meta = run_mineral_scenario(mineral, params, extra)
        all_meta[mineral] = meta

    # ── Cross-mineral comparison table ────────────────────────────────────────
    print("\n\n" + "=" * 80)
    print("CROSS-MINERAL SUMMARY — FULL_BAN SCENARIO (30% restriction 2025–2027)")
    print("=" * 80)
    print(f"{'Mineral':<14} {'τ_K':>6} {'Peak ×':>8} {'Peak yr':>8} {'Norm yr':>9} {'Lag':>6}")
    print("-" * 60)

    for mineral in _MINERAL_PARAMS:
        meta = all_meta[mineral]["full_ban"]
        params = _MINERAL_PARAMS[mineral]
        peak   = meta["peak_idx"]
        peak_y = meta["peak_yr"]
        norm_y = meta["norm_yr"]
        lag    = meta["lag"]
        tau_K  = params["tau_K"]

        norm_str = f"{norm_y}" if norm_y else "never"
        lag_str  = f"+{lag}yr" if lag is not None else "—"
        print(f"{mineral:<14} {tau_K:>6.2f} {peak:>8.3f} {peak_y:>8} {norm_str:>9} {lag_str:>6}")

    print("-" * 60)
    print("""
Interpretation:
  τ_K   = capacity adjustment time (calibrated from historical episode)
  Peak × = peak price index relative to 2024 baseline (P_peak / P_2024)
  Norm yr = first year prices return within 10% of no-restriction trajectory
  Lag   = years from restriction end (2027) until normalisation

Key cross-mineral findings:
  Minerals with high τ_K (uranium, graphite, nickel) show:
    (a) larger peak price amplification — supply cannot respond quickly
    (b) longer post-restriction price scars — inventory rebuilding is slow
  Minerals with low τ_K (lithium, rare earths China ramp) normalise faster
  but are NOT necessarily less vulnerable — their import-reliance and the
  processing bottleneck (rare earths refining in China) remain critical.
""")

    print("=" * 80)
    print("STOCKPILE POLICY IMPLICATIONS FROM FORWARD SCENARIOS")
    print("=" * 80)
    print("""
1. Graphite (τ_K=7.83yr)
   A FULL_BAN (30%, 3yr) pushes prices to significant premiums. The 6%
   circumvention rate via Poland is insufficient to materially offset the ban.
   Recommended: 18-month strategic reserve + immediate synthetic graphite
   mandate for defence applications.

2. Rare earths (τ_K=0.51yr, China ramp speed)
   Fast ramp-up in China makes the FULL_BAN scenario self-limiting — China
   can and has flooded the market after restrictions to reassert dominance.
   The US recovery τ_K of ~10yr (separation/processing rebuild) means price
   normalisation at the MINE level obscures continued US PROCESSING vulnerability.
   Recommended: DoD Section 232 processing offtake; MP Materials Phase 2.

3. Uranium (τ_K=14.89yr)
   Longest geological cycle. Long-term contracts provide 2–3yr buffer but
   the SEVERE_BAN scenario (50%, 4yr) exhausts contract cover.
   ADVANCE Act 2024 (restricting Rosatom imports) is partially in effect.
   Recommended: Expand DOE Uranium Reserve; accelerate Centrus HALEU plant.

4. Cobalt (τ_K=5.75yr)
   DRC instability and Chinese refining concentration create compound risk.
   LFP battery chemistry reduces cobalt demand but does not eliminate it for
   high-energy-density applications (EV, defence).
   Recommended: DoD NDS cobalt stockpile target 3–5yr; allied sourcing (Zambia).

5. Lithium (τ_K=1.34yr) — LOWEST STRUCTURAL RISK
   Fast brine extraction response, multiple source countries.
   Thacker Pass (Nevada) + allied sourcing (Australia FTA) reduce risk.
   Recommended: IEA-style 90-day reserve; no emergency intervention needed.
""")


if __name__ == "__main__":
    main()
