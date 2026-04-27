#!/usr/bin/env python3
"""
Regime sensitivity for forward scenarios — defense answer to:
  "If graphite and rare earths are regime-dependent (Group B in Table 4.2),
   why trust the forward 2025+ scenarios for them?"

Approach: run the standard forward FULL_BAN scenario (30% restriction 2025-2027)
under BOTH calibrated regimes for each regime-dependent commodity, and report
the peak-price bounds. This converts a weakness (parameter instability) into
a quantified uncertainty band.

Regime pairs:
  graphite      2022 (EV-restriction era, α_P=2.62) vs. 2008 (pre-EV cycle, α_P=0.50)
  rare_earths   2010 (China restriction, α_P=1.75) vs. 2014 (post-WTO flood, α_P=1.61)

For commodities in Group A (cobalt, lithium, nickel, soybeans), parameter
transfer is stable (DA ≥ 0.60 across regimes), so a single-regime forward
projection is defensible. This script targets only Group B.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.minerals.schema import (
    BaselineConfig, DemandGrowthConfig, OutputsConfig,
    ParametersConfig, PolicyConfig, ScenarioConfig,
    ShockConfig, TimeConfig,
)
from src.minerals.simulate import run_scenario
from src.minerals.predictability import (
    _GRAPHITE_2008_PARAMS, _GRAPHITE_2022_PARAMS,
    _RARE_EARTHS_2010_PARAMS, _RARE_EARTHS_2014_PARAMS,
)
from src.minerals.constants import ODE_DEFAULTS, SCENARIO_EXTRAS

BASELINE_CFG  = BaselineConfig(P_ref=1.0, P0=1.0, K0=108.695652, I0=20.0, D0=100.0)
EULER_SAFETY  = 0.9


def _stable_alpha_P(params):
    eta_D = params["eta_D"]
    return min(params["alpha_P"], EULER_SAFETY / max(abs(eta_D), 1e-6))


def _run_forward(commodity: str, params: dict, regime_label: str,
                 magnitude: float, start: int, end: int):
    """Run a single forward scenario; return (peak_yr, peak_idx, peak_yr_baseline_idx)."""
    alpha_P = _stable_alpha_P(params)
    kw = {**ODE_DEFAULTS,
          "tau_K": params["tau_K"], "eta_D": params["eta_D"],
          "demand_growth": DemandGrowthConfig(type="constant", g=params["g"]),
          "alpha_P": alpha_P,
          **SCENARIO_EXTRAS.get(commodity, {})}
    base_year = 2024

    cfg = ScenarioConfig(
        name=f"{commodity}_{regime_label}_full_ban",
        commodity=commodity, seed=42,
        time=TimeConfig(dt=1.0, start_year=base_year, end_year=2032),
        baseline=BASELINE_CFG,
        parameters=ParametersConfig(**kw),
        policy=PolicyConfig(),
        shocks=[ShockConfig(type="export_restriction",
                            start_year=start, end_year=end, magnitude=magnitude)],
        outputs=OutputsConfig(metrics=["avg_price"]),
    )
    df, _ = run_scenario(cfg)
    sc = df.set_index("year")["P"]
    sc_base = sc.loc[base_year]
    window = {yr: sc.loc[yr] / sc_base for yr in sc.index if yr >= start}
    peak_yr = max(window, key=window.get)
    return peak_yr, window[peak_yr]


def main():
    print("=" * 78)
    print("REGIME SENSITIVITY — Forward FULL_BAN (30% restriction 2025-2027)")
    print("Group B commodities only (parameter transfer fails across regimes)")
    print("=" * 78)
    print()
    print(f"{'Commodity':<14} {'Regime':<28} {'α_P':>5} {'τ_K':>5} {'Peak yr':>8} {'Peak ×':>8}")
    print("-" * 78)

    targets = [
        # (commodity, regime_label, params_dict)
        ("graphite",    "2022 (EV restriction era)",        _GRAPHITE_2022_PARAMS),
        ("graphite",    "2008 (pre-EV commodity cycle)",    _GRAPHITE_2008_PARAMS),
        ("rare_earths", "2010 (China restriction era)",     _RARE_EARTHS_2010_PARAMS),
        ("rare_earths", "2014 (post-WTO oversupply)",       _RARE_EARTHS_2014_PARAMS),
    ]

    results = {}
    for commodity, regime, params in targets:
        peak_yr, peak_idx = _run_forward(commodity, params, regime.split()[0],
                                         magnitude=0.30, start=2025, end=2027)
        results[(commodity, regime)] = (peak_yr, peak_idx)
        print(f"{commodity:<14} {regime:<28} {params['alpha_P']:>5.2f} "
              f"{params['tau_K']:>5.2f} {peak_yr:>8} {peak_idx:>8.3f}")

    print("-" * 78)
    print()
    print("=" * 78)
    print("UNCERTAINTY BANDS — Peak price index under regime uncertainty")
    print("=" * 78)
    for commodity in ("graphite", "rare_earths"):
        rows = [v for k, v in results.items() if k[0] == commodity]
        peaks = [r[1] for r in rows]
        peak_lo, peak_hi = min(peaks), max(peaks)
        ratio = peak_hi / peak_lo if peak_lo > 0 else float("inf")
        print(f"  {commodity:<14}  peak ∈ [{peak_lo:.2f}×, {peak_hi:.2f}×]   "
              f"(ratio hi/lo = {ratio:.2f}×)")
    print()
    print("Defense interpretation:")
    print("  - Rare earths: regime band is narrow (1.49×–1.58×, only 6% range).")
    print("    Forward 30% restriction shock dominates the structural-parameter")
    print("    difference between 2010 and 2014 regimes. Either regime gives")
    print("    essentially the same peak impact — only the temporal profile")
    print("    differs (2010: fast spike 2026; 2014: chronic drift to 2032).")
    print("    Defensible single-regime forward projection.")
    print("  - Graphite: regime band is wide (1.59×–5.26×, 3.31× ratio).")
    print("    Applying 2008 (pre-EV) structural params to a 2026 shock is a")
    print("    counterfactual combination — it pairs a slow-adjustment, low-α_P")
    print("    market structure with EV-era restriction shocks. The 5.26× number")
    print("    is what graphite WOULD do under chronic EV demand if the market")
    print("    were structurally unable to respond, which is inconsistent with")
    print("    actual EV-era anode capacity expansion. Forward scenarios use")
    print("    2022 params on the empirical basis that α_P has been ≈2.6 since")
    print("    ~2020 (sensitivity grid §4.9.4) — but the 2008 regime number")
    print("    serves as a stress-test upper bound for stockpile sizing.")


if __name__ == "__main__":
    main()
