#!/usr/bin/env python3
"""
Forward Scenario 2026 — Germanium and Gallium under China export controls.

China imposed export licence requirements on germanium (HS 2804.70) and
gallium (HS 2805.30) in August 2023. This script runs L2 do-calculus
forward projections for 2026-2032 under three escalation scenarios.

PROVISIONAL — uses parameter PRIORS from rare_earths_2010 analog.
Replace with differential evolution calibration once CEPII data lands.
See `data/canonical/REQUIRED_DATA.md` for download instructions.

Pearl Layer 2 application:
  do(export_restriction_2026 = {0.30, 0.50, 0.80}) for Ge and Ga.
  L3 unavailable (no observed post-restriction CEPII trajectory for these
  HS codes; Aug 2023 episode is too recent + too sparsely covered).

Why these minerals fit the framework:
  - China supply share 80%+ at production stage
  - Highly inelastic short-run demand (IR optics for Ge; GaN power
    electronics, defense radar, 5G for Ga)
  - No fringe supply at near-current prices
  - Byproduct structure: Ge from Zn smelting, Ga from Al refining
    → tau_K determined by primary metal capacity, not direct mining

Why these results are PROVISIONAL:
  - Parameters are priors, not CEPII-calibrated
  - Forward peaks should be read as "what the rare_earths_2010-analogous
    regime predicts", not as point forecasts
  - Sensitivity to prior choice is the dominant uncertainty
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
    _GERMANIUM_2023_PRIOR, _GALLIUM_2023_PRIOR,
)
from src.minerals.constants import (
    ODE_DEFAULTS, SCENARIO_EXTRAS,
    US_IMPORT_RELIANCE, CIRCUMVENTION_RATE,
)

BASELINE_CFG = BaselineConfig(P_ref=1.0, P0=1.0, K0=108.695652, I0=20.0, D0=100.0)
EULER_SAFETY = 0.9


def _stable_alpha_P(params):
    eta_D = params["eta_D"]
    return min(params["alpha_P"], EULER_SAFETY / max(abs(eta_D), 1e-6))


def _build_cfg(commodity: str, params: dict, magnitude: float,
               start_year: int, end_year: int, name_suffix: str) -> ScenarioConfig:
    alpha_P = _stable_alpha_P(params)
    kw = {**ODE_DEFAULTS,
          "tau_K": params["tau_K"], "eta_D": params["eta_D"],
          "demand_growth": DemandGrowthConfig(type="constant", g=params["g"]),
          "alpha_P": alpha_P,
          **SCENARIO_EXTRAS.get(commodity, {})}
    shocks = []
    if magnitude > 0:
        shocks.append(ShockConfig(
            type="export_restriction",
            start_year=start_year, end_year=end_year,
            magnitude=magnitude,
            country="China",
        ))
    return ScenarioConfig(
        name=f"{commodity}_2026_{name_suffix}",
        commodity=commodity, seed=42,
        time=TimeConfig(dt=1.0, start_year=2025, end_year=2032),
        baseline=BASELINE_CFG,
        parameters=ParametersConfig(**kw),
        policy=PolicyConfig(),
        shocks=shocks,
        outputs=OutputsConfig(metrics=["avg_price"]),
    )


def run_commodity(commodity: str, params: dict, label: str):
    print(f"\n{'─' * 72}")
    print(f"COMMODITY: {commodity.upper()}  ({label})")
    print(f"  α_P = {_stable_alpha_P(params):.3f} (orig {params['alpha_P']:.3f})  "
          f"τ_K = {params['tau_K']:.2f}yr  η_D = {params['eta_D']:.3f}  "
          f"g = {params['g']:.3f}/yr")
    print(f"  US import reliance: {US_IMPORT_RELIANCE[commodity]*100:.0f}%  "
          f"Circumvention rate: {CIRCUMVENTION_RATE[commodity]*100:.0f}%")

    scenarios = [
        ("baseline",   0.00, 2026, 2026),
        ("mild_ban",   0.30, 2026, 2027),  # 2yr 30%
        ("full_ban",   0.50, 2026, 2028),  # 3yr 50% (matches China Aug 2023 documented intent)
        ("severe_ban", 0.80, 2026, 2029),  # 4yr 80% (escalation tail risk)
    ]

    report_years = list(range(2025, 2033))
    base_year = 2025

    results = {}
    for sname, mag, start, end in scenarios:
        cfg = _build_cfg(commodity, params, mag, start, end, sname)
        df, _ = run_scenario(cfg)
        results[sname] = df

    bl_df = results["baseline"]
    bl_base = bl_df.set_index("year")["P"].loc[base_year]

    print(f"\n  Price index (P / P_{base_year}):")
    print(f"  {'Scenario':<12}", end="")
    for yr in report_years:
        print(f"  {yr:>5}", end="")
    print(f"  {'Peak':>6}  {'Peak yr':>7}")
    print("  " + "─" * 12 + "  " + "─" * 5 * len(report_years) + "  " + "─" * 14)

    for sname, mag, start, end in scenarios:
        df = results[sname]
        sc = df.set_index("year")["P"]
        sc_base = sc.loc[base_year]
        idxs = {yr: sc.loc[yr] / sc_base for yr in report_years if yr in sc.index}
        window = {yr: v for yr, v in idxs.items() if yr >= start}
        peak_yr = max(window, key=window.get) if window else base_year
        peak = window[peak_yr] if window else 1.0

        row = f"  {sname:<12}"
        for yr in report_years:
            v = idxs.get(yr, float("nan"))
            row += f"  {v:5.2f}"
        row += f"  {peak:6.2f}  {peak_yr:>7}"
        print(row)


def main():
    print("=" * 78)
    print("FORWARD SCENARIO 2026 — GERMANIUM & GALLIUM (PROVISIONAL)")
    print("China Aug 2023 export controls; what if escalation in 2026?")
    print("=" * 78)
    print("""
PROVISIONAL: Parameters are priors from rare_earths_2010 analog regime.
Calibration against CEPII HS 2804.70 (Ge) and 2805.30 (Ga) is pending.

Pearl L2 forward projection — do(export_restriction = m) for 2026+ window.
L3 unavailable (insufficient post-Aug-2023 CEPII coverage at HS-6 level).

Scenarios:
  BASELINE   — no new restriction beyond Aug 2023 status quo
  MILD_BAN   — 30% restriction 2026-2027 (2yr)
  FULL_BAN   — 50% restriction 2026-2028 (3yr, matches Aug 2023 intent)
  SEVERE_BAN — 80% restriction 2026-2029 (4yr escalation tail)
""")

    run_commodity("germanium", _GERMANIUM_2023_PRIOR,
                  "China ~80% production share; byproduct of Zn smelting; τ_K ≈ 12yr")
    run_commodity("gallium", _GALLIUM_2023_PRIOR,
                  "China ~85% production share; byproduct of Al refining; τ_K ≈ 6yr")

    print("\n" + "=" * 78)
    print("INTERPRETATION (PROVISIONAL — sensitive to prior choice)")
    print("=" * 78)
    print("""
1. Germanium: τ_K = 12yr means Zn smelter capacity expansion is the binding
   supply response timeline. No new mine project can offset within the
   restriction window (2026-2029). US strategic exposure is greatest because
   IR optics for missile seekers and thermal imaging have no substitute.
   Recommended planning floor: 5-year strategic reserve at FULL_BAN consumption.

2. Gallium: τ_K = 6yr means Al refining capacity (which co-produces Ga as
   waste-stream recovery) is the binding constraint. Japan has secondary Ga
   recovery but at <10% of US demand. SEVERE_BAN scenario fully exhausts
   the implicit contract buffer. GaN power semiconductors and 5G/defense
   radar applications have no near-term substitute.

3. Compound scenario risk: GaAs and GaN devices use BOTH Ge (window/substrate)
   and Ga (active layer). A simultaneous 50% restriction on both produces a
   compound shock to US defense semiconductor supply chains that single-mineral
   analysis does not capture. This is research_questions.md §7.4 "compound
   scenario" question and requires a multi-commodity coupling extension to
   the ODE — not addressed in this provisional forward.

LIMITATIONS:
  - Parameter priors borrowed from rare_earths_2010 (defensible analog but
    untested on Ge/Ga price data)
  - τ_K values are first-principles estimates from primary metal cycles
  - No L3 abduction (no CEPII coverage post-Aug 2023)
  - Forward peaks are L2 structural projections, not point forecasts;
    sensitivity to α_P prior is the dominant uncertainty source
  - When CEPII data lands, calibrate via differential evolution and re-run;
    the prior-vs-calibrated comparison is itself a defensible result
""")


if __name__ == "__main__":
    main()
