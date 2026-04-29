#!/usr/bin/env python3
"""
Byproduct donor sweep — apply EACH calibrated mineral regime to germanium
and gallium forward scenarios, report peak-price band.

This generalises the rare_earths_2010 analog used in
`scripts/byproduct_forward_2026.py` to a full cross-commodity transfer
sweep, mirroring the methodology in `scripts/regime_sensitivity.py` for
Group B commodities.

Output: for each (target ∈ {germanium, gallium}, donor ∈ 12 regimes),
the peak price under FULL_BAN (50% restriction 2026-2028) and SEVERE_BAN
(80% restriction 2026-2029). The min/max across donors gives a defensible
peak-price band when no calibration is available for the target commodity.

Methodology: thesis §6.3.5 zero-shot regime priors.
Theoretical basis: structural parameters (α_P, η_D, τ_K, g) reflect
commodity-level physics and may transfer across minerals sharing supply
concentration + demand inelasticity. The transfer test in cross_commodity_transfer.py
shows DA up to 0.667 for cross-mineral transfer when donor and target
share regime characteristics.

Why donors matter for byproduct minerals:
  Ge and Ga are byproducts (Zn smelting, Al refining respectively) — their
  τ_K and α_P cannot be calibrated independently before sufficient post-2023
  CEPII data accumulates. The donor sweep gives a defensible UNCERTAINTY
  RANGE rather than a point estimate dependent on a single donor choice.
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
    _COBALT_2016_PARAMS, _COBALT_2022_PARAMS,
    _LITHIUM_2016_PARAMS, _LITHIUM_2022_PARAMS,
    _NICKEL_2006_PARAMS, _NICKEL_2022_PARAMS,
    _URANIUM_2007_PARAMS, _URANIUM_2022_PARAMS,
)
from src.minerals.constants import ODE_DEFAULTS, SCENARIO_EXTRAS

BASELINE_CFG = BaselineConfig(P_ref=1.0, P0=1.0, K0=108.695652, I0=20.0, D0=100.0)
EULER_SAFETY = 0.9

DONORS = [
    ("graphite_2008",     _GRAPHITE_2008_PARAMS,     "pre-EV cycle"),
    ("graphite_2022",     _GRAPHITE_2022_PARAMS,     "EV restriction era"),
    ("rare_earths_2010",  _RARE_EARTHS_2010_PARAMS,  "China quota era"),
    ("rare_earths_2014",  _RARE_EARTHS_2014_PARAMS,  "post-WTO oversupply"),
    ("cobalt_2016",       _COBALT_2016_PARAMS,       "EV speculation"),
    ("cobalt_2022",       _COBALT_2022_PARAMS,       "LFP transition"),
    ("lithium_2016",      _LITHIUM_2016_PARAMS,      "EV first wave"),
    ("lithium_2022",      _LITHIUM_2022_PARAMS,      "EV demand boom"),
    ("nickel_2006",       _NICKEL_2006_PARAMS,       "stainless boom"),
    ("nickel_2022",       _NICKEL_2022_PARAMS,       "HPAL crash"),
    ("uranium_2007",      _URANIUM_2007_PARAMS,      "Cigar Lake supply squeeze"),
    ("uranium_2022",      _URANIUM_2022_PARAMS,      "Russia sanctions / PRIA"),
]

TARGETS = ["germanium", "gallium"]

SCENARIOS = [
    ("FULL_BAN",   0.50, 2026, 2028),  # 3yr 50% — Aug 2023 China policy intent
    ("SEVERE_BAN", 0.80, 2026, 2029),  # 4yr 80% — escalation tail
]


def _stable_alpha_P(params):
    return min(params["alpha_P"], EULER_SAFETY / max(abs(params["eta_D"]), 1e-6))


def run_one(target: str, donor_params: dict, magnitude: float,
            start_year: int, end_year: int) -> tuple[float, int]:
    """Run one (target, donor, scenario) combination; return (peak_idx, peak_year)."""
    alpha_P = _stable_alpha_P(donor_params)
    kw = {**ODE_DEFAULTS,
          "tau_K": donor_params["tau_K"], "eta_D": donor_params["eta_D"],
          "demand_growth": DemandGrowthConfig(type="constant", g=donor_params["g"]),
          "alpha_P": alpha_P,
          **SCENARIO_EXTRAS.get(target, {})}
    cfg = ScenarioConfig(
        name=f"{target}_donor_sweep",
        commodity=target, seed=42,
        time=TimeConfig(dt=1.0, start_year=2025, end_year=2032),
        baseline=BASELINE_CFG,
        parameters=ParametersConfig(**kw),
        policy=PolicyConfig(),
        shocks=[ShockConfig(type="export_restriction",
                            start_year=start_year, end_year=end_year,
                            magnitude=magnitude, country="China")],
        outputs=OutputsConfig(metrics=["avg_price"]),
    )
    df, _ = run_scenario(cfg)
    sc = df.set_index("year")["P"]
    base = sc.loc[2025]
    window = {yr: sc.loc[yr] / base for yr in sc.index if yr >= start_year}
    peak_yr = max(window, key=window.get)
    return float(window[peak_yr]), int(peak_yr)


def main():
    print("=" * 88)
    print("BYPRODUCT DONOR SWEEP — Ge & Ga forward scenarios across 12 calibrated regimes")
    print("=" * 88)
    print("""
Target commodities (no CEPII calibration available — using donor regimes):
  germanium  (China ~80% production; byproduct of Zn smelting)
  gallium    (China ~85% production; byproduct of Al refining)

Methodology: thesis §6.3.5 zero-shot regime priors. Apply each calibrated
mineral's structural parameters {α_P, η_D, τ_K, g} to Ge/Ga forward
scenarios. Min/max across donors = defensible uncertainty band.

Scenarios:
  FULL_BAN   — do(restriction = 0.50) for 2026-2028 (Aug 2023 policy intent)
  SEVERE_BAN — do(restriction = 0.80) for 2026-2029 (escalation tail)
""")

    for target in TARGETS:
        print()
        print(f"{'=' * 88}")
        print(f"TARGET: {target.upper()}")
        print(f"{'=' * 88}")
        print(f"{'Donor regime':<22} {'Note':<28} "
              f"{'α_P':>5} {'τ_K':>5} {'η_D':>6} "
              f"{'FULL_pk':>7} {'FULL_yr':>7} {'SEV_pk':>6} {'SEV_yr':>6}")
        print("-" * 88)

        peaks_full = []
        peaks_sev = []
        for donor_name, donor_params, donor_note in DONORS:
            full_pk, full_yr = run_one(target, donor_params, 0.50, 2026, 2028)
            sev_pk, sev_yr = run_one(target, donor_params, 0.80, 2026, 2029)
            peaks_full.append((donor_name, full_pk))
            peaks_sev.append((donor_name, sev_pk))
            alpha = _stable_alpha_P(donor_params)
            print(f"{donor_name:<22} {donor_note:<28} "
                  f"{alpha:>5.2f} {donor_params['tau_K']:>5.2f} {donor_params['eta_D']:>6.3f} "
                  f"{full_pk:>7.2f} {full_yr:>7} {sev_pk:>6.2f} {sev_yr:>6}")
        print("-" * 88)

        # Bands
        full_vals = [p for _, p in peaks_full]
        sev_vals = [p for _, p in peaks_sev]
        print(f"\n  FULL_BAN  band: [{min(full_vals):.2f}×, {max(full_vals):.2f}×]   "
              f"median = {sorted(full_vals)[len(full_vals)//2]:.2f}×   "
              f"range = {max(full_vals) - min(full_vals):.2f}")
        print(f"  SEVERE_BAN band: [{min(sev_vals):.2f}×, {max(sev_vals):.2f}×]   "
              f"median = {sorted(sev_vals)[len(sev_vals)//2]:.2f}×   "
              f"range = {max(sev_vals) - min(sev_vals):.2f}")

        # Most/least defensible donors
        full_min_donor = min(peaks_full, key=lambda x: x[1])
        full_max_donor = max(peaks_full, key=lambda x: x[1])
        print(f"\n  Lowest peak (FULL_BAN): {full_min_donor[0]} ({full_min_donor[1]:.2f}×)")
        print(f"  Highest peak (FULL_BAN): {full_max_donor[0]} ({full_max_donor[1]:.2f}×)")

    print()
    print("=" * 88)
    print("INTERPRETATION")
    print("=" * 88)
    print("""
The peak-price BAND across donors is the defensible forward-projection
output for byproduct commodities without their own calibration. Reporting
a single donor's peak as a point estimate would understate uncertainty.

When CEPII Ge/Ga data lands and we calibrate via differential evolution,
the calibrated parameters should fall within the band — if they do, the
zero-shot transfer methodology is empirically validated for byproduct
minerals. If they fall outside, the regime-prior framework needs revision
for byproduct supply structures.

The donors most likely to be physically defensible analogs:
  - rare_earths_2010 (China-dominated restriction with inelastic
    semiconductor/defense demand — closest analog regime)
  - uranium_2007 (long τ_K byproduct-like cycle — closest analog supply
    response timeline for germanium)
  - graphite_2022 (China-dominated EV-era restriction — closest analog
    for gallium given semiconductor/5G demand)

Donors that should be treated as outliers (regime-mismatched):
  - graphite_2008 (pre-EV cycle, low α_P — does not represent China
    Aug 2023 restriction era)
  - rare_earths_2014 (post-WTO flood, opposite supply regime)
  - lithium_2016 (very high τ_K = 18yr from Chilean brine quota era,
    unrepresentative of byproduct supply response)

Recommended for thesis chapter / paper:
  Report the FULL_BAN band excluding the three outliers above.
  This gives a tighter 'defensible regime' band that is more useful for
  policy planning than the full 12-donor range.
""")


if __name__ == "__main__":
    main()
