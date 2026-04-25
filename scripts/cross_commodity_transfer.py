#!/usr/bin/env python3
"""
Cross-commodity parameter transfer test.

Tests whether structural parameters from one commodity's episode can predict
price direction for a different commodity facing the same demand driver.

Hypothesis: commodities sharing a demand driver (EV adoption) share structural
parameters across episodes, even across different minerals. If cobalt_2016 EV
speculation parameters transfer to graphite_2022 better than graphite_2008 does,
that identifies a cross-mineral behavioral pattern exploitable for early warning
and stockpile strategy.

Matrix tested:
  Target episode: graphite_2022 (EV surge + export controls)
  Donor params:
    - graphite_2008   (same mineral, different regime)       ← current OOS result
    - cobalt_2016     (different mineral, same EV driver)
    - cobalt_2022     (different mineral, post-EV)
    - nickel_2022     (different mineral, EV adjacent)
    - lithium_2022    (different mineral, same EV driver)

  Target episode: lithium_2022 (EV demand boom)
  Donor params:
    - cobalt_2016     (same EV driver, earlier)
    - graphite_2022   (same EV driver, same period)
    - nickel_2022     (EV adjacent)
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
    _LITHIUM_2022_PARAMS, _SOYBEANS_2022_PARAMS,
    _COBALT_2016_PARAMS, _COBALT_2022_PARAMS,
    _NICKEL_2006_PARAMS, _NICKEL_2022_PARAMS,
    _cepii_series, _directional_accuracy, _spearman_rho,
)


def _std(p, extra=None):
    d = dict(
        eps=1e-9, u0=0.92, beta_u=0.10, u_min=0.70, u_max=1.00,
        eta_K=0.40, retire_rate=0.0,
        cover_star=0.20, lambda_cover=0.60, sigma_P=0.0,
        tau_K=p["tau_K"], eta_D=p["eta_D"],
        demand_growth=DemandGrowthConfig(type="constant", g=p["g"]),
        alpha_P=p["alpha_P"],
    )
    if extra:
        d.update(extra)
    return ParametersConfig(**d)


def _run(name, params, time_cfg, baseline, shocks, cepii, years, base_year):
    cfg = ScenarioConfig(
        name=name, commodity="graphite", seed=42,
        time=time_cfg, baseline=baseline,
        parameters=params,
        policy=PolicyConfig(), shocks=shocks,
        outputs=OutputsConfig(metrics=["avg_price"]),
    )
    df, _ = run_scenario(cfg)
    m = df.set_index("year")
    model_idx = m.loc[years, "P"] / m.loc[base_year, "P"]
    data_idx = cepii.loc[years, "implied_price"] / cepii.loc[base_year, "implied_price"]
    da  = _directional_accuracy(model_idx, data_idx)
    rho = _spearman_rho(model_idx, data_idx)
    return da, rho


def main():
    cg = _cepii_series("data/canonical/cepii_graphite.csv", "China")
    cl = _cepii_series("data/canonical/cepii_lithium.csv",  "Chile")

    g22_time     = TimeConfig(dt=1.0, start_year=2019, end_year=2025)
    g22_baseline = BaselineConfig(P_ref=1.0, P0=1.0, K0=108.695652, I0=20.0, D0=100.0)
    g22_shocks   = [
        ShockConfig(type="demand_surge",       start_year=2022, end_year=2022, magnitude=0.10),
        ShockConfig(type="export_restriction", start_year=2023, end_year=2024, magnitude=0.35),
        ShockConfig(type="demand_surge",       start_year=2023, end_year=2023, magnitude=-0.30),
        ShockConfig(type="demand_surge",       start_year=2024, end_year=2024, magnitude=-0.05),
        ShockConfig(type="stockpile_release",  start_year=2023, end_year=2023, magnitude=20.0),
    ]
    g22_years, g22_base = [2021, 2022, 2023, 2024], 2021

    li22_time     = TimeConfig(dt=1.0, start_year=2019, end_year=2025)
    li22_baseline = BaselineConfig(P_ref=1.0, P0=1.0, K0=108.695652, I0=20.0, D0=100.0)
    li22_shocks   = [ShockConfig(type="demand_surge", start_year=2022, end_year=2022, magnitude=0.30)]
    li22_years, li22_base = [2021, 2022, 2023, 2024], 2021

    # ── Graphite 2022 target ──────────────────────────────────────────────────
    g22_donors = [
        ("graphite_2008 (same mineral, pre-EV)",      _GRAPHITE_2008_PARAMS, dict(substitution_elasticity=0.8, substitution_cap=0.6)),
        ("cobalt_2016   (EV speculation cycle)",       _COBALT_2016_PARAMS,   dict(substitution_elasticity=0.8, substitution_cap=0.6)),
        ("cobalt_2022   (post-EV, DRC oversupply)",    _COBALT_2022_PARAMS,   dict(substitution_elasticity=0.8, substitution_cap=0.6)),
        ("nickel_2022   (HPAL ramp, EV adjacent)",     _NICKEL_2022_PARAMS,   dict(substitution_elasticity=0.8, substitution_cap=0.6)),
        ("lithium_2022  (same EV driver)",             _LITHIUM_2022_PARAMS,  dict(substitution_elasticity=0.8, substitution_cap=0.6)),
        ("graphite_2022 (in-sample upper bound)",      _GRAPHITE_2022_PARAMS, dict(substitution_elasticity=0.8, substitution_cap=0.6)),
    ]

    print("=" * 72)
    print("TARGET: graphite_2022 (EV surge + export controls)")
    print("Shocks fixed; only structural params {α_P, η_D, τ_K, g} vary")
    print("=" * 72)
    print(f"{'Donor params':<42} {'α_P':>6} {'η_D':>7} {'τ_K':>6} {'DA':>6} {'ρ':>6}")
    print("-" * 72)

    g22_results = []
    for label, p, extra in g22_donors:
        da, rho = _run(
            f"g22_{label[:8]}", _std(p, extra),
            g22_time, g22_baseline, g22_shocks, cg, g22_years, g22_base
        )
        print(f"{label:<42} {p['alpha_P']:>6.3f} {p['eta_D']:>7.3f} {p['tau_K']:>6.3f} {da:>6.3f} {rho:>6.3f}")
        g22_results.append((label, p, da, rho))

    print()

    # ── Lithium 2022 target ───────────────────────────────────────────────────
    li22_donors = [
        ("cobalt_2016   (EV speculation, earlier)",  _COBALT_2016_PARAMS,  dict(fringe_capacity_share=0.4, fringe_entry_price=1.1)),
        ("cobalt_2022   (post-EV correction)",        _COBALT_2022_PARAMS,  dict(fringe_capacity_share=0.4, fringe_entry_price=1.1)),
        ("graphite_2022 (same EV driver, same yr)",   _GRAPHITE_2022_PARAMS,dict(fringe_capacity_share=0.4, fringe_entry_price=1.1)),
        ("nickel_2022   (HPAL ramp)",                 _NICKEL_2022_PARAMS,  dict(fringe_capacity_share=0.4, fringe_entry_price=1.1)),
        ("lithium_2022  (in-sample upper bound)",     _LITHIUM_2022_PARAMS, dict(fringe_capacity_share=0.4, fringe_entry_price=1.1)),
    ]

    print("=" * 72)
    print("TARGET: lithium_2022 (EV demand boom + fringe supply entry)")
    print("=" * 72)
    print(f"{'Donor params':<42} {'α_P':>6} {'η_D':>7} {'τ_K':>6} {'DA':>6} {'ρ':>6}")
    print("-" * 72)

    li22_results = []
    for label, p, extra in li22_donors:
        da, rho = _run(
            f"li22_{label[:8]}", _std(p, extra),
            li22_time, li22_baseline, li22_shocks, cl, li22_years, li22_base
        )
        print(f"{label:<42} {p['alpha_P']:>6.3f} {p['eta_D']:>7.3f} {p['tau_K']:>6.3f} {da:>6.3f} {rho:>6.3f}")
        li22_results.append((label, p, da, rho))

    print()
    print("=" * 72)
    print("INTERPRETATION")
    print("=" * 72)

    # Best cross-mineral donor for graphite_2022
    cross_g22 = [(l, p, da, rho) for l, p, da, rho in g22_results if "in-sample" not in l and "graphite_2008" not in l]
    best_cross = max(cross_g22, key=lambda x: x[2])
    same_period = next(r for r in g22_results if "graphite_2008" in r[0])

    print(f"\nGraphite 2022 OOS:")
    print(f"  Same mineral (2008 params):   DA = {same_period[2]:.3f}")
    print(f"  Best cross-mineral ({best_cross[0][:20]}): DA = {best_cross[2]:.3f}")
    if best_cross[2] > same_period[2]:
        print(f"  → Cross-mineral transfer BETTER by {(best_cross[2]-same_period[2])*100:.1f}pp")
        print(f"    Interpretation: {best_cross[0].split('(')[1].rstrip(')')} shares")
        print(f"    structural parameters with graphite_2022 better than graphite_2008 does.")
        print(f"    This is the behavioral transfer signal: EV-cycle dynamics are")
        print(f"    mineral-agnostic; the 2008 commodity cycle is not.")
    else:
        print(f"  → Same-mineral transfer wins by {(same_period[2]-best_cross[2])*100:.1f}pp")
        print(f"    Interpretation: graphite structural parameters are mineral-specific")
        print(f"    even within the same demand driver.")

    return g22_results, li22_results


if __name__ == "__main__":
    main()
