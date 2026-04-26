#!/usr/bin/env python3
"""
Null distribution for L3 counterfactual causal effects.

Runs Pearl L3 Abduction-Action-Prediction on all 8 in-sample episodes,
removing each episode's primary shock, and reports peak causal effect.

Purpose: show that graphite_2023 (+111.5pp) is in the tail of the distribution
of causal effects across commodity-episode pairs.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.minerals.schema import (
    BaselineConfig, DemandGrowthConfig, OutputsConfig,
    ParametersConfig, PolicyConfig, ScenarioConfig,
    ShockConfig, TimeConfig, load_scenario,
)
from src.minerals.simulate import run_scenario
from src.minerals.constants import ODE_DEFAULTS
from src.minerals.predictability import (
    _GRAPHITE_2008_PARAMS, _GRAPHITE_2022_PARAMS,
    _LITHIUM_2022_PARAMS, _SOYBEANS_2022_PARAMS,
    _cepii_series, _l3_abduct_predict,
)


def _world_soy_series() -> pd.DataFrame:
    df = pd.read_csv("data/canonical/cepii_soybeans.csv")
    world = (
        df.groupby("year")
        .agg(value_kusd=("value_kusd", "sum"), qty_tonnes=("quantity_tonnes", "sum"))
        .reset_index()
    )
    world["implied_price"] = world["value_kusd"] / world["qty_tonnes"]
    return world.set_index("year")


def _run_l3(cfg_actual, cfg_cf, cepii_series, years, base_year, episode_name, dominant_share):
    m_act = run_scenario(cfg_actual)[0].set_index("year")
    m_cf  = run_scenario(cfg_cf)[0].set_index("year")

    available = [y for y in years if y in cepii_series.index and y in m_act.index and y in m_cf.index]
    if len(available) < 2:
        print(f"  {episode_name}: insufficient data ({available})")
        return None

    act_model = {y: float(m_act.loc[y, "P"] / m_act.loc[base_year, "P"]) for y in available}
    cf_model  = {y: float(m_cf.loc[y,  "P"] / m_cf.loc[base_year,  "P"]) for y in available}
    data      = {y: float(cepii_series.loc[y, "implied_price"] / cepii_series.loc[base_year, "implied_price"])
                 for y in available}

    l3_cf, U = _l3_abduct_predict(act_model, cf_model, data, available)
    effects   = {y: (data[y] - l3_cf[y]) * 100 for y in available}
    peak      = max(abs(v) for v in effects.values())

    return {
        "episode": episode_name,
        "dominant_share_pct": dominant_share,
        "years_evaluated": available,
        "peak_causal_effect_pp": round(peak, 1),
        "effects_pp": {y: round(v, 1) for y, v in effects.items()},
        "residuals_U": {y: round(v, 3) for y, v in U.items()},
    }


def _std_params(**kwargs):
    return {**ODE_DEFAULTS, **kwargs}


def _std_baseline():
    return BaselineConfig(P_ref=1.0, P0=1.0, K0=108.695652, I0=20.0, D0=100.0)


def main():
    results = []

    # ── 1. Graphite 2023: do(export_restriction=0) ────────────────────────────
    # China had ~90% of world natural graphite supply in 2022.
    p = _GRAPHITE_2022_PARAMS
    actual_shocks = [
        ShockConfig(type="demand_surge",       start_year=2022, end_year=2022, magnitude=0.10),
        ShockConfig(type="export_restriction", start_year=2023, end_year=2024, magnitude=0.35),
        ShockConfig(type="demand_surge",       start_year=2023, end_year=2023, magnitude=-0.30),
        ShockConfig(type="demand_surge",       start_year=2024, end_year=2024, magnitude=-0.05),
        ShockConfig(type="stockpile_release",  start_year=2023, end_year=2023, magnitude=20.0),
    ]
    cf_shocks = [s for s in actual_shocks if s.type != "export_restriction"]

    def _g22(shocks):
        return ScenarioConfig(
            name="g22_cf", commodity="graphite", seed=42,
            time=TimeConfig(dt=1.0, start_year=2019, end_year=2025),
            baseline=_std_baseline(),
            parameters=ParametersConfig(
                **_std_params(
                    tau_K=p["tau_K"], eta_D=p["eta_D"],
                    demand_growth=DemandGrowthConfig(type="constant", g=p["g"]),
                    alpha_P=p["alpha_P"],
                    substitution_elasticity=0.8, substitution_cap=0.6,
                ),
            ),
            policy=PolicyConfig(), shocks=shocks,
            outputs=OutputsConfig(metrics=["avg_price"]),
        )

    cg = _cepii_series("data/canonical/cepii_graphite.csv", "China")
    r = _run_l3(_g22(actual_shocks), _g22(cf_shocks), cg,
                [2021, 2022, 2023, 2024], 2021,
                "graphite_2023_export_controls", dominant_share=90)
    if r:
        results.append(r)

    # ── 2. Graphite 2008: do(quota + capex_shock=0) ───────────────────────────
    p08 = _GRAPHITE_2008_PARAMS
    actual_shocks_08 = [
        ShockConfig(type="demand_surge",       start_year=2008, end_year=2008, magnitude=0.46),
        ShockConfig(type="macro_demand_shock", start_year=2009, end_year=2009, magnitude=-0.40, demand_destruction=-0.40),
        ShockConfig(type="policy_shock",       start_year=2010, end_year=2011, magnitude=0.35, quota_reduction=0.35),
        ShockConfig(type="capex_shock",        start_year=2010, end_year=2011, magnitude=0.50),
    ]
    cf_shocks_08 = [s for s in actual_shocks_08 if s.type not in ("policy_shock", "capex_shock")]

    def _g08(shocks):
        return ScenarioConfig(
            name="g08_cf", commodity="graphite", seed=123,
            time=TimeConfig(dt=1.0, start_year=2004, end_year=2011),
            baseline=_std_baseline(),
            parameters=ParametersConfig(
                **_std_params(
                    tau_K=p08["tau_K"], eta_D=p08["eta_D"],
                    demand_growth=DemandGrowthConfig(type="constant", g=p08["g"]),
                    alpha_P=p08["alpha_P"],
                ),
            ),
            policy=PolicyConfig(), shocks=shocks,
            outputs=OutputsConfig(metrics=["avg_price"]),
        )

    r = _run_l3(_g08(actual_shocks_08), _g08(cf_shocks_08), cg,
                [2006, 2007, 2008, 2009, 2010, 2011], 2006,
                "graphite_2008_export_quota", dominant_share=80)
    if r:
        results.append(r)

    # ── 3. Lithium 2022: do(demand_surge=0) ──────────────────────────────────
    # Remove EV boom demand surge; fringe supply stays (it's a structural param, not a shock).
    # Chile had ~30% world lithium supply (Australia ~55%). Using Chile CEPII series.
    # Dominant share ≈ 55% (Australia), but both supplied the boom.
    p_li = _LITHIUM_2022_PARAMS
    def _li_cfg(shocks):
        return ScenarioConfig(
            name="li22_cf", commodity="lithium", seed=42,
            time=TimeConfig(dt=1.0, start_year=2019, end_year=2025),
            baseline=_std_baseline(),
            parameters=ParametersConfig(
                **_std_params(
                    tau_K=p_li["tau_K"], eta_D=p_li["eta_D"],
                    demand_growth=DemandGrowthConfig(type="constant", g=p_li["g"]),
                    alpha_P=p_li["alpha_P"],
                    fringe_capacity_share=0.4, fringe_entry_price=1.1,
                ),
            ),
            policy=PolicyConfig(), shocks=shocks,
            outputs=OutputsConfig(metrics=["avg_price"]),
        )

    li_actual = [ShockConfig(type="demand_surge", start_year=2022, end_year=2022, magnitude=0.30)]
    li_cf     = []  # no demand surge counterfactual

    cl = _cepii_series("data/canonical/cepii_lithium.csv", "Chile")
    r = _run_l3(_li_cfg(li_actual), _li_cfg(li_cf), cl,
                [2021, 2022, 2023, 2024], 2021,
                "lithium_2022_ev_demand_boom", dominant_share=55)
    if r:
        results.append(r)

    # ── 4–8. Soybeans episodes ────────────────────────────────────────────────
    # USA had ~35% of world soybean exports; Brazil ~45%.
    cs = _world_soy_series()

    # 4. Soybeans 2011: remove demand_surge 2010-2011 (food price spike)
    cfg_2011 = load_scenario("scenarios/soybeans_2011_food_crisis.yaml")
    cfg_2011_cf = load_scenario("scenarios/soybeans_2011_food_crisis.yaml")
    cfg_2011_cf.shocks = [s for s in cfg_2011_cf.shocks if s.type != "demand_surge"]
    r = _run_l3(cfg_2011, cfg_2011_cf, cs,
                [2009, 2010, 2011], 2009,
                "soybeans_2011_food_price_spike", dominant_share=35)
    if r:
        results.append(r)

    # 5. Soybeans 2015: remove demand_surge (supply glut, negative shock)
    cfg_2015 = load_scenario("scenarios/soybeans_2015_supply_glut.yaml")
    cfg_2015_cf = load_scenario("scenarios/soybeans_2015_supply_glut.yaml")
    cfg_2015_cf.shocks = []  # all shocks are demand_surge type
    r = _run_l3(cfg_2015, cfg_2015_cf, cs,
                [2014, 2015, 2016, 2017], 2014,
                "soybeans_2015_supply_glut", dominant_share=35)
    if r:
        results.append(r)

    # 6. Soybeans 2018: remove export_restriction
    cfg_2018 = load_scenario("scenarios/soybeans_2018_trade_war.yaml")
    cfg_2018_cf = load_scenario("scenarios/soybeans_2018_trade_war.yaml")
    cfg_2018_cf.shocks = [s for s in cfg_2018_cf.shocks if s.type != "export_restriction"]
    years_2018 = [y for y in [2016, 2017, 2018, 2020, 2021] if y in cs.index]
    r = _run_l3(cfg_2018, cfg_2018_cf, cs,
                years_2018, years_2018[0],
                "soybeans_2018_trade_war_tariff", dominant_share=35)
    if r:
        results.append(r)

    # 7. Soybeans 2020: remove demand_surge 2020 (Phase-1 deal recovery)
    cfg_2020 = load_scenario("scenarios/soybeans_2020_phase1.yaml")
    cfg_2020_cf = load_scenario("scenarios/soybeans_2020_phase1.yaml")
    # Keep export_restriction (already happened 2017); remove the demand recovery shocks
    cfg_2020_cf.shocks = [s for s in cfg_2020_cf.shocks if s.type == "export_restriction"]
    years_2020 = [y for y in [2018, 2020, 2021] if y in cs.index]
    r = _run_l3(cfg_2020, cfg_2020_cf, cs,
                years_2020, years_2020[0],
                "soybeans_2020_phase1_la_nina", dominant_share=35)
    if r:
        results.append(r)

    # 8. Soybeans 2022: remove Ukraine-specific shocks (demand_surge 2022 + capex 2022)
    p_s22 = _SOYBEANS_2022_PARAMS
    def _s22(shocks):
        return ScenarioConfig(
            name="s22_cf", commodity="soybeans", seed=42,
            time=TimeConfig(dt=1.0, start_year=2018, end_year=2024),
            baseline=_std_baseline(),
            parameters=ParametersConfig(
                **_std_params(
                    tau_K=p_s22["tau_K"], eta_D=p_s22["eta_D"],
                    demand_growth=DemandGrowthConfig(type="constant", g=p_s22["g"]),
                    alpha_P=p_s22["alpha_P"],
                ),
            ),
            policy=PolicyConfig(), shocks=shocks,
            outputs=OutputsConfig(metrics=["avg_price"]),
        )

    s22_actual = [
        ShockConfig(type="demand_surge", start_year=2021, end_year=2021, magnitude=0.20),
        ShockConfig(type="demand_surge", start_year=2022, end_year=2022, magnitude=0.10),
        ShockConfig(type="capex_shock",  start_year=2022, end_year=2022, magnitude=0.15),
    ]
    # Counterfactual: La Niña 2021 demand surge stays; remove Ukraine-specific 2022 shocks
    s22_cf = [s for s in s22_actual if s.start_year != 2022]

    years_s22 = [y for y in [2020, 2021, 2022, 2023, 2024] if y in cs.index]
    r = _run_l3(_s22(s22_actual), _s22(s22_cf), cs,
                years_s22, years_s22[0],
                "soybeans_2022_ukraine_shock", dominant_share=35)
    if r:
        results.append(r)

    # ── Print results ─────────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("L3 COUNTERFACTUAL NULL DISTRIBUTION")
    print("Peak causal effect of primary intervention, ordered by magnitude")
    print("=" * 72)
    print(f"{'Episode':<42} {'Dom.%':>6} {'Peak pp':>8} {'Years'}")
    print("-" * 72)

    sorted_r = sorted(results, key=lambda x: -x["peak_causal_effect_pp"])
    for r in sorted_r:
        dom  = f"{r['dominant_share_pct']}%"
        peak = f"{r['peak_causal_effect_pp']:+.1f}"
        yrs  = f"{r['years_evaluated'][0]}–{r['years_evaluated'][-1]}"
        print(f"{r['episode']:<42} {dom:>6} {peak:>8} {yrs}")
        if r.get("effects_pp"):
            for y, v in r["effects_pp"].items():
                print(f"    {y}: {v:+.1f} pp")
        print()

    print("=" * 72)
    effects = [r["peak_causal_effect_pp"] for r in sorted_r]
    gfx2023 = next(r["peak_causal_effect_pp"] for r in sorted_r if "2023" in r["episode"])
    rank = sum(1 for e in effects if e >= gfx2023)
    print(f"\nGraphite 2023 peak effect ({gfx2023:.1f} pp) ranks {rank}/{len(effects)}")
    median = sorted(effects)[len(effects) // 2]
    print(f"Median effect across all episodes: {median:.1f} pp")
    print(f"Ratio graphite_2023 / median: {gfx2023/max(median,0.1):.1f}×")

    return results


if __name__ == "__main__":
    main()
