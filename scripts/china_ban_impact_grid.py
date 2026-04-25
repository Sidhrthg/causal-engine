#!/usr/bin/env python3
"""
China Ban Impact Grid — L2 do-calculus projection across restriction magnitudes and start years.

Research question:
  "How long would the US remain affected across all trade routes and imports
   if China restricts X% of its exports in 2026 or 2027?"

Pearl Layer 2 (do-calculus):
  do(export_restriction = {0.25, 0.50, 0.75, 1.00}) starting 2026 or 2027.
  Restriction held for 3 years then removed.

The restriction magnitude applies to China's export share of each mineral.
Since China controls different fractions of global processing for each mineral,
a 25% China export ban has a larger absolute effect on graphite (China 95%
processing) than on lithium (China 70% processing).

Metric: "years affected" = norm_yr - start_year
  (total years from restriction onset until prices return within 10% of
   no-restriction baseline — the full period the US supply chain is disrupted)

This is distinct from the lag metric (norm_yr - end_year), which only counts
the scar period after the restriction has already ended.

All scenarios start from the 2024 calibrated state.
Background demand growth (g) continues at the calibrated episode rate.
Projection runs through 2035 to capture normalisation.
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
from src.minerals.predictability import (
    _GRAPHITE_2022_PARAMS,
    _RARE_EARTHS_2010_PARAMS,
    _COBALT_2016_PARAMS,
    _LITHIUM_2022_PARAMS,
    _NICKEL_2022_PARAMS,
    _URANIUM_2022_PARAMS,
)
from src.minerals.knowledge_graph import build_critical_minerals_kg

BASELINE_CFG   = BaselineConfig(P_ref=1.0, P0=1.0, K0=108.695652, I0=20.0, D0=100.0)
EULER_SAFETY   = 0.9
NORM_THRESHOLD = 0.10   # within 10% of no-restriction baseline = "normalised"
RESTRICTION_DURATION = 3  # years China restriction is held (then removed)
PROJECTION_END = 2035

_MINERAL_PARAMS = {
    "graphite":    _GRAPHITE_2022_PARAMS,
    "rare_earths": _RARE_EARTHS_2010_PARAMS,
    "cobalt":      _COBALT_2016_PARAMS,
    "lithium":     _LITHIUM_2022_PARAMS,
    "nickel":      _NICKEL_2022_PARAMS,
    "uranium":     _URANIUM_2022_PARAMS,
}

# China's effective control at the binding processing stage — queried live from KG.
# Uranium has no CEPII/USGS processing data in the KG; 0.20 is a Russia-SWU proxy.
_KG = build_critical_minerals_kg(data_dir="data/canonical")
_CHINA_PROCESSING_SHARE = {
    mineral: (_KG.effective_control_at("China", mineral, 2022)["effective_share"] or 0.20)
    for mineral in ("graphite", "rare_earths", "cobalt", "lithium", "nickel")
}
_CHINA_PROCESSING_SHARE["uranium"] = 0.20  # Russia SWU proxy — no KG data

_US_IMPORT_RELIANCE = {
    "graphite":    1.00,
    "rare_earths": 0.14,
    "cobalt":      0.76,
    "lithium":     0.50,
    "nickel":      0.40,
    "uranium":     0.95,
}

_EXTRA_PARAMS = {
    "graphite":    dict(substitution_elasticity=0.8, substitution_cap=0.6),
    "rare_earths": dict(substitution_elasticity=0.5, substitution_cap=0.4),
    "cobalt":      dict(substitution_elasticity=0.5, substitution_cap=0.4),
    "lithium":     dict(substitution_elasticity=0.6, substitution_cap=0.5,
                        fringe_capacity_share=0.4, fringe_entry_price=1.1),
    "nickel":      dict(substitution_elasticity=0.5, substitution_cap=0.4,
                        fringe_capacity_share=0.45, fringe_entry_price=1.15),
    "uranium":     {},
}


def _stable_alpha_P(params: dict) -> float:
    eta_D = params["eta_D"]
    limit = EULER_SAFETY / max(abs(eta_D), 1e-6)
    return min(params["alpha_P"], limit)


def _build_cfg(
    mineral: str,
    params: dict,
    extra: dict,
    magnitude: float,
    start_year: int,
    end_year: int,
    label: str,
) -> ScenarioConfig:
    alpha_P = _stable_alpha_P(params)
    kw = dict(
        eps=1e-9, u0=0.92, beta_u=0.10, u_min=0.70, u_max=1.00,
        tau_K=params["tau_K"], eta_K=0.40, retire_rate=0.0,
        eta_D=params["eta_D"],
        demand_growth=DemandGrowthConfig(type="constant", g=params["g"]),
        alpha_P=alpha_P,
        cover_star=0.20, lambda_cover=0.60, sigma_P=0.0,
    )
    kw.update(extra)
    shocks = []
    if magnitude > 0:
        shocks.append(ShockConfig(
            type="export_restriction",
            start_year=start_year,
            end_year=end_year,
            magnitude=magnitude,
        ))
    return ScenarioConfig(
        name=f"{mineral}_{label}",
        commodity="graphite",
        seed=42,
        time=TimeConfig(dt=1.0, start_year=2024, end_year=PROJECTION_END),
        baseline=BASELINE_CFG,
        parameters=ParametersConfig(**kw),
        policy=PolicyConfig(),
        shocks=shocks,
        outputs=OutputsConfig(metrics=["avg_price"]),
    )


def _norm_year(sc_df, bl_df, base_year, search_from):
    """First year >= search_from where scenario is within 10% of baseline."""
    sc = sc_df.set_index("year")["P"]
    bl = bl_df.set_index("year")["P"]
    sc_base = sc.loc[base_year]
    bl_base = bl.loc[base_year]
    for yr in sorted(y for y in sc.index if y >= search_from and y in bl.index):
        if abs(sc.loc[yr] / sc_base - bl.loc[yr] / bl_base) < NORM_THRESHOLD:
            return yr
    return None


def run_mineral_grid(mineral: str, params: dict, extra: dict) -> list[dict]:
    """Run the full magnitude × start_year grid for one mineral. Returns rows."""
    magnitudes  = [0.25, 0.50, 0.75, 1.00]
    start_years = [2026, 2027]
    base_year   = 2024

    # Run baseline once
    bl_cfg = _build_cfg(mineral, params, extra, 0.0, 2026, 2026, "baseline")
    bl_df, _ = run_scenario(bl_cfg)

    rows = []
    for start_yr in start_years:
        end_yr = start_yr + RESTRICTION_DURATION - 1  # 3-year restriction
        for mag in magnitudes:
            label = f"start{start_yr}_ban{int(mag*100)}"
            cfg   = _build_cfg(mineral, params, extra, mag, start_yr, end_yr, label)
            df, _ = run_scenario(cfg)

            sc      = df.set_index("year")["P"]
            sc_base = sc.loc[base_year]
            bl      = bl_df.set_index("year")["P"]
            bl_base = bl.loc[base_year]

            # Peak from restriction start onward
            window   = {yr: sc.loc[yr] / sc_base for yr in sc.index if yr >= start_yr}
            peak_yr  = max(window, key=window.get)
            peak_idx = window[peak_yr]

            # Normalisation: search from restriction end + 1
            norm_yr = _norm_year(df, bl_df, base_year, end_yr + 1)

            # Total years affected = from start to normalisation
            total_affected = (norm_yr - start_yr) if norm_yr else None

            # Effective global supply reduction = magnitude × China processing share
            china_share = _CHINA_PROCESSING_SHARE[mineral]
            global_supply_reduction = mag * china_share

            rows.append({
                "mineral":                 mineral,
                "start_yr":                start_yr,
                "end_yr":                  end_yr,
                "magnitude":               mag,
                "china_processing_share":  china_share,
                "global_supply_reduction": global_supply_reduction,
                "us_import_reliance":      _US_IMPORT_RELIANCE[mineral],
                "peak_idx":                peak_idx,
                "peak_yr":                 peak_yr,
                "norm_yr":                 norm_yr,
                "lag_from_end":            (norm_yr - end_yr) if norm_yr else None,
                "total_affected_yrs":      total_affected,
                "tau_K":                   params["tau_K"],
                "alpha_P":                 _stable_alpha_P(params),
                "eta_D":                   params["eta_D"],
            })

    return rows


def print_mineral_table(mineral: str, rows: list[dict]) -> None:
    china_share    = _CHINA_PROCESSING_SHARE[mineral]
    us_reliance    = _US_IMPORT_RELIANCE[mineral]
    params         = _MINERAL_PARAMS[mineral]
    tau_K          = params["tau_K"]

    print(f"\n{'═' * 78}")
    print(f"MINERAL: {mineral.upper()}")
    print(f"  China processing share: {china_share:.0%}   US import reliance: {us_reliance:.0%}")
    print(f"  τ_K = {tau_K:.2f}yr   α_P = {_stable_alpha_P(params):.3f}   η_D = {params['eta_D']:.3f}")
    print(f"  Restriction duration: {RESTRICTION_DURATION} years (then removed)")
    print(f"  'Years affected' = years from restriction START until prices normalise")
    print()

    for start_yr in [2026, 2027]:
        end_yr = start_yr + RESTRICTION_DURATION - 1
        mineral_rows = [r for r in rows if r["start_yr"] == start_yr]

        print(f"  Restriction starts {start_yr} (ends {end_yr})")
        print(f"  {'Ban %':>6}  {'Global supply':>14}  {'Peak ×':>7}  "
              f"{'Peak yr':>8}  {'Norm yr':>8}  {'Lag from end':>13}  {'Total yrs affected':>19}")
        print("  " + "─" * 6 + "  " + "─" * 14 + "  " + "─" * 7 + "  " +
              "─" * 8 + "  " + "─" * 8 + "  " + "─" * 13 + "  " + "─" * 19)

        for r in mineral_rows:
            ban_pct    = f"{int(r['magnitude']*100)}%"
            gsr        = f"{r['global_supply_reduction']:.0%} reduction"
            peak       = f"{r['peak_idx']:.2f}×"
            peak_y     = str(r["peak_yr"])
            norm_y     = str(r["norm_yr"]) if r["norm_yr"] else "never"
            lag_str    = f"+{r['lag_from_end']}yr" if r["lag_from_end"] is not None else "never"
            total_str  = (f"{r['total_affected_yrs']} yrs" if r["total_affected_yrs"] is not None
                          else f">{PROJECTION_END - start_yr} yrs")
            print(f"  {ban_pct:>6}  {gsr:>14}  {peak:>7}  "
                  f"{peak_y:>8}  {norm_y:>8}  {lag_str:>13}  {total_str:>19}")
        print()


def print_cross_mineral_summary(all_rows: list[dict]) -> None:
    print("\n" + "=" * 78)
    print("CROSS-MINERAL SUMMARY — 50% BAN STARTING 2026 (3yr duration)")
    print("How long does the US remain affected?")
    print("=" * 78)
    print(f"  {'Mineral':<14} {'τ_K':>6} {'US rely':>8} {'Peak ×':>7} "
          f"{'Peak yr':>8} {'Norm yr':>8} {'Yrs affected':>13}")
    print("  " + "─" * 78)

    for mineral in _MINERAL_PARAMS:
        mineral_rows = [r for r in all_rows
                        if r["mineral"] == mineral
                        and r["start_yr"] == 2026
                        and r["magnitude"] == 0.50]
        if not mineral_rows:
            continue
        r     = mineral_rows[0]
        norm  = str(r["norm_yr"]) if r["norm_yr"] else "never"
        total = (f"{r['total_affected_yrs']} yrs" if r["total_affected_yrs"] is not None
                 else f">{PROJECTION_END - 2026} yrs")
        print(f"  {mineral:<14} {r['tau_K']:>6.2f} {r['us_import_reliance']:>8.0%} "
              f"{r['peak_idx']:>7.2f}× {r['peak_yr']:>8} {norm:>8} {total:>13}")

    print()
    print("  Interpretation:")
    print("  'Years affected' includes the restriction period itself + the price scar after.")
    print("  A country exiting the restriction does NOT immediately end the market disruption.")
    print("  τ_K governs how fast the price scar decays; US reliance governs exposure severity.")

    print("\n" + "=" * 78)
    print("CROSS-MINERAL SUMMARY — FULL BAN (100%) STARTING 2026 (3yr duration)")
    print("=" * 78)
    print(f"  {'Mineral':<14} {'China share':>12} {'Peak ×':>7} "
          f"{'Norm yr':>8} {'Yrs affected':>13}")
    print("  " + "─" * 60)

    for mineral in _MINERAL_PARAMS:
        mineral_rows = [r for r in all_rows
                        if r["mineral"] == mineral
                        and r["start_yr"] == 2026
                        and r["magnitude"] == 1.00]
        if not mineral_rows:
            continue
        r     = mineral_rows[0]
        norm  = str(r["norm_yr"]) if r["norm_yr"] else "never"
        total = (f"{r['total_affected_yrs']} yrs" if r["total_affected_yrs"] is not None
                 else f">{PROJECTION_END - 2026} yrs")
        print(f"  {mineral:<14} {r['china_processing_share']:>12.0%} "
              f"{r['peak_idx']:>7.2f}× {norm:>8} {total:>13}")


def print_start_year_comparison(all_rows: list[dict]) -> None:
    """For 50% ban: how much does delaying restriction by 1 year (2026 vs 2027) change outcomes?"""
    print("\n" + "=" * 78)
    print("2026 vs 2027 START YEAR COMPARISON — 50% BAN, 3yr duration")
    print("Does a 1-year delay in imposing the restriction change US exposure?")
    print("=" * 78)
    print(f"  {'Mineral':<14} {'2026 yrs affected':>19} {'2027 yrs affected':>19} {'Difference':>12}")
    print("  " + "─" * 70)

    for mineral in _MINERAL_PARAMS:
        r26 = next((r for r in all_rows if r["mineral"] == mineral
                    and r["start_yr"] == 2026 and r["magnitude"] == 0.50), None)
        r27 = next((r for r in all_rows if r["mineral"] == mineral
                    and r["start_yr"] == 2027 and r["magnitude"] == 0.50), None)
        if not r26 or not r27:
            continue

        t26 = r26["total_affected_yrs"]
        t27 = r27["total_affected_yrs"]
        s26 = (f"{t26} yrs" if t26 is not None else f">{PROJECTION_END - 2026} yrs")
        s27 = (f"{t27} yrs" if t27 is not None else f">{PROJECTION_END - 2027} yrs")

        if t26 is not None and t27 is not None:
            diff_str = f"{t27 - t26:+d} yrs"
        else:
            diff_str = "—"
        print(f"  {mineral:<14} {s26:>19} {s27:>19} {diff_str:>12}")

    print()
    print("  Positive difference = later start extends total affected period.")
    print("  Negative difference = later start reduces affected period (rare — baseline growth).")


def main():
    print("=" * 78)
    print("CHINA BAN IMPACT GRID — L2 DO-CALCULUS PROJECTION")
    print("Research question: How long does the US remain affected if China bans")
    print("X% of its exports in 2026 or 2027, across all trade routes and imports?")
    print("=" * 78)
    print(f"""
Pearl Layer 2 (do-calculus):
  do(export_restriction = {{0.25, 0.50, 0.75, 1.00}}) starting 2026 or 2027.
  Restriction held for {RESTRICTION_DURATION} years, then removed.

Metric: 'years affected' = norm_yr - start_yr
  = total years from restriction onset until prices return to within
    10% of the no-restriction baseline trajectory.
  This captures both the restriction period AND the post-restriction price scar.

Effective global supply reduction = ban_magnitude × China_processing_share.
  Example: 50% China graphite ban × 95% China processing share
           = 47.5% global anode supply reduction.

All minerals use their most recent calibrated episode parameters.
Projection window: 2024–{PROJECTION_END}.
""")

    all_rows = []
    for mineral, params in _MINERAL_PARAMS.items():
        extra = _EXTRA_PARAMS.get(mineral, {})
        rows  = run_mineral_grid(mineral, params, extra)
        all_rows.extend(rows)
        print_mineral_table(mineral, rows)

    print_cross_mineral_summary(all_rows)
    print_start_year_comparison(all_rows)

    print("\n" + "=" * 78)
    print("KEY POLICY FINDING")
    print("=" * 78)
    print("""
  The 'years affected' metric is the operationally relevant horizon for
  strategic reserve and supply diversification planning.

  A strategic reserve must cover the FULL affected period — not just the
  restriction duration — because market normalisation lags restriction end
  by 1–5 years depending on τ_K.

  Minimum reserve = total_affected_years × annual_consumption
  (at maintained drawdown rate throughout the affected period)

  The 2026 vs 2027 comparison shows whether the US has a meaningful window
  to pre-position reserves before a plausible restriction horizon.
""")


if __name__ == "__main__":
    main()
