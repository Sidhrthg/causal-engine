"""
Cross-commodity parameter transfer evaluation.

Tests whether structural parameters calibrated on one commodity can predict
price trajectories for a *different* commodity, given only the target's
observable shock inputs (export restrictions, demand surges, etc.).

This is distinct from within-commodity OOS (predictability.py), where the
same commodity is tested across different time periods.

Framing
-------
The ODE has four structural parameters:
  alpha_P  — price adjustment speed (market concentration × demand inelasticity)
  eta_D    — demand price elasticity (substitutability)
  tau_K    — capacity adjustment time in years (geological / investment cycle)
  g        — demand growth rate (period-specific, NOT transferred)

The hypothesis: minerals that share supply structure and demand characteristics
will have similar alpha_P, eta_D, tau_K even if g differs. We therefore
transfer only (alpha_P, eta_D, tau_K) and keep the target's own g.

Transfer pairs (donor → target)
---------------------------------
  graphite_2008  → rare_earths_2010   China export quota, inelastic industrial demand
  graphite_2022  → lithium_2022       EV-era regime, China-concentrated supply
  cobalt_2016    → lithium_2016       Early EV wave, battery metal, concentrated supply
  nickel_2006    → cobalt_2016        Metal supply squeeze, stainless/EV demand

Zero-shot section
-----------------
Given only regime classification (no calibration data), what parameters should
we assume? The alpha_P regime signal from Chapter 5 provides the prior:
  EV-driven + geopolitically concentrated → alpha_P ≥ 1.5
  Pre-EV or distributed supply            → alpha_P ~ 0.5

Usage
-----
    python -m src.minerals.cross_commodity_transfer
    python -m src.minerals.cross_commodity_transfer --verbose
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .schema import (
    BaselineConfig, DemandGrowthConfig, OutputsConfig,
    ParametersConfig, PolicyConfig, ScenarioConfig,
    ShockConfig, TimeConfig,
)
from .simulate import run_scenario
from .predictability import (
    _directional_accuracy, _spearman_rho, _log_price_rmse, _magnitude_ratio,
    _cepii_series, _wb_cobalt_series, _wb_nickel_series,
    EpisodeResult,
    _GRAPHITE_2008_PARAMS, _GRAPHITE_2022_PARAMS,
    _LITHIUM_2016_PARAMS, _LITHIUM_2022_PARAMS,
    _RARE_EARTHS_2010_PARAMS,
    _COBALT_2016_PARAMS, _COBALT_2022_PARAMS,
    _NICKEL_2006_PARAMS, _NICKEL_2022_PARAMS,
)


# ── Regime priors (zero-shot parameter sets) ──────────────────────────────────
#
# Derived from the alpha_P regime signal (Chapter 5):
# All EV/restriction episodes cluster at alpha_P >= 1.5.
# tau_K and eta_D are median values across calibrated episodes in each regime.

REGIME_PRIORS: Dict[str, Dict[str, float]] = {
    "ev_restricted": dict(
        # EV-driven demand + geopolitically concentrated supply (China/DRC dominant)
        # Prototype: graphite_2022, lithium_2022, cobalt_2016, rare_earths_2010
        alpha_P=2.0,   # median of ev-regime calibrated episodes
        eta_D=-0.5,    # moderate inelasticity
        tau_K=6.0,     # ~6yr investment cycle typical for battery metals
        # g is always episode-specific — set from demand projections
    ),
    "pre_ev": dict(
        # Pre-EV or distributed supply (graphite_2008, soybeans)
        alpha_P=0.5,
        eta_D=-0.1,
        tau_K=8.0,
    ),
    "long_cycle": dict(
        # Long-cycle mining (uranium, deep-sea, rare earths hard rock)
        alpha_P=1.8,
        eta_D=-0.3,
        tau_K=14.0,
    ),
}


# ── Transfer pair definitions ─────────────────────────────────────────────────

@dataclass
class TransferPair:
    donor: str
    target: str
    rationale: str
    donor_params: Dict[str, float]   # alpha_P, eta_D, tau_K, g (full set)
    target_g: float                  # target's own demand growth (not transferred)

    @property
    def transferred_params(self) -> Dict[str, float]:
        """alpha_P, eta_D, tau_K from donor; g from target."""
        return dict(
            alpha_P=self.donor_params["alpha_P"],
            eta_D=self.donor_params["eta_D"],
            tau_K=self.donor_params["tau_K"],
            g=self.target_g,
        )


TRANSFER_PAIRS: List[TransferPair] = [
    TransferPair(
        donor="graphite_2008",
        target="rare_earths_2010",
        rationale="China export quota + inelastic industrial demand",
        donor_params=_GRAPHITE_2008_PARAMS,
        target_g=_RARE_EARTHS_2010_PARAMS["g"],
    ),
    TransferPair(
        donor="graphite_2022",
        target="lithium_2022",
        rationale="EV-era regime, China-concentrated supply",
        donor_params=_GRAPHITE_2022_PARAMS,
        target_g=_LITHIUM_2022_PARAMS["g"],
    ),
    TransferPair(
        donor="cobalt_2016",
        target="lithium_2016",
        rationale="Early EV wave, battery metal, concentrated supply",
        donor_params=_COBALT_2016_PARAMS,
        target_g=_LITHIUM_2016_PARAMS["g"],
    ),
    TransferPair(
        donor="nickel_2006",
        target="cobalt_2016",
        rationale="Metal supply squeeze, stainless/EV demand transition",
        donor_params=_NICKEL_2006_PARAMS,
        target_g=_COBALT_2016_PARAMS["g"],
    ),
]


# ── Target episode runners ────────────────────────────────────────────────────
#
# Each function accepts a parameter dict and runs the TARGET episode's shocks,
# scoring against the target's actual price data.

def _run_rare_earths_2010(params: Dict[str, float]) -> EpisodeResult:
    p = params
    cfg = ScenarioConfig(
        name="rare_earths_2010_transfer",
        commodity="graphite",   # ODE is commodity-agnostic; graphite used as proxy
        seed=42,
        time=TimeConfig(dt=1.0, start_year=2008, end_year=2013),
        baseline=BaselineConfig(P_ref=1.0, P0=1.0, K0=108.695652, I0=20.0, D0=100.0),
        parameters=ParametersConfig(
            eps=1e-9, u0=0.92, beta_u=0.10, u_min=0.70, u_max=1.00,
            tau_K=p["tau_K"], eta_K=0.40, retire_rate=0.0, eta_D=p["eta_D"],
            demand_growth=DemandGrowthConfig(type="constant", g=p["g"]),
            alpha_P=p["alpha_P"], cover_star=0.20, lambda_cover=0.60, sigma_P=0.0,
        ),
        policy=PolicyConfig(),
        shocks=[
            ShockConfig(type="demand_surge",  start_year=2010, end_year=2010, magnitude=0.30),
            ShockConfig(type="policy_shock",  start_year=2010, end_year=2012, magnitude=0.40, quota_reduction=0.40),
            ShockConfig(type="capex_shock",   start_year=2010, end_year=2012, magnitude=0.50),
        ],
        outputs=OutputsConfig(metrics=["total_shortage", "peak_shortage", "avg_price", "final_inventory_cover"]),
    )
    df, _ = run_scenario(cfg)
    m = df.set_index("year")

    cepii = _cepii_series("data/canonical/cepii_rare_earths.csv", "China")
    years = [yr for yr in [2009, 2010, 2011, 2012, 2013] if yr in cepii.index]
    base = years[0]
    model_idx = m.loc[years, "P"] / m.loc[base, "P"]
    data_idx = cepii.loc[years, "implied_price"] / cepii.loc[base, "implied_price"]

    return EpisodeResult(
        name="rare_earths_2010_transfer",
        commodity="rare_earths",
        years=years,
        directional_accuracy=_directional_accuracy(model_idx, data_idx),
        spearman_rho=_spearman_rho(model_idx, data_idx),
        log_price_rmse=_log_price_rmse(model_idx, data_idx),
        magnitude_ratio=_magnitude_ratio(model_idx, data_idx),
        model_idx=model_idx,
        data_idx=data_idx,
    )


def _run_lithium_2022(params: Dict[str, float]) -> EpisodeResult:
    p = params
    cfg = ScenarioConfig(
        name="lithium_2022_transfer",
        commodity="lithium",
        seed=42,
        time=TimeConfig(dt=1.0, start_year=2019, end_year=2024),
        baseline=BaselineConfig(P_ref=1.0, P0=1.0, K0=108.695652, I0=20.0, D0=100.0),
        parameters=ParametersConfig(
            eps=1e-9, u0=0.92, beta_u=0.10, u_min=0.70, u_max=1.00,
            tau_K=p["tau_K"], eta_K=0.40, retire_rate=0.0, eta_D=p["eta_D"],
            demand_growth=DemandGrowthConfig(type="constant", g=p["g"]),
            alpha_P=p["alpha_P"], cover_star=0.20, lambda_cover=0.60, sigma_P=0.0,
        ),
        policy=PolicyConfig(),
        shocks=[
            ShockConfig(type="demand_surge", start_year=2021, end_year=2021, magnitude=0.35),
            ShockConfig(type="demand_surge", start_year=2022, end_year=2022, magnitude=0.40),
            ShockConfig(type="demand_surge", start_year=2023, end_year=2024, magnitude=-0.25),
        ],
        outputs=OutputsConfig(metrics=["total_shortage", "peak_shortage", "avg_price", "final_inventory_cover"]),
    )
    df, _ = run_scenario(cfg)
    m = df.set_index("year")

    cepii = _cepii_series("data/canonical/cepii_lithium.csv", "Australia")
    years = [yr for yr in [2020, 2021, 2022, 2023, 2024] if yr in cepii.index]
    base = years[0]
    model_idx = m.loc[years, "P"] / m.loc[base, "P"]
    data_idx = cepii.loc[years, "implied_price"] / cepii.loc[base, "implied_price"]

    return EpisodeResult(
        name="lithium_2022_transfer",
        commodity="lithium",
        years=years,
        directional_accuracy=_directional_accuracy(model_idx, data_idx),
        spearman_rho=_spearman_rho(model_idx, data_idx),
        log_price_rmse=_log_price_rmse(model_idx, data_idx),
        magnitude_ratio=_magnitude_ratio(model_idx, data_idx),
        model_idx=model_idx,
        data_idx=data_idx,
    )


def _run_lithium_2016(params: Dict[str, float]) -> EpisodeResult:
    p = params
    cfg = ScenarioConfig(
        name="lithium_2016_transfer",
        commodity="lithium",
        seed=42,
        time=TimeConfig(dt=1.0, start_year=2014, end_year=2019),
        baseline=BaselineConfig(P_ref=1.0, P0=1.0, K0=108.695652, I0=20.0, D0=100.0),
        parameters=ParametersConfig(
            eps=1e-9, u0=0.92, beta_u=0.10, u_min=0.70, u_max=1.00,
            tau_K=p["tau_K"], eta_K=0.40, retire_rate=0.0, eta_D=p["eta_D"],
            demand_growth=DemandGrowthConfig(type="constant", g=p["g"]),
            alpha_P=p["alpha_P"], cover_star=0.20, lambda_cover=0.60, sigma_P=0.0,
        ),
        policy=PolicyConfig(),
        shocks=[
            ShockConfig(type="demand_surge", start_year=2016, end_year=2016, magnitude=0.25),
            ShockConfig(type="demand_surge", start_year=2017, end_year=2017, magnitude=0.20),
        ],
        outputs=OutputsConfig(metrics=["total_shortage", "peak_shortage", "avg_price", "final_inventory_cover"]),
    )
    df, _ = run_scenario(cfg)
    m = df.set_index("year")

    cepii = _cepii_series("data/canonical/cepii_lithium.csv", "Chile")
    years = [yr for yr in [2015, 2016, 2017, 2018, 2019] if yr in cepii.index]
    base = years[0]
    model_idx = m.loc[years, "P"] / m.loc[base, "P"]
    data_idx = cepii.loc[years, "implied_price"] / cepii.loc[base, "implied_price"]

    return EpisodeResult(
        name="lithium_2016_transfer",
        commodity="lithium",
        years=years,
        directional_accuracy=_directional_accuracy(model_idx, data_idx),
        spearman_rho=_spearman_rho(model_idx, data_idx),
        log_price_rmse=_log_price_rmse(model_idx, data_idx),
        magnitude_ratio=_magnitude_ratio(model_idx, data_idx),
        model_idx=model_idx,
        data_idx=data_idx,
    )


def _run_cobalt_2016(params: Dict[str, float]) -> EpisodeResult:
    p = params
    cfg = ScenarioConfig(
        name="cobalt_2016_transfer",
        commodity="cobalt",
        seed=42,
        time=TimeConfig(dt=1.0, start_year=2013, end_year=2019),
        baseline=BaselineConfig(P_ref=1.0, P0=1.0, K0=108.695652, I0=20.0, D0=100.0),
        parameters=ParametersConfig(
            eps=1e-9, u0=0.92, beta_u=0.10, u_min=0.70, u_max=1.00,
            tau_K=p["tau_K"], eta_K=0.40, retire_rate=0.0, eta_D=p["eta_D"],
            demand_growth=DemandGrowthConfig(type="constant", g=p["g"]),
            alpha_P=p["alpha_P"], cover_star=0.20, lambda_cover=0.60, sigma_P=0.0,
        ),
        policy=PolicyConfig(),
        shocks=[
            ShockConfig(type="demand_surge", start_year=2016, end_year=2017, magnitude=0.30),
            ShockConfig(type="demand_surge", start_year=2018, end_year=2018, magnitude=-0.15),
        ],
        outputs=OutputsConfig(metrics=["total_shortage", "peak_shortage", "avg_price", "final_inventory_cover"]),
    )
    df, _ = run_scenario(cfg)
    m = df.set_index("year")

    wb = _wb_cobalt_series()
    years = [yr for yr in [2014, 2015, 2016, 2017, 2018, 2019] if yr in wb.index]
    base = years[0]
    model_idx = m.loc[years, "P"] / m.loc[base, "P"]
    data_idx = wb.loc[years, "implied_price"] / wb.loc[base, "implied_price"]

    return EpisodeResult(
        name="cobalt_2016_transfer",
        commodity="cobalt",
        years=years,
        directional_accuracy=_directional_accuracy(model_idx, data_idx),
        spearman_rho=_spearman_rho(model_idx, data_idx),
        log_price_rmse=_log_price_rmse(model_idx, data_idx),
        magnitude_ratio=_magnitude_ratio(model_idx, data_idx),
        model_idx=model_idx,
        data_idx=data_idx,
    )


# Registry: target name → runner function
_TARGET_RUNNERS = {
    "rare_earths_2010": _run_rare_earths_2010,
    "lithium_2022":     _run_lithium_2022,
    "lithium_2016":     _run_lithium_2016,
    "cobalt_2016":      _run_cobalt_2016,
}

# In-sample DA for each target episode (from predictability.py results)
_INSAMPLE_DA: Dict[str, float] = {
    "rare_earths_2010": 1.000,
    "lithium_2022":     1.000,
    "lithium_2016":     1.000,
    "cobalt_2016":      1.000,
}


# ── Zero-shot demonstration ───────────────────────────────────────────────────

def zero_shot_prediction(
    mineral: str,
    regime: str,
    target_g: float,
    shocks: List[ShockConfig],
    start_year: int,
    end_year: int,
    commodity_proxy: str = "graphite",
) -> pd.DataFrame:
    """
    Run a forward prediction for a mineral with no calibration data.

    Parameters
    ----------
    mineral    : descriptive name (for output labelling only)
    regime     : one of 'ev_restricted', 'pre_ev', 'long_cycle'
    target_g   : demand growth rate from external projections (e.g. IEA)
    shocks     : observable shock inputs derived from geopolitical events
    commodity_proxy : valid ScenarioConfig commodity to use as structural proxy

    Returns
    -------
    DataFrame with columns [year, P, K, I] — model price/capacity/inventory path
    """
    if regime not in REGIME_PRIORS:
        raise ValueError(f"Unknown regime '{regime}'. Choose: {list(REGIME_PRIORS)}")

    prior = REGIME_PRIORS[regime]
    p = dict(**prior, g=target_g)

    cfg = ScenarioConfig(
        name=f"{mineral}_zero_shot",
        commodity=commodity_proxy,
        seed=42,
        time=TimeConfig(dt=1.0, start_year=start_year, end_year=end_year),
        baseline=BaselineConfig(P_ref=1.0, P0=1.0, K0=108.695652, I0=20.0, D0=100.0),
        parameters=ParametersConfig(
            eps=1e-9, u0=0.92, beta_u=0.10, u_min=0.70, u_max=1.00,
            tau_K=p["tau_K"], eta_K=0.40, retire_rate=0.0, eta_D=p["eta_D"],
            demand_growth=DemandGrowthConfig(type="constant", g=p["g"]),
            alpha_P=p["alpha_P"], cover_star=0.20, lambda_cover=0.60, sigma_P=0.0,
        ),
        policy=PolicyConfig(),
        shocks=shocks,
        outputs=OutputsConfig(metrics=["total_shortage", "peak_shortage", "avg_price", "final_inventory_cover"]),
    )
    df, _ = run_scenario(cfg)
    return df[["year", "P", "K", "I"]].copy()


# ── Main evaluation ───────────────────────────────────────────────────────────

def run_cross_commodity_transfer() -> List[dict]:
    results = []
    for pair in TRANSFER_PAIRS:
        runner = _TARGET_RUNNERS[pair.target]
        result = runner(pair.transferred_params)
        insample = _INSAMPLE_DA.get(pair.target, float("nan"))
        results.append({
            "donor":         pair.donor,
            "target":        pair.target,
            "rationale":     pair.rationale,
            "transfer_da":   round(result.directional_accuracy, 3),
            "insample_da":   round(insample, 3),
            "spearman_rho":  round(result.spearman_rho, 3),
            "transferred_params": {
                k: round(v, 3) for k, v in pair.transferred_params.items()
            },
            "result": result,
        })
    return results


def _print_results(results: List[dict], verbose: bool = False) -> None:
    print("\n" + "=" * 70)
    print("  CROSS-COMMODITY PARAMETER TRANSFER")
    print("=" * 70)
    print(f"  {'Donor':<18} {'Target':<20} {'In-sample':>10} {'Transfer':>9} {'ρ':>6}")
    print("  " + "-" * 66)

    transfer_das = []
    for r in results:
        flag = "  ✓" if r["transfer_da"] >= 0.60 else ("  ~" if r["transfer_da"] >= 0.40 else "  ✗")
        print(
            f"  {r['donor']:<18} {r['target']:<20} "
            f"{r['insample_da']:>10.3f} {r['transfer_da']:>9.3f} "
            f"{r['spearman_rho']:>6.3f}{flag}"
        )
        transfer_das.append(r["transfer_da"])

    mean_transfer = sum(transfer_das) / len(transfer_das)
    mean_insample = sum(r["insample_da"] for r in results) / len(results)
    print("  " + "-" * 66)
    print(f"  {'Mean':<38} {mean_insample:>10.3f} {mean_transfer:>9.3f}")
    print(f"\n  DA degradation (in-sample → transfer): -{(mean_insample - mean_transfer)*100:.1f}pp")
    print(f"  Random baseline: 0.500")

    if verbose:
        print("\n  Transferred parameter sets:")
        for r in results:
            tp = r["transferred_params"]
            print(f"    {r['donor']} → {r['target']}:")
            print(f"      alpha_P={tp['alpha_P']}, eta_D={tp['eta_D']}, tau_K={tp['tau_K']}, g={tp['g']}")

    print("\n  Zero-shot regime priors (no calibration data):")
    print(f"  {'Regime':<20} {'alpha_P':>8} {'eta_D':>7} {'tau_K':>7}")
    print("  " + "-" * 44)
    for regime, p in REGIME_PRIORS.items():
        print(f"  {regime:<20} {p['alpha_P']:>8.2f} {p['eta_D']:>7.3f} {p['tau_K']:>7.1f}")
    print()


def main() -> None:
    verbose = "--verbose" in sys.argv or "-v" in sys.argv
    results = run_cross_commodity_transfer()
    _print_results(results, verbose=verbose)


if __name__ == "__main__":
    main()
