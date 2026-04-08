from __future__ import annotations

from dataclasses import asdict
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from .schema import ScenarioConfig
from .shocks import shocks_for_year
from .model import State, step
from .metrics import compute_metrics


def run_scenario(cfg: ScenarioConfig) -> Tuple[pd.DataFrame, Dict[str, float]]:
    rng = np.random.default_rng(cfg.seed)

    # initialize
    s = State(
        year=cfg.time.start_year,
        t_index=0,
        K=cfg.baseline.K0,
        I=cfg.baseline.I0,
        P=cfg.baseline.P0,
    )

    rows = []
    years = cfg.years
    # run transitions for each year except last (since next state uses year+dt)
    for idx, year in enumerate(years):
        shock = shocks_for_year(cfg.shocks, year)
        # record current state row first (pre-step)
        # step to next
        s_next, res = step(cfg, s, shock, rng)
        row = {
            "year": year,
            "K": s.K,
            "I": s.I,
            "P": s.P,
            "Q": res.Q,
            "Q_eff": res.Q_eff,
            "D": res.D,
            "shortage": res.shortage,
            "tight": res.tight,
            "cover": res.cover,
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
    metrics = compute_metrics(df, cfg.shocks)
    return df, metrics
