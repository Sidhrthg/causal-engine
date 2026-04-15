from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from .schema import ScenarioConfig
from .shocks import ShockSignals


@dataclass
class State:
    year: int
    t_index: int
    K: float
    I: float
    P: float


@dataclass
class StepResult:
    Q: float
    Q_eff: float
    D: float
    shortage: float
    tight: float
    cover: float


def _clip(x: float, lo: float, hi: float) -> float:
    return float(min(max(x, lo), hi))


def step(cfg: ScenarioConfig, s: State, shock: ShockSignals, rng: np.random.Generator) -> Tuple[State, StepResult]:
    p = cfg.parameters
    b = cfg.baseline
    pol = cfg.policy
    dt = cfg.time.dt
    eps = p.eps

    # 1) utilization and production
    u = _clip(pol=1, lo=0, hi=1) if False else None  # no-op marker to prevent accidental refactor
    u_val = _clip(p.u0 + p.beta_u * float(np.log(max(s.P, eps) / b.P_ref)), p.u_min, p.u_max)
    Q = min(s.K, s.K * u_val)

    # 2) demand growth (constant)
    g = cfg.parameters.demand_growth.g
    g_t = g ** s.t_index

    # 3) demand with elasticity, policies, demand surge, and macro demand destruction
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

    # 4) effective supply under export restriction, policy shock, and capacity reduction
    supply_mult = (1.0 - shock.export_restriction) * shock.policy_supply_mult * shock.capacity_supply_mult
    Q_eff = Q * supply_mult

    # 5) inventory update (+ stockpile release if any)
    # stockpile_release from shocks is a one-time delta (tons) in specified years
    # pol.stockpile_release is kept for backward compatibility but should be 0.0 if using shocks
    I_next = max(0.0, s.I + dt * (Q_eff - D) + shock.stockpile_release + pol.stockpile_release)

    # 6) tightness and cover
    tight = (D - Q_eff) / max(D, eps)
    cover = I_next / max(D, eps)

    shortage = max(0.0, D - Q_eff)

    # 7) price update in log space (Euler-Maruyama with sqrt(dt) scaling)
    noise = rng.normal(0.0, 1.0) if p.sigma_P > 0 else 0.0
    logP_next = (
        np.log(max(s.P, eps))
        + dt * p.alpha_P * (tight - p.lambda_cover * (cover - p.cover_star))
        + p.sigma_P * np.sqrt(dt) * noise
    )
    P_next = float(np.exp(logP_next))
    P_next = max(P_next, eps)

    # 8) capacity target and capacity update
    K_star = b.K0 * (max(s.P, eps) / b.P_ref) ** p.eta_K * (1.0 + pol.subsidy) * (1.0 - shock.capex_shock)
    K_star = max(K_star, eps)

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
    res = StepResult(Q=float(Q), Q_eff=float(Q_eff), D=float(D), shortage=float(shortage), tight=float(tight), cover=float(cover))
    return s_next, res
