"""Comprehensive tests for model.State, model.step, and StepResult."""

from __future__ import annotations

import numpy as np
import pytest

from src.minerals.model import State, StepResult, step
from src.minerals.schema import (
    BaselineConfig,
    DemandGrowthConfig,
    OutputsConfig,
    ParametersConfig,
    PolicyConfig,
    ScenarioConfig,
    TimeConfig,
)
from src.minerals.shocks import ShockSignals


def _minimal_config(
    sigma_P: float = 0.0,
    g: float = 1.0,
    retire_rate: float = 0.0,
) -> ScenarioConfig:
    return ScenarioConfig(
        name="test",
        commodity="graphite",
        seed=42,
        time=TimeConfig(dt=1.0, start_year=2024, end_year=2030),
        baseline=BaselineConfig(P_ref=1.0, P0=1.0, K0=100.0, I0=20.0, D0=100.0),
        parameters=ParametersConfig(
            eps=1e-9,
            u0=0.92,
            beta_u=0.10,
            u_min=0.70,
            u_max=1.00,
            tau_K=3.0,
            eta_K=0.40,
            retire_rate=retire_rate,
            eta_D=-0.25,
            demand_growth=DemandGrowthConfig(type="constant", g=g),
            alpha_P=0.80,
            cover_star=0.20,
            lambda_cover=0.60,
            sigma_P=sigma_P,
        ),
        policy=PolicyConfig(),
        shocks=[],
        outputs=OutputsConfig(metrics=["total_shortage", "peak_shortage", "avg_price", "final_inventory_cover"]),
    )


@pytest.fixture
def minimal_cfg() -> ScenarioConfig:
    return _minimal_config()


def test_state_creation():
    s = State(year=2024, t_index=0, K=100.0, I=20.0, P=1.0)
    assert s.year == 2024
    assert s.t_index == 0
    assert s.K == 100.0
    assert s.I == 20.0
    assert s.P == 1.0


def test_step_result_has_all_fields():
    r = StepResult(Q=90.0, Q_eff=90.0, D=100.0, shortage=10.0, tight=0.1, cover=0.2)
    assert r.Q == 90.0
    assert r.Q_eff == 90.0
    assert r.D == 100.0
    assert r.shortage == 10.0
    assert r.tight == 0.1
    assert r.cover == 0.2


def test_step_no_shock_deterministic(minimal_cfg: ScenarioConfig):
    s0 = State(year=2024, t_index=0, K=100.0, I=20.0, P=1.0)
    no_shock = ShockSignals()
    rng = np.random.default_rng(42)

    s1, res = step(minimal_cfg, s0, no_shock, rng)

    assert s1.year == 2025
    assert s1.t_index == 1
    assert s1.K >= 0
    assert s1.I >= 0
    assert s1.P > 0
    assert res.Q >= 0
    assert res.Q_eff >= 0
    assert res.D > 0
    assert res.shortage >= 0
    assert res.cover >= 0


def test_step_with_export_restriction_increases_shortage(minimal_cfg: ScenarioConfig):
    s0 = State(year=2024, t_index=0, K=100.0, I=20.0, P=1.0)
    rng = np.random.default_rng(42)

    no_shock = ShockSignals()
    with_restriction = ShockSignals(export_restriction=0.3)

    _, res_no = step(minimal_cfg, s0, no_shock, rng)
    _, res_restrict = step(minimal_cfg, s0, with_restriction, rng)

    assert res_restrict.Q_eff < res_no.Q_eff
    assert res_restrict.shortage >= res_no.shortage


def test_step_demand_surge_increases_demand(minimal_cfg: ScenarioConfig):
    s0 = State(year=2024, t_index=0, K=100.0, I=20.0, P=1.0)
    rng = np.random.default_rng(42)

    no_shock = ShockSignals()
    surge = ShockSignals(demand_surge=0.2)

    _, res_no = step(minimal_cfg, s0, no_shock, rng)
    _, res_surge = step(minimal_cfg, s0, surge, rng)

    assert res_surge.D > res_no.D


def test_step_stockpile_release_increases_inventory(minimal_cfg: ScenarioConfig):
    s0 = State(year=2024, t_index=0, K=100.0, I=20.0, P=1.0)
    rng = np.random.default_rng(42)

    no_shock = ShockSignals()
    release = ShockSignals(stockpile_release=50.0)

    s_no, _ = step(minimal_cfg, s0, no_shock, rng)
    s_release, _ = step(minimal_cfg, s0, release, rng)

    assert s_release.I > s_no.I


def test_step_deterministic_with_sigma_zero(minimal_cfg: ScenarioConfig):
    s0 = State(year=2024, t_index=0, K=100.0, I=20.0, P=1.0)
    no_shock = ShockSignals()
    rng1 = np.random.default_rng(0)
    rng2 = np.random.default_rng(999)

    s1, r1 = step(minimal_cfg, s0, no_shock, rng1)
    s2, r2 = step(minimal_cfg, s0, no_shock, rng2)

    assert s1.P == s2.P
    assert r1.shortage == r2.shortage
    assert s1.I == s2.I


def test_step_year_increments_by_dt(minimal_cfg: ScenarioConfig):
    s0 = State(year=2024, t_index=0, K=100.0, I=20.0, P=1.0)
    no_shock = ShockSignals()
    rng = np.random.default_rng(42)

    s_next, _ = step(minimal_cfg, s0, no_shock, rng)
    assert s_next.year == s0.year + int(minimal_cfg.time.dt)
    assert s_next.t_index == s0.t_index + 1
