"""Comprehensive tests for scenario schema and load_scenario."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from src.minerals.schema import (
    BaselineConfig,
    DemandGrowthConfig,
    OutputsConfig,
    ParametersConfig,
    PolicyConfig,
    ScenarioConfig,
    ShockConfig,
    TimeConfig,
    load_scenario,
)


# ----- TimeConfig -----


def test_time_config_valid():
    tc = TimeConfig(dt=1.0, start_year=2020, end_year=2030)
    assert tc.dt == 1.0
    assert tc.start_year == 2020
    assert tc.end_year == 2030


def test_time_config_end_year_must_be_after_start():
    with pytest.raises(ValueError, match="end_year must be after start_year"):
        TimeConfig(dt=1.0, start_year=2030, end_year=2025)


def test_time_config_end_year_equals_start_invalid():
    with pytest.raises(ValueError, match="end_year must be after start_year"):
        TimeConfig(dt=1.0, start_year=2025, end_year=2025)


# ----- BaselineConfig -----


def test_baseline_config_valid():
    b = BaselineConfig(P_ref=1.0, P0=1.0, K0=100.0, I0=20.0, D0=100.0)
    assert b.P_ref == 1.0
    assert b.D0 == 100.0


def test_baseline_config_positive_fields():
    with pytest.raises(ValueError):
        BaselineConfig(P_ref=0.0, P0=1.0, K0=100.0, I0=20.0, D0=100.0)
    with pytest.raises(ValueError):
        BaselineConfig(P_ref=1.0, P0=-1.0, K0=100.0, I0=20.0, D0=100.0)


# ----- ShockConfig -----


@pytest.mark.parametrize(
    "shock_type",
    [
        "export_restriction",
        "demand_surge",
        "capex_shock",
        "stockpile_release",
        "capacity_reduction",
        "policy_shock",
        "macro_demand_shock",
    ],
)
def test_shock_config_types(shock_type: str):
    s = ShockConfig(type=shock_type, start_year=2025, end_year=2026, magnitude=0.1)
    assert s.type == shock_type
    assert s.start_year == 2025
    assert s.end_year == 2026
    assert s.magnitude == 0.1


def test_shock_config_invalid_type():
    with pytest.raises(ValueError):
        ShockConfig(type="invalid", start_year=2025, end_year=2026, magnitude=0.1)


# ----- ScenarioConfig commodity -----


def test_scenario_config_commodity_normalized_to_lower():
    # Build minimal valid config with commodity "Graphite" -> should become "graphite"
    cfg = ScenarioConfig(
        name="test",
        commodity="Graphite",
        seed=42,
        time=TimeConfig(dt=1.0, start_year=2024, end_year=2030),
        baseline=BaselineConfig(P_ref=1.0, P0=1.0, K0=100.0, I0=20.0, D0=100.0),
        parameters=ParametersConfig(
            u0=0.9,
            beta_u=0.1,
            u_min=0.7,
            u_max=1.0,
            tau_K=3.0,
            eta_K=0.4,
            retire_rate=0.0,
            eta_D=-0.25,
            demand_growth=DemandGrowthConfig(g=1.0),
            alpha_P=0.8,
            cover_star=0.2,
            lambda_cover=0.6,
        ),
        outputs=OutputsConfig(metrics=["total_shortage", "peak_shortage", "avg_price", "final_inventory_cover"]),
    )
    assert cfg.commodity == "graphite"


def test_scenario_config_unsupported_commodity():
    with pytest.raises(ValueError, match="Unsupported commodity"):
        ScenarioConfig(
            name="test",
            commodity="unobtainium",
            seed=42,
            time=TimeConfig(dt=1.0, start_year=2024, end_year=2030),
            baseline=BaselineConfig(P_ref=1.0, P0=1.0, K0=100.0, I0=20.0, D0=100.0),
            parameters=ParametersConfig(
                u0=0.9,
                beta_u=0.1,
                u_min=0.7,
                u_max=1.0,
                tau_K=3.0,
                eta_K=0.4,
                retire_rate=0.0,
                eta_D=-0.25,
                demand_growth=DemandGrowthConfig(g=1.0),
                alpha_P=0.8,
                cover_star=0.2,
                lambda_cover=0.6,
            ),
            outputs=OutputsConfig(metrics=["total_shortage"]),
        )


def test_scenario_config_years_property():
    cfg = load_scenario("scenarios/graphite_baseline.yaml")
    assert cfg.years == list(range(cfg.time.start_year, cfg.time.end_year + 1))
    assert cfg.years[0] == cfg.time.start_year
    assert cfg.years[-1] == cfg.time.end_year


# ----- load_scenario -----


def test_load_scenario_success():
    cfg = load_scenario("scenarios/graphite_baseline.yaml")
    assert cfg.name == "graphite_baseline"
    assert cfg.commodity == "graphite"
    assert cfg.seed == 123
    assert cfg.time.start_year == 2024
    assert cfg.time.end_year == 2030
    assert len(cfg.shocks) == 0
    assert "total_shortage" in cfg.outputs.metrics


def test_load_scenario_file_not_found():
    with pytest.raises(FileNotFoundError, match="Scenario file not found"):
        load_scenario("scenarios/nonexistent.yaml")


def test_load_scenario_with_shocks():
    path = "scenarios/graphite_export_restriction.yaml"
    if not Path(path).exists():
        pytest.skip(f"Scenario file not found: {path}")
    cfg = load_scenario(path)
    assert len(cfg.shocks) >= 1
    assert cfg.shocks[0].type == "export_restriction"


def test_policy_config_defaults():
    pol = PolicyConfig()
    assert pol.substitution == 0.0
    assert pol.efficiency == 0.0
    assert pol.subsidy == 0.0
    assert pol.stockpile_release == 0.0


def test_outputs_config_defaults():
    out = OutputsConfig(metrics=["total_shortage"])
    assert out.out_dir == "runs"
    assert out.save_csv is True
