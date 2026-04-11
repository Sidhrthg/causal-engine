"""Comprehensive tests for shocks.ShockSignals, apply_shocks, shocks_for_year."""

from __future__ import annotations

import pytest

from src.minerals.schema import ShockConfig
from src.minerals.shocks import ShockSignals, apply_shocks, shocks_for_year


def test_shock_signals_defaults():
    s = ShockSignals()
    assert s.export_restriction == 0.0
    assert s.demand_surge == 0.0
    assert s.capex_shock == 0.0
    assert s.stockpile_release == 0.0
    assert s.policy_supply_mult == 1.0
    assert s.capacity_supply_mult == 1.0
    assert s.demand_destruction_mult == 1.0


def test_shock_signals_custom_values():
    s = ShockSignals(export_restriction=0.2, demand_surge=0.1, policy_supply_mult=0.8)
    assert s.export_restriction == 0.2
    assert s.demand_surge == 0.1
    assert s.policy_supply_mult == 0.8


def test_shocks_for_year_empty_list():
    sig = shocks_for_year([], 2025)
    assert sig.export_restriction == 0.0
    assert sig.demand_surge == 0.0
    assert sig.policy_supply_mult == 1.0
    assert sig.demand_destruction_mult == 1.0


def test_shocks_for_year_outside_shock_window():
    shocks = [
        ShockConfig(type="export_restriction", start_year=2026, end_year=2027, magnitude=0.2),
    ]
    sig = shocks_for_year(shocks, 2025)
    assert sig.export_restriction == 0.0


def test_shocks_for_year_export_restriction_active():
    shocks = [
        ShockConfig(type="export_restriction", start_year=2025, end_year=2026, magnitude=0.3),
    ]
    sig = shocks_for_year(shocks, 2025)
    assert sig.export_restriction == 0.3


def test_shocks_for_year_demand_surge_active():
    shocks = [
        ShockConfig(type="demand_surge", start_year=2025, end_year=2026, magnitude=0.1),
    ]
    sig = shocks_for_year(shocks, 2025)
    assert sig.demand_surge == 0.1


def test_shocks_for_year_stockpile_release_active():
    shocks = [
        ShockConfig(type="stockpile_release", start_year=2025, end_year=2025, magnitude=100.0),
    ]
    sig = shocks_for_year(shocks, 2025)
    assert sig.stockpile_release == 100.0


def test_apply_shocks_empty():
    out = apply_shocks(2025.0, [])
    assert out["export_restriction"] == 1.0
    assert out["demand_shock"] == 1.0
    assert out["policy_shock"] == 1.0
    assert out["capacity_shock"] == 1.0


def test_apply_shocks_export_restriction_multiplier():
    # export_restriction: impact is 1.0 - magnitude (not used in same way in apply_shocks for legacy signals;
    # shocks_for_year uses additive export_restriction). apply_shocks returns multipliers for policy/capacity/demand.
    # For export_restriction type, apply_shocks does: shock_impacts["export_restriction"] *= 1.0 - magnitude
    shocks = [
        ShockConfig(type="export_restriction", start_year=2025, end_year=2026, magnitude=0.2),
    ]
    out = apply_shocks(2025.0, shocks)
    assert out["export_restriction"] == 0.8


def test_apply_shocks_policy_shock():
    shocks = [
        ShockConfig(type="policy_shock", start_year=2025, end_year=2026, magnitude=0.3),
    ]
    out = apply_shocks(2025.0, shocks)
    assert out["policy_shock"] == 0.7


def test_apply_shocks_macro_demand_shock():
    # magnitude -0.4 => 40% demand drop => demand_destruction_mult = 1 + (-0.4) = 0.6
    shocks = [
        ShockConfig(type="macro_demand_shock", start_year=2025, end_year=2026, magnitude=-0.4),
    ]
    out = apply_shocks(2025.0, shocks)
    assert out["demand_shock"] == pytest.approx(0.6)


def test_apply_shocks_demand_surge():
    shocks = [
        ShockConfig(type="demand_surge", start_year=2025, end_year=2026, magnitude=0.2),
    ]
    out = apply_shocks(2025.0, shocks)
    assert out["demand_shock"] == 1.2


def test_shocks_for_year_policy_supply_mult_from_apply_shocks():
    shocks = [
        ShockConfig(type="policy_shock", start_year=2025, end_year=2026, magnitude=0.2),
    ]
    sig = shocks_for_year(shocks, 2025)
    assert sig.policy_supply_mult == 0.8


def test_shocks_for_year_capacity_reduction():
    shocks = [
        ShockConfig(type="capacity_reduction", start_year=2025, end_year=2026, magnitude=0.1),
    ]
    out = apply_shocks(2025.0, shocks)
    assert out["capacity_shock"] == 0.9
    sig = shocks_for_year(shocks, 2025)
    assert sig.capacity_supply_mult == 0.9


def test_shocks_for_year_multiple_years():
    shocks = [
        ShockConfig(type="export_restriction", start_year=2024, end_year=2026, magnitude=0.1),
    ]
    assert shocks_for_year(shocks, 2023).export_restriction == 0.0
    assert shocks_for_year(shocks, 2024).export_restriction == 0.1
    assert shocks_for_year(shocks, 2025).export_restriction == 0.1
    assert shocks_for_year(shocks, 2026).export_restriction == 0.1
    assert shocks_for_year(shocks, 2027).export_restriction == 0.0
