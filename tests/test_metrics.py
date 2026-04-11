"""Comprehensive tests for metrics.compute_metrics and helpers."""

from __future__ import annotations

import pytest
import pandas as pd

from src.minerals.metrics import compute_metrics
from src.minerals.schema import ShockConfig


def _make_df(years: list[int], shortage: list[float], P: list[float], cover: list[float]) -> pd.DataFrame:
    return pd.DataFrame({
        "year": years,
        "P": P,
        "K": [100.0] * len(years),
        "I": [20.0] * len(years),
        "Q_eff": [95.0] * len(years),
        "D": [100.0] * len(years),
        "shortage": shortage,
        "cover": cover,
    })


def test_compute_metrics_required_keys():
    df = _make_df(
        years=[2024, 2025, 2026],
        shortage=[0.0, 0.0, 0.0],
        P=[1.0, 1.0, 1.0],
        cover=[0.2, 0.2, 0.2],
    )
    m = compute_metrics(df, shocks=None)
    assert "total_shortage" in m
    assert "peak_shortage" in m
    assert "avg_price" in m
    assert "final_inventory_cover" in m


def test_compute_metrics_total_shortage_sum():
    df = _make_df(
        years=[2024, 2025, 2026],
        shortage=[1.0, 2.0, 3.0],
        P=[1.0, 1.0, 1.0],
        cover=[0.2, 0.2, 0.2],
    )
    m = compute_metrics(df, shocks=None)
    assert m["total_shortage"] == 6.0
    assert m["peak_shortage"] == 3.0


def test_compute_metrics_avg_price():
    df = _make_df(
        years=[2024, 2025],
        shortage=[0.0, 0.0],
        P=[1.0, 3.0],
        cover=[0.2, 0.2],
    )
    m = compute_metrics(df, shocks=None)
    assert m["avg_price"] == 2.0


def test_compute_metrics_final_inventory_cover():
    df = _make_df(
        years=[2024, 2025, 2026],
        shortage=[0.0, 0.0, 0.0],
        P=[1.0, 1.0, 1.0],
        cover=[0.1, 0.2, 0.25],
    )
    m = compute_metrics(df, shocks=None)
    assert m["final_inventory_cover"] == 0.25


def test_compute_metrics_no_shocks_omits_shock_window():
    df = _make_df(
        years=[2024, 2025],
        shortage=[0.0, 0.0],
        P=[1.0, 1.0],
        cover=[0.2, 0.2],
    )
    m = compute_metrics(df, shocks=None)
    assert "shock_window_total_shortage" not in m
    assert "post_shock_total_shortage" not in m


def test_compute_metrics_with_shocks_adds_shock_window_and_post():
    df = _make_df(
        years=[2024, 2025, 2026, 2027],
        shortage=[0.0, 5.0, 3.0, 0.0],
        P=[1.0, 1.0, 1.0, 1.0],
        cover=[0.2, 0.2, 0.2, 0.2],
    )
    shocks = [
        ShockConfig(type="export_restriction", start_year=2025, end_year=2026, magnitude=0.2),
    ]
    m = compute_metrics(df, shocks=shocks)
    assert "shock_window_total_shortage" in m
    assert "post_shock_total_shortage" in m
    assert m["shock_window_total_shortage"] == 8.0  # 5 + 3 in 2025, 2026
    assert m["post_shock_total_shortage"] == 0.0   # only 2027 after 2026


def test_compute_metrics_shock_year_shortage():
    df = _make_df(
        years=[2024, 2025, 2026],
        shortage=[0.0, 10.0, 5.0],
        P=[1.0, 1.0, 1.0],
        cover=[0.2, 0.2, 0.2],
    )
    shocks = [
        ShockConfig(type="export_restriction", start_year=2025, end_year=2026, magnitude=0.2),
    ]
    m = compute_metrics(df, shocks=shocks)
    assert "shock_year_shortage" in m
    assert m["shock_year_shortage"] == 10.0  # first shock year is 2025


def test_compute_metrics_empty_shocks_list_same_as_none():
    df = _make_df(
        years=[2024],
        shortage=[0.0],
        P=[1.0],
        cover=[0.2],
    )
    m_none = compute_metrics(df, shocks=None)
    m_empty = compute_metrics(df, shocks=[])
    assert m_none.keys() == m_empty.keys()
    for k in m_none:
        assert m_none[k] == m_empty[k]
