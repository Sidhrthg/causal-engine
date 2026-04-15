"""
Tests for the causal engine predictability evaluation.

These tests verify that:
1. The evaluation runs without error
2. Metric values are in valid ranges
3. Known-good episodes meet their expected grade
4. Known structural gaps are correctly identified (model fails where expected)

Importantly, failing predictions are NOT fixed here — they are documented.
A test that asserts a known gap confirms the model behaves consistently.
"""

import math
import pytest
from src.minerals.predictability import (
    run_predictability_evaluation,
    EpisodeResult,
    _directional_accuracy,
    _spearman_rho,
    _log_price_rmse,
    _magnitude_ratio,
)
import pandas as pd


@pytest.fixture(scope="module")
def results():
    return run_predictability_evaluation()


@pytest.fixture(scope="module")
def by_name(results):
    return {r.name: r for r in results}


# ── Structural checks ─────────────────────────────────────────────────────────

def test_all_episodes_run(results):
    """Evaluation produces results for all three episodes."""
    names = {r.name for r in results}
    assert "graphite_2008_demand_spike_and_quota"          in names
    assert "graphite_2022_ev_surge_and_export_controls"    in names
    assert "lithium_2022_ev_boom"                          in names


def test_metrics_in_valid_ranges(results):
    """All metrics are finite and in expected ranges."""
    for r in results:
        assert not math.isnan(r.directional_accuracy), f"{r.name}: DA is NaN"
        assert 0.0 <= r.directional_accuracy <= 1.0,   f"{r.name}: DA out of [0,1]"
        assert not math.isnan(r.spearman_rho),          f"{r.name}: rho is NaN"
        assert -1.0 <= r.spearman_rho <= 1.0,           f"{r.name}: rho out of [-1,1]"
        assert not math.isnan(r.log_price_rmse),        f"{r.name}: RMSE is NaN"
        assert r.log_price_rmse >= 0.0,                 f"{r.name}: RMSE < 0"
        assert not math.isnan(r.magnitude_ratio),       f"{r.name}: MagR is NaN"
        assert r.magnitude_ratio >= 0.0,                f"{r.name}: MagR < 0"


def test_grade_is_valid(results):
    """Every episode has a valid grade."""
    valid_grades = {"A", "B", "C", "F", "?"}
    for r in results:
        assert r.grade in valid_grades, f"{r.name}: invalid grade {r.grade!r}"


def test_to_dict_contains_required_keys(results):
    required = {"name", "commodity", "years", "grade",
                "directional_accuracy", "spearman_rho",
                "log_price_rmse", "magnitude_ratio", "known_gap"}
    for r in results:
        d = r.to_dict()
        assert required <= set(d.keys()), f"{r.name}: missing keys {required - set(d.keys())}"


# ── Episode-specific predictability checks ────────────────────────────────────

class TestGraphite2008Predictability:
    """
    The 2008 demand spike is the best-calibrated graphite episode.
    The model should get directional accuracy ≥ 0.60 (3/5 year-on-year moves right).
    Known gap: 2010 price crashes in model due to GFC surplus, while CEPII rises.
    """

    def test_directional_accuracy_at_least_60pct(self, by_name):
        r = by_name["graphite_2008_demand_spike_and_quota"]
        assert r.directional_accuracy >= 0.60, (
            f"Graphite 2008 DA={r.directional_accuracy:.2f} < 0.60. "
            f"Model should predict ≥3 of 5 directional moves correctly."
        )

    def test_price_rise_2008_2009_predicted(self, by_name):
        """Model predicts price rises 2008→2009 (shortage response with 1-year lag)."""
        r = by_name["graphite_2008_demand_spike_and_quota"]
        m = r.model_idx
        assert m.loc[2009] > m.loc[2007], (
            f"Model should show net price rise by 2009 after 2008 demand surge: "
            f"idx_2007={m.loc[2007]:.3f}, idx_2009={m.loc[2009]:.3f}"
        )

    def test_known_gap_2010_price_crash(self, by_name):
        """
        Documents structural gap: model crashes P in 2010 (GFC surplus + quota)
        while CEPII shows continued rise.  This confirms the gap is stable.
        """
        r = by_name["graphite_2008_demand_spike_and_quota"]
        m = r.model_idx
        d = r.data_idx
        model_drops_2010 = m.loc[2010] < m.loc[2009]
        cepii_rises_2010 = d.loc[2010] > d.loc[2009]
        assert model_drops_2010 and cepii_rises_2010, (
            f"Known gap should hold: model drops in 2010, CEPII rises. "
            f"model: {m.loc[2009]:.3f}→{m.loc[2010]:.3f}, "
            f"CEPII: {d.loc[2009]:.3f}→{d.loc[2010]:.3f}"
        )


class TestGraphite2023Predictability:
    """
    The 2022-23 graphite episode: partial export controls, buyer diversification.
    Model is expected to under-perform here (known structural gap: no supply substitution).
    """

    def test_price_rises_after_demand_surge(self, by_name):
        """Model predicts price rise in 2023 after 2022 demand surge (causal core)."""
        r = by_name["graphite_2022_ev_surge_and_export_controls"]
        m = r.model_idx
        assert m.loc[2023] > m.loc[2021], (
            f"Model should predict price rise 2021→2023: "
            f"idx_2021={m.loc[2021]:.3f}, idx_2023={m.loc[2023]:.3f}"
        )

    def test_known_gap_post_2022_divergence(self, by_name):
        """
        Documents structural gap: model predicts rising price 2022→2024 while
        CEPII shows price declined.  Confirms the gap is stable and not masked.
        """
        r = by_name["graphite_2022_ev_surge_and_export_controls"]
        m = r.model_idx
        d = r.data_idx
        model_rises = m.loc[2024] > m.loc[2022]
        cepii_falls = d.loc[2024] < d.loc[2022]
        assert model_rises and cepii_falls, (
            f"Known gap should hold: model rises, CEPII falls 2022→2024. "
            f"model: {m.loc[2022]:.3f}→{m.loc[2024]:.3f}, "
            f"CEPII: {d.loc[2022]:.3f}→{d.loc[2024]:.3f}"
        )

    def test_directional_accuracy_below_threshold(self, by_name):
        """
        DA < 0.60 is expected here — documents that the model cannot predict
        post-export-control price direction for graphite.
        """
        r = by_name["graphite_2022_ev_surge_and_export_controls"]
        assert r.directional_accuracy < 0.60, (
            f"This episode is expected to fail directional accuracy (known gap). "
            f"If DA improved to {r.directional_accuracy:.2f} ≥ 0.60, the structural "
            f"gap may have been fixed — verify and update this test."
        )


class TestLithium2022Predictability:
    """
    Lithium 2022 EV boom: model correctly predicts the surge direction.
    Known gap: model cannot reproduce the 2023-2024 price collapse.
    """

    def test_directional_accuracy_at_least_60pct(self, by_name):
        """Model gets direction right on most moves despite missing the collapse."""
        r = by_name["lithium_2022_ev_boom"]
        assert r.directional_accuracy >= 0.60, (
            f"Lithium 2022 DA={r.directional_accuracy:.2f} < 0.60"
        )

    def test_price_rises_after_2022_surge(self, by_name):
        """Core causal chain: demand_surge → shortage → P rises (lag-adjusted)."""
        r = by_name["lithium_2022_ev_boom"]
        m = r.model_idx
        assert m.loc[2023] > m.loc[2021], (
            f"Model should predict price rise 2021→2023: {m.loc[2021]:.3f}→{m.loc[2023]:.3f}"
        )

    def test_known_gap_price_collapse_not_reproduced(self, by_name):
        """
        Documents structural gap: model P rises 2022→2024 while CEPII collapses -72 %.
        The model lacks cost-curve supply expansion and inventory liquidation.
        """
        r = by_name["lithium_2022_ev_boom"]
        m = r.model_idx
        d = r.data_idx
        model_rises_to_2024 = m.loc[2024] > m.loc[2022]
        cepii_collapses = (d.loc[2024] - d.loc[2022]) / d.loc[2022] < -0.50
        assert model_rises_to_2024, (
            f"Model should still predict rising price 2022→2024 (gap check): "
            f"{m.loc[2022]:.3f}→{m.loc[2024]:.3f}"
        )
        assert cepii_collapses, (
            f"CEPII should show >50 % price collapse 2022→2024 (gap check): "
            f"{d.loc[2022]:.3f}→{d.loc[2024]:.3f}"
        )


# ── Metric unit tests ─────────────────────────────────────────────────────────

def test_directional_accuracy_perfect():
    m = pd.Series([1.0, 1.2, 1.5, 1.3], index=[2020, 2021, 2022, 2023])
    d = pd.Series([1.0, 1.1, 1.4, 1.2], index=[2020, 2021, 2022, 2023])
    assert _directional_accuracy(m, d) == 1.0


def test_directional_accuracy_zero():
    m = pd.Series([1.0, 1.2, 1.4], index=[2020, 2021, 2022])
    d = pd.Series([1.0, 0.9, 0.7], index=[2020, 2021, 2022])
    assert _directional_accuracy(m, d) == 0.0


def test_log_price_rmse_perfect():
    s = pd.Series([1.0, 1.5, 2.0], index=[2020, 2021, 2022])
    assert _log_price_rmse(s, s) == pytest.approx(0.0, abs=1e-10)


def test_magnitude_ratio_perfect():
    m = pd.Series([1.0, 1.2, 1.44], index=[2020, 2021, 2022])
    d = pd.Series([1.0, 1.2, 1.44], index=[2020, 2021, 2022])
    assert _magnitude_ratio(m, d) == pytest.approx(1.0, abs=1e-6)
