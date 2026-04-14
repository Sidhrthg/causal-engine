"""Tests for causal_identification: SyntheticControl, InstrumentalVariable,
RegressionDiscontinuity, DifferenceInDifferences."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.minerals.causal_identification import (
    SyntheticControl, TreatmentEffect,
    InstrumentalVariable, IVResult,
    RegressionDiscontinuity, RDResult,
    DifferenceInDifferences, DIDResult,
)


def _synthetic_panel(
    years: list[int],
    units: list[str],
    treated_unit: str,
    treatment_time: int,
    treatment_effect_size: float = -20.0,
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for year in years:
        for unit in units:
            base = 100.0 + (year - years[0]) * 2.0 + rng.normal(0, 2.0)
            if unit == treated_unit and year >= treatment_time:
                base += treatment_effect_size
            rows.append({"country": unit, "year": year, "trade_value": base})
    return pd.DataFrame(rows)


@pytest.fixture
def minimal_panel() -> pd.DataFrame:
    years = list(range(2005, 2015))
    units = ["USA", "EU", "Japan"]
    return _synthetic_panel(years, units, treated_unit="USA", treatment_time=2010)


def test_treatment_effect_dataclass():
    actual = pd.Series([1.0, 2.0], index=[2010, 2011])
    counterfactual = pd.Series([1.2, 2.1], index=[2010, 2011])
    te = TreatmentEffect(
        treatment_effect=actual - counterfactual,
        counterfactual=counterfactual,
        actual=actual,
        weights={"EU": 0.5, "Japan": 0.5},
        pre_treatment_rmse=0.1,
        post_treatment_years=[2010, 2011],
    )
    assert len(te.treatment_effect) == 2
    assert te.pre_treatment_rmse == 0.1
    assert te.weights["EU"] == 0.5
    assert 2010 in te.post_treatment_years


def test_synthetic_control_estimate_treatment_effect(minimal_panel: pd.DataFrame):
    sc = SyntheticControl(verbose=False)
    result = sc.estimate_treatment_effect(
        data=minimal_panel,
        treated_unit="USA",
        control_units=["EU", "Japan"],
        treatment_time=2010,
        outcome_var="trade_value",
        unit_col="country",
        time_col="year",
    )
    assert isinstance(result, TreatmentEffect)
    assert result.actual is not None
    assert result.counterfactual is not None
    assert result.treatment_effect is not None
    assert len(result.weights) == 2
    assert "EU" in result.weights
    assert "Japan" in result.weights
    assert result.pre_treatment_rmse >= 0.0
    assert len(result.post_treatment_years) >= 1
    # We injected negative effect for USA after 2010, so mean treatment effect should be negative
    assert result.treatment_effect.mean() < 0


def test_synthetic_control_no_pre_treatment_data_raises():
    df = pd.DataFrame({
        "country": ["USA", "EU"],
        "year": [2010, 2010],
        "trade_value": [100.0, 100.0],
    })
    sc = SyntheticControl(verbose=False)
    with pytest.raises(ValueError, match="No pre-treatment data"):
        sc.estimate_treatment_effect(
            data=df,
            treated_unit="USA",
            control_units=["EU"],
            treatment_time=2010,
            outcome_var="trade_value",
        )


def test_synthetic_control_placebo_test_returns_dict(minimal_panel: pd.DataFrame):
    sc = SyntheticControl(verbose=False)
    out = sc.placebo_test(
        data=minimal_panel,
        treated_unit="USA",
        control_units=["EU", "Japan"],
        treatment_time=2010,
        outcome_var="trade_value",
        n_placebos=2,
    )
    assert isinstance(out, dict)
    assert "actual_effect" in out
    assert "placebo_effects" in out
    assert "p_value" in out
    assert "significant" in out
    assert isinstance(out["placebo_effects"], list)
    assert 0 <= out["p_value"] <= 1


def test_synthetic_control_weights_sum_to_one(minimal_panel: pd.DataFrame):
    sc = SyntheticControl(verbose=False)
    result = sc.estimate_treatment_effect(
        data=minimal_panel,
        treated_unit="USA",
        control_units=["EU", "Japan"],
        treatment_time=2010,
        outcome_var="trade_value",
    )
    w_sum = sum(result.weights.values())
    assert abs(w_sum - 1.0) < 1e-5


def test_synthetic_control_weights_non_negative(minimal_panel: pd.DataFrame):
    sc = SyntheticControl(verbose=False)
    result = sc.estimate_treatment_effect(
        data=minimal_panel,
        treated_unit="USA",
        control_units=["EU", "Japan"],
        treatment_time=2010,
        outcome_var="trade_value",
    )
    for w in result.weights.values():
        assert w >= -1e-6  # allow tiny numerical error


# ---------------------------------------------------------------------------
# InstrumentalVariable (2SLS)
# ---------------------------------------------------------------------------

@pytest.fixture
def iv_data() -> pd.DataFrame:
    """
    True model: T = 2*Z + noise, Y = -1.5*T + noise.
    OLS of Y ~ T is biased; 2SLS recovers -1.5.
    """
    rng = np.random.default_rng(7)
    n = 400
    Z = rng.normal(0, 1, n)
    T = 2.0 * Z + rng.normal(0, 1, n)
    Y = -1.5 * T + rng.normal(0, 1, n)
    return pd.DataFrame({"outcome": Y, "treatment": T, "instrument": Z})


def test_iv_result_type(iv_data):
    res = InstrumentalVariable().estimate(iv_data, "outcome", "treatment", "instrument")
    assert isinstance(res, IVResult)


def test_iv_estimate_close_to_true(iv_data):
    res = InstrumentalVariable().estimate(iv_data, "outcome", "treatment", "instrument")
    assert abs(res.estimate - (-1.5)) < 0.15, f"2SLS estimate {res.estimate:.3f} too far from -1.5"


def test_iv_se_positive(iv_data):
    res = InstrumentalVariable().estimate(iv_data, "outcome", "treatment", "instrument")
    assert res.se > 0


def test_iv_strong_instrument_not_flagged(iv_data):
    res = InstrumentalVariable().estimate(iv_data, "outcome", "treatment", "instrument")
    assert res.first_stage_f_stat > 10
    assert not res.weak_instrument


def test_iv_ci_contains_true_value(iv_data):
    res = InstrumentalVariable().estimate(iv_data, "outcome", "treatment", "instrument")
    lo, hi = res.confidence_interval
    assert lo < -1.5 < hi, f"CI ({lo:.3f}, {hi:.3f}) does not contain true value -1.5"


def test_iv_weak_instrument_flagged():
    rng = np.random.default_rng(99)
    n = 200
    Z = rng.normal(0, 0.01, n)   # very weak instrument
    T = 0.01 * Z + rng.normal(0, 1, n)
    Y = -T + rng.normal(0, 1, n)
    df = pd.DataFrame({"y": Y, "t": T, "z": Z})
    res = InstrumentalVariable().estimate(df, "y", "t", "z")
    assert res.weak_instrument


def test_iv_with_controls(iv_data):
    iv_data = iv_data.copy()
    iv_data["control"] = np.random.default_rng(1).normal(0, 1, len(iv_data))
    res = InstrumentalVariable().estimate(iv_data, "outcome", "treatment", "instrument", controls=["control"])
    assert isinstance(res, IVResult)
    assert res.n_obs == len(iv_data)


def test_iv_se_accuracy_large_sample():
    """SE should match asymptotic formula: sigma_e / (pi * sqrt(n)).
    Regression test for bug where T_hat residuals inflated sigma^2 ~1.6x,
    causing SE overestimation of ~1.26x."""
    rng = np.random.default_rng(42)
    n = 10_000
    Z = rng.normal(0, 1, n)
    T = 2.0 * Z + rng.normal(0, 0.5, n)   # pi = 2.0 (first stage slope)
    Y = -1.5 * T + rng.normal(0, 1, n)    # sigma_e = 1.0
    df = pd.DataFrame({"y": Y, "t": T, "z": Z})
    res = InstrumentalVariable().estimate(df, "y", "t", "z")
    # Asymptotic SE = sigma_e / (pi * sqrt(n)) = 1.0 / (2.0 * 100) = 0.005
    true_se = 1.0 / (2.0 * np.sqrt(n))
    assert abs(res.se - true_se) < 0.001, (
        f"SE {res.se:.5f} too far from true {true_se:.5f}; "
        "likely using T_hat instead of T for residuals"
    )


def test_iv_f_stat_not_inflated_by_strong_controls():
    """Partial F-stat for the instrument must not be inflated by unrelated controls.
    Regression test for bug where overall model F was used instead of partial F,
    masking weak instruments when controls are strong predictors of treatment."""
    rng = np.random.default_rng(7)
    n = 300
    Z = rng.normal(0, 0.01, n)            # near-useless instrument
    ctrl = rng.normal(0, 1, n)
    T = 0.01 * Z + 2.0 * ctrl + rng.normal(0, 0.3, n)   # ctrl drives T, Z does not
    Y = -T + rng.normal(0, 1, n)
    df = pd.DataFrame({"y": Y, "t": T, "z": Z, "ctrl": ctrl})
    res = InstrumentalVariable().estimate(df, "y", "t", "z", controls=["ctrl"])
    # Partial F for the instrument should be small (<10), not the overall model F (~12000)
    assert res.weak_instrument, (
        f"F-stat={res.first_stage_f_stat:.1f} should flag weak instrument; "
        "likely computing overall model F instead of partial F"
    )
    assert res.first_stage_f_stat < 10, (
        f"Partial F-stat {res.first_stage_f_stat:.1f} should be < 10 for near-useless instrument"
    )


# ---------------------------------------------------------------------------
# RegressionDiscontinuity (local linear)
# ---------------------------------------------------------------------------

@pytest.fixture
def rd_data() -> pd.DataFrame:
    """Sharp RD: jump of 3.0 at threshold=0."""
    rng = np.random.default_rng(13)
    n = 300
    x = np.linspace(-1.5, 1.5, n)
    y = 3.0 * (x >= 0).astype(float) + 0.5 * x + rng.normal(0, 0.4, n)
    return pd.DataFrame({"running": x, "outcome": y})


def test_rd_result_type(rd_data):
    res = RegressionDiscontinuity().estimate(rd_data, "running", "outcome", threshold=0.0)
    assert isinstance(res, RDResult)


def test_rd_discontinuity_close_to_true(rd_data):
    res = RegressionDiscontinuity().estimate(rd_data, "running", "outcome", threshold=0.0, bandwidth=1.0)
    assert abs(res.discontinuity - 3.0) < 0.3, f"RD={res.discontinuity:.3f} too far from 3.0"


def test_rd_se_positive(rd_data):
    res = RegressionDiscontinuity().estimate(rd_data, "running", "outcome", threshold=0.0)
    assert res.se > 0


def test_rd_n_left_right_positive(rd_data):
    res = RegressionDiscontinuity().estimate(rd_data, "running", "outcome", threshold=0.0, bandwidth=1.0)
    assert res.n_left > 0
    assert res.n_right > 0


def test_rd_ci_contains_true_value(rd_data):
    res = RegressionDiscontinuity().estimate(rd_data, "running", "outcome", threshold=0.0, bandwidth=1.0)
    lo, hi = res.confidence_interval
    assert lo < 3.0 < hi, f"CI ({lo:.3f}, {hi:.3f}) does not contain true value 3.0"


def test_rd_default_bandwidth_uses_std(rd_data):
    res = RegressionDiscontinuity().estimate(rd_data, "running", "outcome", threshold=0.0)
    expected_bw = float(np.std(rd_data["running"].values))  # numpy std (ddof=0), matches implementation
    assert abs(res.bandwidth - expected_bw) < 1e-6


def test_rd_too_narrow_bandwidth_raises(rd_data):
    with pytest.raises(ValueError, match="Too few observations"):
        RegressionDiscontinuity().estimate(rd_data, "running", "outcome", threshold=0.0, bandwidth=0.001)


# ---------------------------------------------------------------------------
# DifferenceInDifferences
# ---------------------------------------------------------------------------

@pytest.fixture
def did_data() -> pd.DataFrame:
    """Panel with treatment effect of -25 for 'Treated' unit after 2010."""
    rng = np.random.default_rng(21)
    rows = []
    for unit in ["Treated", "Control_A", "Control_B", "Control_C"]:
        for yr in range(2000, 2020):
            base = 200.0 + (yr - 2000) * 1.5 + rng.normal(0, 4.0)
            if unit == "Treated" and yr >= 2010:
                base -= 25.0
            rows.append({"unit": unit, "year": yr, "outcome": base,
                         "treated": 1 if unit == "Treated" else 0})
    return pd.DataFrame(rows)


def test_did_result_type(did_data):
    res = DifferenceInDifferences().estimate(did_data, "outcome", "treated", "year", 2010)
    assert isinstance(res, DIDResult)


def test_did_att_close_to_true(did_data):
    res = DifferenceInDifferences().estimate(did_data, "outcome", "treated", "year", 2010)
    assert abs(res.att - (-25.0)) < 4.0, f"DiD ATT {res.att:.2f} too far from -25.0"


def test_did_att_negative(did_data):
    res = DifferenceInDifferences().estimate(did_data, "outcome", "treated", "year", 2010)
    assert res.att < 0


def test_did_se_positive(did_data):
    res = DifferenceInDifferences().estimate(did_data, "outcome", "treated", "year", 2010)
    assert res.se > 0


def test_did_parallel_pre_trends_passes(did_data):
    res = DifferenceInDifferences().estimate(did_data, "outcome", "treated", "year", 2010)
    assert res.pre_trend_pvalue > 0.05, f"Pre-trend p={res.pre_trend_pvalue:.3f} rejects parallel trends"


def test_did_ci_contains_true_value(did_data):
    res = DifferenceInDifferences().estimate(did_data, "outcome", "treated", "year", 2010)
    lo, hi = res.confidence_interval
    assert lo < -25.0 < hi, f"CI ({lo:.2f}, {hi:.2f}) does not contain true value -25"


def test_did_unit_counts(did_data):
    res = DifferenceInDifferences().estimate(
        did_data, "outcome", "treated", "year", 2010, unit_col="unit"
    )
    assert res.n_treated == 1
    assert res.n_control == 3


def test_did_post_treatment_years(did_data):
    res = DifferenceInDifferences().estimate(did_data, "outcome", "treated", "year", 2010)
    assert 2010 in res.post_treatment_years
    assert all(yr >= 2010 for yr in res.post_treatment_years)


def test_did_placebo_test_returns_dict(did_data):
    out = DifferenceInDifferences().placebo_test(
        did_data, "outcome", "treated", "year", 2010, n_placebos=3, unit_col="unit"
    )
    assert "actual_att" in out
    assert "placebo_atts" in out
    assert "p_value" in out
    assert 0.0 <= out["p_value"] <= 1.0


# ---------------------------------------------------------------------------
# GraphiteSupplyChainDAG.estimate_parameter dispatch
# ---------------------------------------------------------------------------

def test_estimate_parameter_eta_d_dispatches():
    from src.minerals.causal_inference import GraphiteSupplyChainDAG
    rng = np.random.default_rng(5)
    n = 200
    Z = rng.normal(0, 1, n)
    T = 2 * Z + rng.normal(0, 1, n)
    Y = -1.0 * T + rng.normal(0, 1, n)
    df = pd.DataFrame({"Demand": Y, "Price": T, "supply_shock": Z})
    dag = GraphiteSupplyChainDAG()
    res = dag.estimate_parameter("eta_D", df, instrument_var="supply_shock")
    assert hasattr(res, "estimate")
    assert hasattr(res, "first_stage_f_stat")


def test_estimate_parameter_policy_shock_dispatches():
    from src.minerals.causal_inference import GraphiteSupplyChainDAG
    rng = np.random.default_rng(6)
    rows = []
    for u in ["T", "C"]:
        for yr in range(2000, 2015):
            rows.append({"unit": u, "year": yr,
                         "Supply": 100 + rng.normal(0, 3) + (-10 if u == "T" and yr >= 2010 else 0),
                         "treated": 1 if u == "T" else 0})
    df = pd.DataFrame(rows)
    dag = GraphiteSupplyChainDAG()
    res = dag.estimate_parameter("policy_shock_magnitude", df, treatment_time=2010, outcome_var="Supply")
    assert hasattr(res, "att")


def test_estimate_parameter_unknown_raises():
    from src.minerals.causal_inference import GraphiteSupplyChainDAG
    dag = GraphiteSupplyChainDAG()
    with pytest.raises(ValueError, match="Unknown parameter"):
        dag.estimate_parameter("not_a_param", pd.DataFrame())
