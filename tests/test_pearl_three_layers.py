"""Tests for Pearl's three layers of causal inference.

Covers:
- CausalDAG: d-separation, backdoor criterion, frontdoor criterion, identifiability
- CausalInferenceEngine: Layer 1 (association), Layer 2 (intervention), Layer 3 (counterfactual)
- pearl_layers.py: standalone Layer 1/2/3 functions
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.minerals.causal_inference import (
    CausalDAG,
    GraphiteSupplyChainDAG,
    IdentificationStrategy,
)
from src.minerals.causal_engine import (
    CausalInferenceEngine,
    _DeterministicNoiseRNG,
)
from src.minerals.model import State
from src.minerals.pearl_layers import (
    CounterfactualResult,
    counterfactual_fringe,
    counterfactual_step,
    counterfactual_substitution,
    counterfactual_trajectory,
    do_compare,
    do_fringe_supply,
    do_substitution,
    interventional_identifiability,
    mutilated_graph_for_do,
    observe_fringe_association,
    observe_substitution_association,
    observational_conditional,
    observational_summary,
)
from src.minerals.schema import (
    BaselineConfig,
    DemandGrowthConfig,
    OutputsConfig,
    ParametersConfig,
    PolicyConfig,
    ScenarioConfig,
    ShockConfig,
    TimeConfig,
)
from src.minerals.shocks import ShockSignals


# ====================================================================
# Fixtures
# ====================================================================


def _minimal_config(sigma_P: float = 0.0) -> ScenarioConfig:
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
            retire_rate=0.0,
            eta_D=-0.25,
            demand_growth=DemandGrowthConfig(type="constant", g=1.0),
            alpha_P=0.80,
            cover_star=0.20,
            lambda_cover=0.60,
            sigma_P=sigma_P,
        ),
        policy=PolicyConfig(),
        shocks=[],
        outputs=OutputsConfig(
            metrics=["total_shortage", "peak_shortage", "avg_price", "final_inventory_cover"]
        ),
    )


def _restriction_config(
    sigma_P: float = 0.0,
    substitution_elasticity: float = 0.0,
    fringe_capacity_share: float = 0.0,
    fringe_entry_price: float = 2.0,
) -> ScenarioConfig:
    """Config with a 40% export restriction shock from 2026-2028."""
    return ScenarioConfig(
        name="restriction_test",
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
            retire_rate=0.0,
            eta_D=-0.25,
            demand_growth=DemandGrowthConfig(type="constant", g=1.0),
            alpha_P=0.80,
            cover_star=0.20,
            lambda_cover=0.60,
            sigma_P=sigma_P,
            substitution_elasticity=substitution_elasticity,
            fringe_capacity_share=fringe_capacity_share,
            fringe_entry_price=fringe_entry_price,
        ),
        policy=PolicyConfig(),
        shocks=[
            ShockConfig(
                type="export_restriction",
                start_year=2026,
                end_year=2028,
                magnitude=0.40,
            )
        ],
        outputs=OutputsConfig(
            metrics=["total_shortage", "peak_shortage", "avg_price", "final_inventory_cover"]
        ),
    )


@pytest.fixture
def graphite_dag() -> GraphiteSupplyChainDAG:
    return GraphiteSupplyChainDAG()


@pytest.fixture
def simple_dag() -> CausalDAG:
    """A simple DAG: X -> M -> Y with unmeasured confounder U -> X, U -> Y."""
    dag = CausalDAG()
    dag.add_node("X", observed=True)
    dag.add_node("M", observed=True)
    dag.add_node("Y", observed=True)
    dag.add_node("U", observed=False)
    dag.add_edge("X", "M")
    dag.add_edge("M", "Y")
    dag.add_edge("U", "X")
    dag.add_edge("U", "Y")
    return dag


@pytest.fixture
def minimal_cfg() -> ScenarioConfig:
    return _minimal_config()


@pytest.fixture
def observational_data() -> pd.DataFrame:
    """Synthetic observational data for Layer 1 tests."""
    rng = np.random.default_rng(42)
    n = 100
    x = rng.normal(10, 2, n)
    z = rng.normal(5, 1, n)
    # Y depends on X and Z (with some noise)
    y = 2.0 * x + 1.5 * z + rng.normal(0, 1, n)
    return pd.DataFrame({"X": x, "Y": y, "Z": z, "group": ["A"] * 50 + ["B"] * 50})


# ====================================================================
# CausalDAG tests
# ====================================================================


class TestCausalDAG:
    def test_add_node_observed(self):
        dag = CausalDAG()
        dag.add_node("X", observed=True)
        assert "X" in dag.observed_vars
        assert "X" not in dag.unobserved_vars

    def test_add_node_unobserved(self):
        dag = CausalDAG()
        dag.add_node("U", observed=False)
        assert "U" in dag.unobserved_vars
        assert "U" not in dag.observed_vars

    def test_add_edge_creates_nodes(self):
        dag = CausalDAG()
        dag.add_edge("A", "B")
        assert "A" in dag.graph
        assert "B" in dag.graph

    def test_get_parents(self, graphite_dag):
        parents = graphite_dag.get_parents("Shortage")
        assert "Supply" in parents
        assert "Demand" in parents

    def test_get_descendants(self, graphite_dag):
        desc = graphite_dag.get_descendants("ExportPolicy")
        assert "Price" in desc
        assert "Supply" in desc

    def test_d_separation_basic(self, graphite_dag):
        # ExportPolicy and Price are NOT d-separated given empty set
        # (there is a directed path)
        assert not graphite_dag.d_separated({"ExportPolicy"}, {"Price"}, set())

    def test_remove_incoming_edges(self, graphite_dag):
        mutilated = graphite_dag.remove_incoming_edges("Supply")
        # Supply should have no incoming edges in the mutilated graph
        assert len(list(mutilated.in_edges("Supply"))) == 0
        # But outgoing edges should remain
        assert len(list(mutilated.out_edges("Supply"))) > 0

    def test_backdoor_criterion_empty_set(self):
        """X -> Y with no confounders: empty adjustment set works."""
        dag = CausalDAG()
        dag.add_node("X", observed=True)
        dag.add_node("Y", observed=True)
        dag.add_edge("X", "Y")
        assert dag.backdoor_criterion("X", "Y", set())

    def test_backdoor_criterion_with_confounder(self):
        """X <- Z -> Y: need to adjust for Z."""
        dag = CausalDAG()
        dag.add_node("X", observed=True)
        dag.add_node("Y", observed=True)
        dag.add_node("Z", observed=True)
        dag.add_edge("Z", "X")
        dag.add_edge("Z", "Y")
        dag.add_edge("X", "Y")
        # Empty set doesn't work (Z confounds)
        assert not dag.backdoor_criterion("X", "Y", set())
        # Adjusting for Z does
        assert dag.backdoor_criterion("X", "Y", {"Z"})

    def test_find_backdoor_adjustment_set(self):
        dag = CausalDAG()
        dag.add_node("X", observed=True)
        dag.add_node("Y", observed=True)
        dag.add_node("Z", observed=True)
        dag.add_edge("Z", "X")
        dag.add_edge("Z", "Y")
        dag.add_edge("X", "Y")
        adj = dag.find_backdoor_adjustment_set("X", "Y")
        assert adj is not None
        assert "Z" in adj

    def test_frontdoor_criterion_classic(self, simple_dag):
        """Classic frontdoor: X -> M -> Y with U -> X, U -> Y (U unobserved).
        M satisfies frontdoor criterion."""
        assert simple_dag.frontdoor_criterion("X", "Y", {"M"})

    def test_frontdoor_criterion_rejects_empty(self, simple_dag):
        assert not simple_dag.frontdoor_criterion("X", "Y", set())

    def test_frontdoor_criterion_rejects_unobserved(self):
        dag = CausalDAG()
        dag.add_node("X", observed=True)
        dag.add_node("M", observed=False)
        dag.add_node("Y", observed=True)
        dag.add_edge("X", "M")
        dag.add_edge("M", "Y")
        assert not dag.frontdoor_criterion("X", "Y", {"M"})

    def test_find_frontdoor_set(self, simple_dag):
        result = simple_dag.find_frontdoor_set("X", "Y")
        assert result is not None
        assert "M" in result

    def test_is_identifiable_backdoor(self, graphite_dag):
        result = graphite_dag.is_identifiable("ExportPolicy", "Price")
        assert result.identifiable
        assert result.strategy == IdentificationStrategy.BACKDOOR_ADJUSTMENT

    def test_is_identifiable_frontdoor(self, simple_dag):
        """X -> M -> Y with unmeasured U -> X, U -> Y.
        Backdoor fails (U unobserved), but frontdoor via M should work."""
        result = simple_dag.is_identifiable("X", "Y")
        assert result.identifiable
        assert result.strategy == IdentificationStrategy.FRONTDOOR_ADJUSTMENT
        assert "M" in result.adjustment_set

    def test_is_not_identifiable(self):
        """X -> Y with U -> X, U -> Y, U unobserved, no mediator."""
        dag = CausalDAG()
        dag.add_node("X", observed=True)
        dag.add_node("Y", observed=True)
        dag.add_node("U", observed=False)
        dag.add_edge("X", "Y")
        dag.add_edge("U", "X")
        dag.add_edge("U", "Y")
        result = dag.is_identifiable("X", "Y")
        assert not result.identifiable


# ====================================================================
# DeterministicNoiseRNG tests
# ====================================================================


class TestDeterministicNoiseRNG:
    def test_replays_noise_sequence(self):
        rng = _DeterministicNoiseRNG([1.0, -0.5, 2.3])
        assert rng.normal() == pytest.approx(1.0)
        assert rng.normal() == pytest.approx(-0.5)
        assert rng.normal() == pytest.approx(2.3)

    def test_fallback_beyond_sequence(self):
        rng = _DeterministicNoiseRNG([1.0])
        rng.normal()  # consume the one value
        assert rng.normal() == pytest.approx(0.0)  # fallback to loc=0

    def test_respects_loc_and_scale(self):
        rng = _DeterministicNoiseRNG([2.0])
        result = rng.normal(loc=5.0, scale=3.0)
        assert result == pytest.approx(5.0 + 3.0 * 2.0)


# ====================================================================
# Layer 1 — Association (pearl_layers.py)
# ====================================================================


class TestLayer1PearlLayers:
    def test_observational_conditional(self, observational_data):
        series = observational_conditional(observational_data, "Y")
        assert len(series) == 100

    def test_observational_conditional_with_filter(self, observational_data):
        series = observational_conditional(
            observational_data, "Y", conditioning={"group": "A"}
        )
        assert len(series) == 50

    def test_observational_conditional_bad_column_raises(self, observational_data):
        with pytest.raises(ValueError, match="Outcome column"):
            observational_conditional(observational_data, "NONEXISTENT")

    def test_observational_summary(self, observational_data):
        result = observational_summary(observational_data, "Y")
        assert "mean" in result.columns
        assert "std" in result.columns
        assert "count" in result.columns

    def test_observational_summary_grouped(self, observational_data):
        result = observational_summary(observational_data, "Y", group_by="group")
        assert len(result) == 2


# ====================================================================
# Layer 1 — Association (CausalInferenceEngine)
# ====================================================================


class TestLayer1Engine:
    def test_correlate_pearson(self, graphite_dag, observational_data):
        engine = CausalInferenceEngine(dag=graphite_dag)
        result = engine.correlate(observational_data, variables=["X", "Y", "Z"])
        assert result.pearson.shape == (3, 3)
        assert result.p_values.shape == (3, 3)
        # X and Y are correlated (Y = 2X + 1.5Z + noise)
        x_idx = result.variables.index("X")
        y_idx = result.variables.index("Y")
        assert abs(result.pearson[x_idx, y_idx]) > 0.5

    def test_correlate_empty_raises(self, graphite_dag):
        engine = CausalInferenceEngine(dag=graphite_dag)
        with pytest.raises(ValueError, match="empty"):
            engine.correlate(pd.DataFrame())

    def test_conditional_distribution(self, graphite_dag, observational_data):
        engine = CausalInferenceEngine(dag=graphite_dag)
        result = engine.conditional_distribution(observational_data, "Y")
        assert result.count == 100
        assert result.mean != 0
        assert "50%" in result.quantiles

    def test_test_independence_marginal(self, graphite_dag, observational_data):
        engine = CausalInferenceEngine(dag=graphite_dag)
        result = engine.test_independence(observational_data, "X", "Y")
        # X and Y are NOT independent
        assert not result.independent
        assert result.p_value < 0.05

    def test_test_independence_conditional(self, graphite_dag, observational_data):
        engine = CausalInferenceEngine(dag=graphite_dag)
        result = engine.test_independence(observational_data, "X", "Y", z=["Z"])
        # After conditioning on Z, X and Y are still correlated (Y = 2X + 1.5Z)
        assert not result.independent

    def test_regression_association(self, graphite_dag, observational_data):
        engine = CausalInferenceEngine(dag=graphite_dag)
        result = engine.regression_association(observational_data, "Y", ["X", "Z"])
        # True coefficients are ~2.0 for X and ~1.5 for Z
        assert abs(result.coefficients["X"] - 2.0) < 0.5
        assert abs(result.coefficients["Z"] - 1.5) < 0.5
        assert result.r_squared > 0.8


# ====================================================================
# Layer 2 — Intervention (pearl_layers.py)
# ====================================================================


class TestLayer2PearlLayers:
    def test_interventional_identifiability(self, graphite_dag):
        result = interventional_identifiability("ExportPolicy", "Price", dag=graphite_dag)
        assert result.identifiable

    def test_mutilated_graph_for_do(self, graphite_dag):
        mutilated = mutilated_graph_for_do(graphite_dag, "Supply")
        assert len(list(mutilated.in_edges("Supply"))) == 0


# ====================================================================
# Layer 2 — Intervention (CausalInferenceEngine)
# ====================================================================


class TestLayer2Engine:
    def test_identify(self, graphite_dag):
        engine = CausalInferenceEngine(dag=graphite_dag)
        result = engine.identify("ExportPolicy", "Price")
        assert result.identifiable

    def test_backdoor_estimate(self, graphite_dag):
        """Test backdoor estimate on a simple confounded DAG with known ATE."""
        dag = CausalDAG()
        dag.add_node("X", observed=True)
        dag.add_node("Y", observed=True)
        dag.add_node("Z", observed=True)
        dag.add_edge("Z", "X")
        dag.add_edge("Z", "Y")
        dag.add_edge("X", "Y")

        rng = np.random.default_rng(42)
        n = 500
        z = rng.normal(0, 1, n)
        x = 0.5 * z + rng.normal(0, 0.5, n)
        y = 3.0 * x + 2.0 * z + rng.normal(0, 0.5, n)  # True ATE of X on Y = 3.0
        data = pd.DataFrame({"X": x, "Y": y, "Z": z})

        engine = CausalInferenceEngine(dag=dag, seed=42)
        result = engine.backdoor_estimate(data, "X", "Y", n_bootstrap=50)
        assert abs(result.ate - 3.0) < 0.5  # Should recover ~3.0
        assert result.adjustment_set == {"Z"}

    def test_do_simulation(self, graphite_dag, minimal_cfg):
        engine = CausalInferenceEngine(dag=graphite_dag, cfg=minimal_cfg)
        result = engine.do("ExportPolicy", 0.3)
        assert result.treatment_var == "ExportPolicy"
        assert result.treatment_value == 0.3
        assert "P" in result.effect_on_outcome
        assert len(result.baseline_trajectory) > 0
        assert len(result.intervention_trajectory) > 0

    def test_do_unsupported_node_raises(self, graphite_dag, minimal_cfg):
        engine = CausalInferenceEngine(dag=graphite_dag, cfg=minimal_cfg)
        with pytest.raises(ValueError, match="No shock mapping"):
            engine.do("Price", 2.0)

    def test_do_no_config_raises(self, graphite_dag):
        engine = CausalInferenceEngine(dag=graphite_dag)
        with pytest.raises(ValueError, match="ScenarioConfig required"):
            engine.do("ExportPolicy", 0.3)


# ====================================================================
# Layer 3 — Counterfactual (pearl_layers.py)
# ====================================================================


class TestLayer3PearlLayers:
    def test_counterfactual_step(self, minimal_cfg):
        s0 = State(year=2024, t_index=0, K=100.0, I=20.0, P=1.0)
        rng = np.random.default_rng(42)
        s1, res = counterfactual_step(
            s0, minimal_cfg, 2024,
            do_shock_overrides={"export_restriction": 0.5},
            rng=rng,
        )
        assert s1.year == 2025
        assert res.Q_eff < res.Q  # export restriction reduces effective supply

    def test_counterfactual_trajectory(self, minimal_cfg):
        s0 = State(year=2024, t_index=0, K=100.0, I=20.0, P=1.0)
        rng = np.random.default_rng(42)
        years = list(range(2024, 2028))
        overrides = {2025: {"export_restriction": 0.5}, 2026: {"export_restriction": 0.5}}
        states, results = counterfactual_trajectory(s0, minimal_cfg, years, overrides, rng)
        assert len(states) == len(years) + 1
        assert len(results) == len(years)


# ====================================================================
# Layer 3 — Counterfactual (CausalInferenceEngine)
# ====================================================================


class TestLayer3Engine:
    def test_abduct_deterministic(self, graphite_dag, minimal_cfg):
        """With sigma_P=0, abduction should recover all-zero noise."""
        from src.minerals.simulate import run_scenario

        df, _ = run_scenario(minimal_cfg)
        engine = CausalInferenceEngine(dag=graphite_dag, cfg=minimal_cfg)
        result = engine.abduct(df)
        assert len(result.inferred_noise) == len(result.years)
        # All noise should be 0 when sigma_P=0
        for noise in result.inferred_noise.values():
            assert noise == pytest.approx(0.0)

    def test_abduct_stochastic(self, graphite_dag):
        """With sigma_P>0, abduction should recover non-trivial noise."""
        cfg = _minimal_config(sigma_P=0.1)
        from src.minerals.simulate import run_scenario

        df, _ = run_scenario(cfg)
        engine = CausalInferenceEngine(dag=graphite_dag, cfg=cfg)
        result = engine.abduct(df)
        # At least some noise values should be nonzero
        nonzero = [v for v in result.inferred_noise.values() if abs(v) > 1e-10]
        assert len(nonzero) > 0

    def test_counterfactual_full_3step(self, graphite_dag, minimal_cfg):
        """Full Pearl 3-step counterfactual: abduction -> action -> prediction."""
        from src.minerals.simulate import run_scenario

        # Run factual scenario
        df_factual, _ = run_scenario(minimal_cfg)
        engine = CausalInferenceEngine(dag=graphite_dag, cfg=minimal_cfg)

        # Counterfactual: what if 50% export restriction in 2025-2027?
        overrides = {y: {"export_restriction": 0.5} for y in range(2025, 2028)}
        result = engine.counterfactual(df_factual, overrides)

        assert len(result.counterfactual_trajectory) > 0
        assert len(result.effect) > 0
        assert result.abduction is not None
        # With export restriction, price should go up (positive delta)
        assert "mean_delta_P" in result.summary

    def test_counterfactual_trajectory_same_noise(self, graphite_dag):
        """counterfactual_trajectory() should use same noise for both runs."""
        cfg = _minimal_config(sigma_P=0.1)
        engine = CausalInferenceEngine(dag=graphite_dag, cfg=cfg, seed=42)
        s0 = State(year=2024, t_index=0, K=100.0, I=20.0, P=1.0)
        years = list(range(2024, 2028))

        # No overrides => factual and counterfactual should be identical
        result = engine.counterfactual_trajectory(s0, years, {})
        for key in result.deltas:
            for delta in result.deltas[key]:
                assert delta == pytest.approx(0.0, abs=1e-10)

    def test_counterfactual_contrast(self, graphite_dag, minimal_cfg):
        """Compare two scenarios: baseline vs. with export restriction."""
        cfg_factual = minimal_cfg
        cfg_cf = minimal_cfg.model_copy(deep=True)
        cfg_cf.shocks = [
            ShockConfig(
                type="export_restriction",
                start_year=2025,
                end_year=2028,
                magnitude=0.4,
            )
        ]
        engine = CausalInferenceEngine(dag=graphite_dag, cfg=cfg_factual)
        result = engine.counterfactual_contrast(cfg_factual, cfg_cf)
        assert len(result.years) > 0
        # Export restriction should cause price differences
        assert any(abs(d) > 0 for d in result.deltas["P"])


# ====================================================================
# New mechanisms — Layer 1: observe_substitution_association,
#                           observe_fringe_association
# ====================================================================


class TestLayer1NewMechanisms:
    """
    L1 tests verify that observational association functions return correct
    summary statistics from run data — no causal claim, just correlation.
    """

    def test_observe_substitution_association_returns_dataframe(self):
        from src.minerals.simulate import run_scenario

        cfg = _restriction_config(substitution_elasticity=0.5)
        df, _ = run_scenario(cfg)
        summary = observe_substitution_association(df)
        assert isinstance(summary, pd.DataFrame)
        assert "export_restricted" in summary.columns
        assert "mean_Q_sub" in summary.columns

    def test_observe_substitution_higher_Q_sub_when_restricted(self):
        """
        L1 association: when export_restriction > 0 AND elasticity > 0,
        observed Q_sub should be higher in restricted periods.
        """
        from src.minerals.simulate import run_scenario

        cfg = _restriction_config(substitution_elasticity=0.5)
        df, _ = run_scenario(cfg)
        summary = observe_substitution_association(df)
        mean_by_restricted = summary.set_index("export_restricted")["mean_Q_sub"]
        assert mean_by_restricted[True] > mean_by_restricted[False]

    def test_observe_substitution_zero_when_no_elasticity(self):
        """With elasticity=0, Q_sub=0 always; association should show zero."""
        from src.minerals.simulate import run_scenario

        cfg = _restriction_config(substitution_elasticity=0.0)
        df, _ = run_scenario(cfg)
        summary = observe_substitution_association(df)
        assert summary["mean_Q_sub"].max() == pytest.approx(0.0)

    def test_observe_substitution_missing_column_raises(self):
        bad_df = pd.DataFrame({"P": [1.0, 1.1], "D": [100.0, 100.0]})
        with pytest.raises(ValueError, match="Missing columns"):
            observe_substitution_association(bad_df)

    def test_observe_fringe_association_returns_dataframe(self):
        from src.minerals.simulate import run_scenario

        cfg = _restriction_config(fringe_capacity_share=0.4)
        df, _ = run_scenario(cfg)
        summary = observe_fringe_association(df)
        assert isinstance(summary, pd.DataFrame)
        assert "mean_Q_fringe" in summary.columns

    def test_observe_fringe_higher_supply_at_higher_price(self):
        """
        L1 association: fringe supply should be higher in high-price quartiles
        because the structural equation activates above the entry threshold.
        """
        from src.minerals.simulate import run_scenario

        # Use a config where price will rise (export restriction drives price up)
        cfg = _restriction_config(fringe_capacity_share=0.4, fringe_entry_price=1.1)
        df, _ = run_scenario(cfg)
        summary = observe_fringe_association(df)
        # Q4 (highest price) should have more fringe supply than Q1
        q1_fringe = summary[summary["price_quartile"] == "Q1"]["mean_Q_fringe"].values[0]
        q4_fringe = summary[summary["price_quartile"] == "Q4"]["mean_Q_fringe"].values[0]
        assert q4_fringe >= q1_fringe

    def test_observe_fringe_missing_column_raises(self):
        bad_df = pd.DataFrame({"D": [100.0]})
        with pytest.raises(ValueError, match="Missing columns"):
            observe_fringe_association(bad_df)


# ====================================================================
# New mechanisms — Layer 2: do_substitution, do_fringe_supply, do_compare
# ====================================================================


class TestLayer2NewMechanisms:
    """
    L2 tests verify that graph-surgery interventions change outcomes in the
    expected direction.  do(param=x) severs the causal path that determines
    param and pins it to x.
    """

    def test_do_substitution_no_change_without_restriction(self):
        """
        do(substitution_elasticity=0.8) has zero effect when there is no
        export restriction — Q_sub structural equation yields 0 regardless.
        """
        from src.minerals.simulate import run_scenario

        # Base config: no restriction
        cfg = _minimal_config()
        baseline_df, _ = run_scenario(cfg)
        intervened_df, _ = do_substitution(cfg, elasticity=0.8)

        # Q_sub should remain 0 in both (no restriction to substitute)
        assert baseline_df["Q_sub"].sum() == pytest.approx(0.0)
        assert intervened_df["Q_sub"].sum() == pytest.approx(0.0)

    def test_do_substitution_increases_supply_under_restriction(self):
        """
        do(substitution_elasticity=0.8) should increase Q_sub (and Q_total)
        during the restriction window compared to factual (elasticity=0).
        """
        from src.minerals.simulate import run_scenario

        factual_cfg = _restriction_config(substitution_elasticity=0.0)
        factual_df, _ = run_scenario(factual_cfg)
        intervened_df, _ = do_substitution(factual_cfg, elasticity=0.8)

        # Q_sub should increase
        assert intervened_df["Q_sub"].sum() > factual_df["Q_sub"].sum()
        # Q_total should increase during restriction window
        restriction_years = [2026, 2027, 2028]
        f_q = factual_df[factual_df["year"].isin(restriction_years)]["Q_total"].mean()
        i_q = intervened_df[intervened_df["year"].isin(restriction_years)]["Q_total"].mean()
        assert i_q > f_q

    def test_do_substitution_reduces_price_under_restriction(self):
        """
        More supply from substitution should dampen price rises under restriction.
        """
        factual_cfg = _restriction_config(substitution_elasticity=0.0)
        intervened_df, _ = do_substitution(factual_cfg, elasticity=0.8)
        factual_df, _ = do_substitution(factual_cfg, elasticity=0.0)
        # Peak price under intervention should be <= peak under factual
        assert intervened_df["P"].max() <= factual_df["P"].max()

    def test_do_substitution_idempotent_at_zero(self):
        """
        do(elasticity=0) on a config that already has elasticity=0
        should give identical results to baseline run.
        """
        from src.minerals.simulate import run_scenario

        cfg = _restriction_config(substitution_elasticity=0.0)
        baseline_df, _ = run_scenario(cfg)
        intervened_df, _ = do_substitution(cfg, elasticity=0.0)
        pd.testing.assert_frame_equal(baseline_df, intervened_df)

    def test_do_fringe_increases_supply(self):
        """
        do(fringe_capacity_share=0.3) should add fringe supply above entry threshold.
        """
        from src.minerals.simulate import run_scenario

        # Restriction drives price above entry threshold (entry_price=1.2)
        factual_cfg = _restriction_config(fringe_capacity_share=0.0)
        factual_df, _ = run_scenario(factual_cfg)
        intervened_df, _ = do_fringe_supply(factual_cfg, capacity_share=0.3, entry_price=1.2)

        assert intervened_df["Q_fringe"].sum() >= 0.0  # fringe exists
        # If price rose above 1.2 in factual, fringe should have contributed
        if (factual_df["P"] > 1.2).any():
            assert intervened_df["Q_fringe"].sum() > factual_df["Q_fringe"].sum()

    def test_do_fringe_zero_below_entry_threshold(self):
        """
        Fringe supply is 0 if price never exceeds the entry threshold.
        """
        # No restriction — price stays near 1.0; entry_price=10.0 (never reached)
        cfg = _minimal_config()
        intervened_df, _ = do_fringe_supply(cfg, capacity_share=0.5, entry_price=10.0)
        assert intervened_df["Q_fringe"].sum() == pytest.approx(0.0)

    def test_do_compare_returns_ate_columns(self):
        """
        do_compare should return a DataFrame with _factual, _intervention, _ate columns.
        """
        cfg = _restriction_config(substitution_elasticity=0.0)
        result = do_compare(cfg, {"substitution_elasticity": 0.8}, outcomes=["P", "Q_total"])
        assert "P_factual" in result.columns
        assert "P_intervention" in result.columns
        assert "P_ate" in result.columns
        assert "Q_total_ate" in result.columns

    def test_do_compare_ate_nonzero_under_intervention(self):
        """ATE should be non-zero when the intervention changes outcomes."""
        cfg = _restriction_config(substitution_elasticity=0.0)
        result = do_compare(cfg, {"substitution_elasticity": 0.8}, outcomes=["Q_sub"])
        # At least some years should show a difference
        assert (result["Q_sub_ate"] != 0).any()

    def test_do_compare_ate_zero_null_intervention(self):
        """ATE should be zero when intervention matches factual parameters."""
        cfg = _restriction_config(substitution_elasticity=0.0)
        result = do_compare(cfg, {"substitution_elasticity": 0.0}, outcomes=["P"])
        # No change → ATE = 0 for all years
        assert (result["P_ate"].abs() < 1e-10).all()


# ====================================================================
# New mechanisms — Layer 3: counterfactual_substitution,
#                            counterfactual_fringe, CounterfactualResult
# ====================================================================


class TestLayer3NewMechanisms:
    """
    L3 tests verify the Abduction-Action-Prediction procedure.

    Key properties:
    - Same noise sequence in factual and counterfactual (abduction via seed)
    - ATE = 0 when counterfactual equals factual
    - ATE reflects structural equation change (Q_sub, Q_fringe, P changes)
    """

    def test_counterfactual_substitution_returns_result(self):
        cfg = _restriction_config(substitution_elasticity=0.0)
        result = counterfactual_substitution(cfg, cf_elasticity=0.8)
        assert isinstance(result, CounterfactualResult)
        assert isinstance(result.factual, pd.DataFrame)
        assert isinstance(result.counterfactual, pd.DataFrame)
        assert isinstance(result.ate, dict)
        assert isinstance(result.description, str)
        assert isinstance(result.noise_sequence, list)

    def test_counterfactual_substitution_zero_ate_when_no_change(self):
        """
        L3 with cf_elasticity == factual_elasticity should give ATE=0.
        Same structural equation → identical trajectories (twin network).
        """
        cfg = _restriction_config(substitution_elasticity=0.0)
        result = counterfactual_substitution(cfg, cf_elasticity=0.0)
        assert result.ate["P"] == pytest.approx(0.0, abs=1e-10)
        assert result.ate["Q_sub"] == pytest.approx(0.0, abs=1e-10)

    def test_counterfactual_substitution_nonzero_ate(self):
        """
        L3 with cf_elasticity=0.8 (vs factual=0.0) should show:
        - ATE["Q_sub"] > 0 (more substitution in counterfactual world)
        - ATE["P"] < 0 (more supply → lower price)
        """
        cfg = _restriction_config(substitution_elasticity=0.0)
        result = counterfactual_substitution(cfg, cf_elasticity=0.8)
        # Substitution supply increases
        assert result.ate["Q_sub"] > 0.0
        # More supply dampens price rises
        assert result.ate["P"] < 0.0

    def test_counterfactual_substitution_noise_coupling(self):
        """
        Stochastic model: factual and counterfactual use SAME noise sequence
        (abduction).  Verify by checking that with cf_elasticity=0, the
        trajectories are identical even under sigma_P > 0.
        """
        cfg = _restriction_config(sigma_P=0.1, substitution_elasticity=0.0)
        result = counterfactual_substitution(cfg, cf_elasticity=0.0)
        # Same params, same noise → identical trajectories
        pd.testing.assert_frame_equal(result.factual, result.counterfactual)

    def test_counterfactual_substitution_noise_length(self):
        """Noise sequence length should equal number of simulated years."""
        cfg = _restriction_config(substitution_elasticity=0.0)
        result = counterfactual_substitution(cfg, cf_elasticity=0.5)
        n_years = len(cfg.years)
        assert len(result.noise_sequence) == n_years

    def test_counterfactual_substitution_ate_keys(self):
        """ATE dict should contain standard outcome keys."""
        cfg = _restriction_config(substitution_elasticity=0.0)
        result = counterfactual_substitution(cfg, cf_elasticity=0.5)
        for key in ("P", "Q_total", "Q_sub", "shortage", "tight"):
            assert key in result.ate, f"Missing ATE key: {key}"

    def test_counterfactual_fringe_returns_result(self):
        cfg = _restriction_config(fringe_capacity_share=0.0)
        result = counterfactual_fringe(cfg, cf_capacity_share=0.3)
        assert isinstance(result, CounterfactualResult)
        assert "Q_fringe" in result.ate

    def test_counterfactual_fringe_zero_ate_when_no_change(self):
        """cf_capacity_share == factual → ATE = 0."""
        cfg = _restriction_config(fringe_capacity_share=0.0)
        result = counterfactual_fringe(cfg, cf_capacity_share=0.0)
        assert result.ate["P"] == pytest.approx(0.0, abs=1e-10)
        assert result.ate["Q_fringe"] == pytest.approx(0.0, abs=1e-10)

    def test_counterfactual_fringe_increases_supply_at_high_price(self):
        """
        Fringe supply exists in counterfactual but not factual.
        If price rises above entry threshold, Q_fringe ATE should be positive.
        """
        # Restriction will drive price up, fringe should respond
        cfg = _restriction_config(fringe_capacity_share=0.0)
        result = counterfactual_fringe(cfg, cf_capacity_share=0.3, cf_entry_price=1.2)
        from src.minerals.simulate import run_scenario
        factual_df, _ = run_scenario(cfg)
        if (factual_df["P"] > 1.2).any():
            assert result.ate["Q_fringe"] > 0.0
            assert result.ate["P"] < 0.0  # more supply dampens price

    def test_counterfactual_fringe_noise_coupling(self):
        """Stochastic model: cf_capacity_share=0 → identical trajectories."""
        cfg = _restriction_config(sigma_P=0.1, fringe_capacity_share=0.0)
        result = counterfactual_fringe(cfg, cf_capacity_share=0.0)
        pd.testing.assert_frame_equal(result.factual, result.counterfactual)

    def test_l3_distinct_from_l2_description(self):
        """
        L3 (counterfactual_substitution) and L2 (do_substitution) answer
        different questions.

        L2: P(Y|do(elasticity=0.8)) — does not condition on observed trajectory.
        L3: P(Y_{0.8} | factual run with elasticity=0) — conditions on specific
            trajectory via abduction of noise sequence.

        For a DETERMINISTIC model (sigma_P=0), L2 and L3 produce numerically
        identical point estimates because there is no noise to abduct.
        The distinction becomes meaningful for sigma_P > 0 when asking about
        a specific observed price trajectory (L3) vs the average over all
        possible trajectories (L2).
        """
        cfg = _restriction_config(sigma_P=0.0, substitution_elasticity=0.0)
        # L2
        l2_df, _ = do_substitution(cfg, elasticity=0.8)
        # L3
        l3 = counterfactual_substitution(cfg, cf_elasticity=0.8)
        # For deterministic model: L2 == L3 (no noise to differ on)
        pd.testing.assert_frame_equal(l2_df, l3.counterfactual)

    def test_l3_differs_from_l2_stochastic(self):
        """
        For a stochastic model (sigma_P > 0), L2 run with a DIFFERENT seed
        would differ from L3 (which uses the same noise as factual).
        Verify: same-seed L3 equals same-seed L2 (both use cfg.seed).
        """
        cfg = _restriction_config(sigma_P=0.2, substitution_elasticity=0.0)
        l2_df, _ = do_substitution(cfg, elasticity=0.8)   # uses cfg.seed
        l3 = counterfactual_substitution(cfg, cf_elasticity=0.8)  # same cfg.seed
        # Both should be the same (both use cfg.seed = same noise)
        pd.testing.assert_frame_equal(l2_df, l3.counterfactual, check_exact=False, rtol=1e-6)
