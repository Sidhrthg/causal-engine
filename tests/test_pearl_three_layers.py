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
    counterfactual_step,
    counterfactual_trajectory,
    interventional_identifiability,
    mutilated_graph_for_do,
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
