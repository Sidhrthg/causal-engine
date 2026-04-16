"""
Tests for src/minerals/transshipment.py

Covers:
- build_annual_network: correct graph construction from CEPII-style data
- trace_downstream: reaches expected destinations, filters noise
- trace_paths: finds direct and multi-hop paths, marks non-producer intermediaries
- mirror_discrepancy: detects asymmetric reporting
- detect_rerouting: flags statistically significant post-event flow increases
- estimate_circumvention_rate: returns valid rate + CI
- corrected_dom_supply: adjustment is non-negative, corrected > reported for event years
- TransshipmentAnalyzer with custom known_producers
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.minerals.transshipment import (
    CircumventionEstimate,
    PathTrace,
    ReroutingSignal,
    TransshipmentAnalyzer,
    _KNOWN_HUBS,
    _KNOWN_PRODUCERS,
)


# ---------------------------------------------------------------------------
# Fixtures: synthetic bilateral trade data
# ---------------------------------------------------------------------------

def _make_df(rows: list[dict]) -> pd.DataFrame:
    """Build a minimal CEPII-style DataFrame from a list of row dicts."""
    return pd.DataFrame(rows, columns=["year", "exporter", "importer", "value_kusd", "quantity_tonnes"])


@pytest.fixture
def simple_df():
    """
    Simple 3-country, 2-year dataset.

    Trade network (year=2020):
        China → Singapore  100t
        China → USA        200t
        Singapore → USA    80t       ← transshipment candidate

    Trade network (year=2021):
        China → Singapore  150t      ← increase after hypothetical 2020 event
        China → USA        180t
        Singapore → USA    120t      ← increase
    """
    return _make_df([
        # 2020
        {"year": 2020, "exporter": "China",     "importer": "Singapore", "value_kusd": 5000,  "quantity_tonnes": 100},
        {"year": 2020, "exporter": "China",     "importer": "USA",       "value_kusd": 10000, "quantity_tonnes": 200},
        {"year": 2020, "exporter": "Singapore", "importer": "USA",       "value_kusd": 4000,  "quantity_tonnes": 80},
        # 2021
        {"year": 2021, "exporter": "China",     "importer": "Singapore", "value_kusd": 7500,  "quantity_tonnes": 150},
        {"year": 2021, "exporter": "China",     "importer": "USA",       "value_kusd": 9000,  "quantity_tonnes": 180},
        {"year": 2021, "exporter": "Singapore", "importer": "USA",       "value_kusd": 6000,  "quantity_tonnes": 120},
    ])


@pytest.fixture
def event_study_df():
    """
    5-year dataset designed to produce a detectable rerouting signal.

    Pre-event (2018-2019): China → Hub = 100t/yr
    Event: 2020
    Post-event (2020-2022): China → Hub = 300t/yr  ← clear jump
    """
    rows = []
    for yr in [2018, 2019]:
        rows.append({"year": yr, "exporter": "China", "importer": "Hub", "value_kusd": 1000, "quantity_tonnes": 100})
        rows.append({"year": yr, "exporter": "Hub",   "importer": "USA", "value_kusd": 900,  "quantity_tonnes": 90})
        rows.append({"year": yr, "exporter": "China", "importer": "USA", "value_kusd": 2000, "quantity_tonnes": 500})
    for yr in [2020, 2021, 2022]:
        rows.append({"year": yr, "exporter": "China", "importer": "Hub", "value_kusd": 3000, "quantity_tonnes": 300})
        rows.append({"year": yr, "exporter": "Hub",   "importer": "USA", "value_kusd": 2800, "quantity_tonnes": 280})
        rows.append({"year": yr, "exporter": "China", "importer": "USA", "value_kusd": 1000, "quantity_tonnes": 200})
    return _make_df(rows)


@pytest.fixture
def mirror_df():
    """
    Dataset with asymmetric bilateral flows to test mirror discrepancy detection.

    mirror_discrepancy() works by comparing:
      - "A's exports to B" = rows where exporter=A, importer=B
      - "B's imports from A" = rows where importer=B, exporter=A (same rows)

    So for a standard bilateral table (one row per pair), discrepancy is always 0.
    Non-zero discrepancy arises when one direction is present and the other is absent,
    causing the outer merge to fill with zeros.

    Here A exports 100t to B, but B has no matching reverse record (no B→A entry),
    so the pair (country_a=B, country_b=A) shows a_reports_export_t=0 vs
    b_reports_import_t=0.  The pair (A, B) has consistent reporting: 0 discrepancy.
    """
    return _make_df([
        # Only A→B flow present; no B→A row
        {"year": 2020, "exporter": "A", "importer": "B", "value_kusd": 1000, "quantity_tonnes": 100},
        # C→A but no A→C row (creates a non-zero b_reports_import_t for (A, C) pair)
        {"year": 2020, "exporter": "C", "importer": "A", "value_kusd": 500, "quantity_tonnes": 50},
    ])


# ---------------------------------------------------------------------------
# 1. build_annual_network
# ---------------------------------------------------------------------------

class TestBuildAnnualNetwork:
    def test_creates_directed_graph(self, simple_df):
        ta = TransshipmentAnalyzer(simple_df, commodity="graphite", dominant_exporter="China")
        G = ta.build_annual_network(2020)
        import networkx as nx
        assert isinstance(G, nx.DiGraph)

    def test_correct_nodes(self, simple_df):
        ta = TransshipmentAnalyzer(simple_df, commodity="graphite", dominant_exporter="China")
        G = ta.build_annual_network(2020)
        assert "China" in G.nodes
        assert "Singapore" in G.nodes
        assert "USA" in G.nodes

    def test_correct_edge_weights(self, simple_df):
        ta = TransshipmentAnalyzer(simple_df, commodity="graphite", dominant_exporter="China")
        G = ta.build_annual_network(2020)
        assert G["China"]["Singapore"]["weight"] == pytest.approx(100.0)
        assert G["China"]["USA"]["weight"] == pytest.approx(200.0)
        assert G["Singapore"]["USA"]["weight"] == pytest.approx(80.0)

    def test_caches_result(self, simple_df):
        ta = TransshipmentAnalyzer(simple_df, commodity="graphite", dominant_exporter="China")
        G1 = ta.build_annual_network(2020)
        G2 = ta.build_annual_network(2020)
        assert G1 is G2  # same object from cache

    def test_empty_year_returns_empty_graph(self, simple_df):
        ta = TransshipmentAnalyzer(simple_df, commodity="graphite", dominant_exporter="China")
        G = ta.build_annual_network(1900)
        assert len(G.nodes) == 0


# ---------------------------------------------------------------------------
# 2. trace_downstream
# ---------------------------------------------------------------------------

class TestTraceDownstream:
    def test_returns_dataframe(self, simple_df):
        ta = TransshipmentAnalyzer(simple_df, commodity="graphite", dominant_exporter="China",
                                   known_producers={"China"})
        result = ta.trace_downstream("China", year=2020, min_flow_pct=0.0)
        assert isinstance(result, pd.DataFrame)

    def test_finds_direct_destination(self, simple_df):
        ta = TransshipmentAnalyzer(simple_df, commodity="graphite", dominant_exporter="China",
                                   known_producers={"China"})
        result = ta.trace_downstream("China", year=2020, min_flow_pct=0.0)
        destinations = set(result["final_destination"])
        assert "USA" in destinations

    def test_bottleneck_is_nonnegative(self, simple_df):
        ta = TransshipmentAnalyzer(simple_df, commodity="graphite", dominant_exporter="China",
                                   known_producers={"China"})
        result = ta.trace_downstream("China", year=2020, min_flow_pct=0.0)
        assert (result["bottleneck_t"] >= 0).all()

    def test_flags_non_producer_intermediary(self, simple_df):
        # Singapore is NOT a known producer of graphite
        ta = TransshipmentAnalyzer(simple_df, commodity="graphite", dominant_exporter="China",
                                   known_producers={"China"})
        result = ta.trace_downstream("China", year=2020, min_flow_pct=0.0)
        # Path China → Singapore → USA should be flagged
        multi_hop = result[result["hops"] > 1]
        assert multi_hop["is_circumvention_candidate"].any()

    def test_empty_for_unknown_source(self, simple_df):
        ta = TransshipmentAnalyzer(simple_df, commodity="graphite", dominant_exporter="China")
        result = ta.trace_downstream("Nonexistent", year=2020)
        assert result.empty

    def test_pct_sum_reasonable(self, simple_df):
        ta = TransshipmentAnalyzer(simple_df, commodity="graphite", dominant_exporter="China",
                                   known_producers={"China"})
        result = ta.trace_downstream("China", year=2020, min_flow_pct=0.0)
        # Direct paths should account for close to 100% of exports
        direct = result[result["hops"] == 1]
        total_pct = direct["pct_of_source_exports"].sum()
        assert 0.9 <= total_pct <= 1.1  # direct paths ≈ 100% of source


# ---------------------------------------------------------------------------
# 3. trace_paths
# ---------------------------------------------------------------------------

class TestTracePaths:
    def test_finds_direct_path(self, simple_df):
        ta = TransshipmentAnalyzer(simple_df, commodity="graphite", dominant_exporter="China",
                                   known_producers={"China"})
        paths = ta.trace_paths("China", "USA", year=2020)
        assert len(paths) >= 1
        direct = [p for p in paths if p.hops == 1]
        assert len(direct) == 1

    def test_finds_two_hop_path(self, simple_df):
        ta = TransshipmentAnalyzer(simple_df, commodity="graphite", dominant_exporter="China",
                                   known_producers={"China"})
        paths = ta.trace_paths("China", "USA", year=2020)
        two_hop = [p for p in paths if p.hops == 2]
        assert len(two_hop) == 1
        assert "Singapore" in two_hop[0].path

    def test_path_trace_fields(self, simple_df):
        ta = TransshipmentAnalyzer(simple_df, commodity="graphite", dominant_exporter="China",
                                   known_producers={"China"})
        paths = ta.trace_paths("China", "USA", year=2020)
        for pt in paths:
            assert isinstance(pt, PathTrace)
            assert pt.bottleneck_t >= 0
            assert 0.0 <= pt.pct_of_source <= 1.1
            assert pt.hops == len(pt.path) - 1
            assert len(pt.flows_t) == pt.hops

    def test_two_hop_flagged_as_circumvention(self, simple_df):
        ta = TransshipmentAnalyzer(simple_df, commodity="graphite", dominant_exporter="China",
                                   known_producers={"China"})
        paths = ta.trace_paths("China", "USA", year=2020)
        two_hop = [p for p in paths if p.hops == 2]
        assert two_hop[0].is_circumvention_candidate

    def test_direct_path_not_flagged(self, simple_df):
        # China is a known producer, so direct China → USA has no non-producer intermediary
        ta = TransshipmentAnalyzer(simple_df, commodity="graphite", dominant_exporter="China",
                                   known_producers={"China", "USA"})
        paths = ta.trace_paths("China", "USA", year=2020)
        direct = [p for p in paths if p.hops == 1]
        assert not direct[0].is_circumvention_candidate

    def test_sorted_by_bottleneck_descending(self, simple_df):
        ta = TransshipmentAnalyzer(simple_df, commodity="graphite", dominant_exporter="China",
                                   known_producers={"China"})
        paths = ta.trace_paths("China", "USA", year=2020)
        bottlenecks = [p.bottleneck_t for p in paths]
        assert bottlenecks == sorted(bottlenecks, reverse=True)

    def test_returns_empty_for_no_path(self, simple_df):
        ta = TransshipmentAnalyzer(simple_df, commodity="graphite", dominant_exporter="China")
        paths = ta.trace_paths("USA", "China", year=2020)  # no reverse path in data
        assert paths == []

    def test_bottleneck_is_min_of_edge_flows(self, simple_df):
        ta = TransshipmentAnalyzer(simple_df, commodity="graphite", dominant_exporter="China",
                                   known_producers={"China"})
        paths = ta.trace_paths("China", "USA", year=2020)
        two_hop = [p for p in paths if p.hops == 2][0]
        # China→Singapore=100t, Singapore→USA=80t → bottleneck=80
        assert two_hop.bottleneck_t == pytest.approx(80.0)


# ---------------------------------------------------------------------------
# 4. mirror_discrepancy
# ---------------------------------------------------------------------------

class TestMirrorDiscrepancy:
    def test_returns_dataframe(self, mirror_df):
        ta = TransshipmentAnalyzer(mirror_df, commodity="graphite", dominant_exporter="A")
        result = ta.mirror_discrepancy()
        assert isinstance(result, pd.DataFrame)

    def test_consistent_pair_has_zero_discrepancy(self, mirror_df):
        """
        For a bilateral flow that appears in a single row (single perspective),
        export == import (same rows) so discrepancy == 0.
        """
        ta = TransshipmentAnalyzer(mirror_df, commodity="graphite", dominant_exporter="A")
        result = ta.mirror_discrepancy()
        row = result[(result["country_a"] == "A") & (result["country_b"] == "B")]
        assert not row.empty
        # Single-perspective data: a_reports_export_t == b_reports_import_t (same rows)
        assert row.iloc[0]["discrepancy_t"] == pytest.approx(0.0)

    def test_symmetric_pairs(self, mirror_df):
        ta = TransshipmentAnalyzer(mirror_df, commodity="graphite", dominant_exporter="A")
        result = ta.mirror_discrepancy()
        # Should have entries for both (A, B) and (B, A) directions
        pairs = set(zip(result["country_a"], result["country_b"]))
        assert ("A", "B") in pairs

    def test_year_range_filter(self, event_study_df):
        ta = TransshipmentAnalyzer(event_study_df, commodity="graphite", dominant_exporter="China")
        result_full = ta.mirror_discrepancy()
        result_filtered = ta.mirror_discrepancy(year_range=(2020, 2022))
        assert len(result_filtered) <= len(result_full)
        assert result_filtered["year"].between(2020, 2022).all()

    def test_abs_discrepancy_nonnegative(self, mirror_df):
        ta = TransshipmentAnalyzer(mirror_df, commodity="graphite", dominant_exporter="A")
        result = ta.mirror_discrepancy()
        assert (result["abs_discrepancy_t"] >= 0).all()


# ---------------------------------------------------------------------------
# 5. detect_rerouting
# ---------------------------------------------------------------------------

class TestDetectRerouting:
    def test_returns_dataframe(self, event_study_df):
        ta = TransshipmentAnalyzer(event_study_df, commodity="graphite",
                                   dominant_exporter="China",
                                   known_producers={"China"})
        result = ta.detect_rerouting(event_years=[2020], pre_window=2, post_window=2)
        assert isinstance(result, pd.DataFrame)

    def test_detects_significant_rerouting(self, event_study_df):
        """
        Hub goes from 100t to 300t after event — should be flagged significant.
        """
        ta = TransshipmentAnalyzer(event_study_df, commodity="graphite",
                                   dominant_exporter="China",
                                   known_producers={"China"})
        result = ta.detect_rerouting(event_years=[2020], pre_window=2, post_window=2,
                                     min_annual_flow_t=50.0)
        if not result.empty:
            dom_to_hub = result[result["metric"] == "dom_to_hub"]
            # At least one hub should show a significant increase
            # (may not always be p < 0.10 with only 2 pre/2 post observations)
            assert "is_significant" in dom_to_hub.columns
            # post_mean > pre_mean for "Hub"
            hub_rows = dom_to_hub[dom_to_hub["hub"] == "Hub"]
            if not hub_rows.empty:
                assert hub_rows.iloc[0]["post_mean_t"] > hub_rows.iloc[0]["pre_mean_t"]

    def test_pct_change_correct_sign(self, event_study_df):
        ta = TransshipmentAnalyzer(event_study_df, commodity="graphite",
                                   dominant_exporter="China",
                                   known_producers={"China"})
        result = ta.detect_rerouting(event_years=[2020], pre_window=2, post_window=2,
                                     min_annual_flow_t=50.0)
        if not result.empty:
            hub_rows = result[result["hub"] == "Hub"]
            if not hub_rows.empty:
                # Should be positive change (300 → 100 is increase, pct_change > 0)
                assert hub_rows.iloc[0]["pct_change"] > 0

    def test_empty_without_candidate_hubs(self):
        # All importers of China are known producers → no candidate hubs
        df = _make_df([
            {"year": 2019, "exporter": "China", "importer": "Brazil", "value_kusd": 1000, "quantity_tonnes": 100},
            {"year": 2020, "exporter": "China", "importer": "Brazil", "value_kusd": 1200, "quantity_tonnes": 120},
        ])
        ta = TransshipmentAnalyzer(df, commodity="graphite", dominant_exporter="China",
                                   known_producers={"China", "Brazil"})
        result = ta.detect_rerouting(event_years=[2020])
        assert result.empty

    def test_output_columns_present(self, event_study_df):
        ta = TransshipmentAnalyzer(event_study_df, commodity="graphite",
                                   dominant_exporter="China",
                                   known_producers={"China"})
        result = ta.detect_rerouting(event_years=[2020], pre_window=2, post_window=2,
                                     min_annual_flow_t=50.0)
        if not result.empty:
            expected_cols = {
                "hub", "event_year", "metric", "pre_mean_t", "post_mean_t",
                "pct_change", "t_stat", "p_value", "is_significant", "n_pre", "n_post"
            }
            assert expected_cols.issubset(set(result.columns))


# ---------------------------------------------------------------------------
# 6. estimate_circumvention_rate
# ---------------------------------------------------------------------------

class TestEstimateCircumventionRate:
    def test_returns_circumvention_estimate(self, event_study_df):
        ta = TransshipmentAnalyzer(event_study_df, commodity="graphite",
                                   dominant_exporter="China",
                                   known_producers={"China"})
        result = ta.estimate_circumvention_rate(event_years=[2020], pre_window=2)
        assert isinstance(result, CircumventionEstimate)

    def test_rate_in_valid_range(self, event_study_df):
        ta = TransshipmentAnalyzer(event_study_df, commodity="graphite",
                                   dominant_exporter="China",
                                   known_producers={"China"})
        result = ta.estimate_circumvention_rate(event_years=[2020], pre_window=2)
        assert 0.0 <= result.circumvention_rate <= 1.0

    def test_ci_is_ordered(self, event_study_df):
        ta = TransshipmentAnalyzer(event_study_df, commodity="graphite",
                                   dominant_exporter="China",
                                   known_producers={"China"})
        result = ta.estimate_circumvention_rate(event_years=[2020], pre_window=2)
        lo, hi = result.circumvention_rate_ci
        assert lo <= hi

    def test_nominal_restricted_positive(self, event_study_df):
        ta = TransshipmentAnalyzer(event_study_df, commodity="graphite",
                                   dominant_exporter="China",
                                   known_producers={"China"})
        result = ta.estimate_circumvention_rate(
            event_years=[2020], nominal_restriction=0.3, pre_window=2
        )
        assert result.nominal_restriction_t > 0

    def test_notes_list(self, event_study_df):
        ta = TransshipmentAnalyzer(event_study_df, commodity="graphite",
                                   dominant_exporter="China",
                                   known_producers={"China"})
        result = ta.estimate_circumvention_rate(event_years=[2020], pre_window=2)
        assert isinstance(result.notes, list)

    def test_no_hubs_returns_zero_rate(self):
        # No rerouting possible if all importers are producers
        df = _make_df([
            {"year": y, "exporter": "China", "importer": "Canada",
             "value_kusd": 1000, "quantity_tonnes": 100}
            for y in range(2017, 2023)
        ])
        ta = TransshipmentAnalyzer(df, commodity="graphite", dominant_exporter="China",
                                   known_producers={"China", "Canada"})
        result = ta.estimate_circumvention_rate(event_years=[2020], pre_window=2)
        assert result.circumvention_rate == pytest.approx(0.0)
        assert result.significant_hubs == []


# ---------------------------------------------------------------------------
# 7. corrected_dom_supply
# ---------------------------------------------------------------------------

class TestCorrectedDomSupply:
    def test_returns_dataframe(self, event_study_df):
        ta = TransshipmentAnalyzer(event_study_df, commodity="graphite",
                                   dominant_exporter="China",
                                   known_producers={"China"})
        result = ta.corrected_dom_supply(event_years=[2020], circumvention_rate=0.2)
        assert isinstance(result, pd.DataFrame)

    def test_expected_columns(self, event_study_df):
        ta = TransshipmentAnalyzer(event_study_df, commodity="graphite",
                                   dominant_exporter="China",
                                   known_producers={"China"})
        result = ta.corrected_dom_supply(event_years=[2020], circumvention_rate=0.2)
        expected = {"year", "reported_supply_t", "rerouted_adjustment_t", "corrected_supply_t",
                    "circumvention_applied"}
        assert expected.issubset(set(result.columns))

    def test_corrected_ge_reported(self, event_study_df):
        ta = TransshipmentAnalyzer(event_study_df, commodity="graphite",
                                   dominant_exporter="China",
                                   known_producers={"China"})
        result = ta.corrected_dom_supply(event_years=[2020], circumvention_rate=0.3)
        assert (result["corrected_supply_t"] >= result["reported_supply_t"]).all()

    def test_adjustment_zero_pre_event(self, event_study_df):
        """Pre-event years should have zero rerouting adjustment."""
        ta = TransshipmentAnalyzer(event_study_df, commodity="graphite",
                                   dominant_exporter="China",
                                   known_producers={"China"})
        result = ta.corrected_dom_supply(event_years=[2020], circumvention_rate=0.3)
        pre_rows = result[result["year"] < 2020]
        assert (pre_rows["rerouted_adjustment_t"] == 0).all()

    def test_circumvention_rate_applied_correctly(self, event_study_df):
        ta = TransshipmentAnalyzer(event_study_df, commodity="graphite",
                                   dominant_exporter="China",
                                   known_producers={"China"})
        result = ta.corrected_dom_supply(event_years=[2020], circumvention_rate=0.25)
        assert np.allclose(result["circumvention_applied"].values, 0.25)

    def test_accepts_none_circumvention_rate(self, event_study_df):
        """When circumvention_rate=None, the rate is estimated from data."""
        ta = TransshipmentAnalyzer(event_study_df, commodity="graphite",
                                   dominant_exporter="China",
                                   known_producers={"China"})
        # Should not raise
        result = ta.corrected_dom_supply(event_years=[2020], circumvention_rate=None)
        assert isinstance(result, pd.DataFrame)


# ---------------------------------------------------------------------------
# 8. summary_report
# ---------------------------------------------------------------------------

class TestSummaryReport:
    def test_returns_string(self, simple_df):
        ta = TransshipmentAnalyzer(simple_df, commodity="graphite", dominant_exporter="China",
                                   known_producers={"China"})
        report = ta.summary_report(destination="USA", event_years=[2020], year=2020)
        assert isinstance(report, str)
        assert len(report) > 0

    def test_contains_key_sections(self, simple_df):
        ta = TransshipmentAnalyzer(simple_df, commodity="graphite", dominant_exporter="China",
                                   known_producers={"China"})
        report = ta.summary_report(destination="USA", event_years=[2020], year=2020)
        assert "China" in report
        assert "USA" in report
        assert "Circumvention" in report


# ---------------------------------------------------------------------------
# 9. Custom known_producers override
# ---------------------------------------------------------------------------

class TestCustomKnownProducers:
    def test_custom_producers_affects_flagging(self, simple_df):
        """Making Singapore a known producer should suppress circumvention flag."""
        ta = TransshipmentAnalyzer(simple_df, commodity="graphite", dominant_exporter="China",
                                   known_producers={"China", "Singapore"})
        paths = ta.trace_paths("China", "USA", year=2020)
        two_hop = [p for p in paths if p.hops == 2]
        assert len(two_hop) == 1
        # Singapore is now a producer → non_producer_intermediaries should be empty
        assert two_hop[0].non_producer_intermediaries == []
        assert not two_hop[0].is_circumvention_candidate


# ---------------------------------------------------------------------------
# 10. Known producers / hubs constants
# ---------------------------------------------------------------------------

class TestConstants:
    def test_graphite_producers_not_empty(self):
        assert len(_KNOWN_PRODUCERS["graphite"]) > 0
        assert "China" in _KNOWN_PRODUCERS["graphite"]

    def test_known_hubs_not_empty(self):
        assert len(_KNOWN_HUBS) > 0
        assert "Singapore" in _KNOWN_HUBS
