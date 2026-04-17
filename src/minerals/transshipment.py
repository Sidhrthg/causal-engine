"""
Transshipment and trade circumvention detection for critical minerals.

Answers the question: "Where does China's graphite actually end up?"

Given CEPII bilateral trade data (year, exporter, importer, quantity_tonnes,
value_kusd), this module:

1. Traces multi-hop flows — China → Singapore → Brazil → USA — with
   value matching at each hop to see how much of the original flow
   survives to the final destination.

2. Detects statistically significant rerouting after restriction events —
   do non-producer hub countries show abnormal inbound flows from the
   restricting country AND outbound flows to final destinations?

3. Computes mirror statistics discrepancy — A reports exporting X tonnes
   to B; B reports importing Y tonnes from A.  Y > X = hidden flows or
   transshipment mislabelling.

4. Estimates the effective circumvention rate — what fraction of a
   nominal export restriction is actually evaded through third countries?

5. Produces a corrected supply series for use in parameter fitting,
   removing the circumvention bias from eta_D and tau_K estimates.

Usage
-----
    from src.minerals.transshipment import TransshipmentAnalyzer
    import pandas as pd

    df = pd.read_csv("data/canonical/cepii_graphite.csv")
    ta = TransshipmentAnalyzer(df, commodity="graphite", dominant_exporter="China")

    # Where does China's graphite go (all hops)?
    flows = ta.trace_downstream("China", year=2023)
    print(flows.head(20))

    # Specific path: China → Singapore → USA
    paths = ta.trace_paths("China", "United States", year=2023)
    for p in paths[:5]:
        print(p["path"], "→", f"{p['bottleneck_t']:.0f}t")

    # Did flows through non-producer hubs spike after 2023 restriction?
    rerouting = ta.detect_rerouting(event_years=[2023])
    print(rerouting[rerouting["is_significant"]])

    # Mirror stats: who under/over-reports?
    disc = ta.mirror_discrepancy()
    print(disc.nlargest(10, "discrepancy_pct"))

    # Effective restriction
    cr = ta.estimate_circumvention_rate(event_years=[2023], nominal_restriction=0.3)
    print(f"Circumvention rate: {cr['circumvention_rate']:.1%}")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import networkx as nx
import numpy as np
import pandas as pd
from scipy import stats

from src.minerals.country_codes import normalize_country_names

logger = logging.getLogger(__name__)

# Known primary producers per commodity (all names in canonical resolved form).
# A country NOT in this set that shows large exports of the commodity
# is likely a transshipment hub or processor, not a primary producer.
# Note: "processor" countries (South Korea for graphite anode, Japan for lithium
# battery materials) appear here because they add genuine value — flagging them
# as circumvention candidates would be misleading.
_KNOWN_PRODUCERS: Dict[str, Set[str]] = {
    "graphite": {
        # Primary mining nations (USGS MCS 2024)
        # World mine production: China 1.23Mt, Mozambique 96kt, Madagascar 100kt,
        # Brazil 73kt, North Korea 8.1kt, India 11.5kt, Tanzania ~6kt, Sri Lanka 16kt
        # Ukraine 2kt (disrupted), Austria 0.5kt, Vietnam 0.5kt
        "China", "Mozambique", "Tanzania", "Madagascar",
        "Ukraine", "Brazil", "Canada", "India", "North Korea",
        "South Africa", "Zimbabwe", "Namibia", "Austria", "Sri Lanka",
        "Vietnam", "Turkey", "Russia",
        # Major anode/electrode processors — add genuine value via manufacturing,
        # not transshipment. South Korea/Japan transform raw flake into battery-grade
        # spherical graphite (different product, substantial transformation).
        "South Korea", "Japan",
    },
    "lithium": {
        # Mining nations
        "Australia", "Chile", "Argentina", "China", "Zimbabwe",
        "Portugal", "Brazil", "USA", "Canada", "Bolivia",
        # Processors
        "Japan", "South Korea",
    },
    "cobalt": {
        # Mining nations
        "DRC", "Congo", "Russia", "Australia", "Philippines", "Cuba",
        "Papua New Guinea", "Canada", "China", "Zambia",
        # Processors
        "Finland", "Norway",
    },
    "nickel": {
        # Mining nations
        "Indonesia", "Philippines", "Russia", "Canada",
        "Australia", "Brazil", "China", "New Caledonia",
        "Guatemala", "Cuba",
    },
    "rare_earths": {
        "China", "USA", "Australia", "Myanmar", "India",
        "Brazil", "Russia", "Vietnam", "Canada", "Thailand",
    },
}

# Known transshipment / entrepôt hubs — countries that re-export more
# than they produce for most commodities.
_KNOWN_HUBS = {
    "Hong Kong", "Singapore", "United Arab Emirates", "UAE",
    "Netherlands", "Belgium", "Switzerland", "Malaysia",
    "South Korea", "Japan",  # sometimes process + re-export
}


@dataclass
class PathTrace:
    """A single multi-hop trade route with value matching."""
    path: List[str]
    flows_t: List[float]            # quantity at each edge (tonnes)
    flows_kusd: List[float]         # value at each edge (kUSD)
    bottleneck_t: float             # min flow along path (tonnes)
    source_total_t: float           # total exports of source node this year
    pct_of_source: float            # bottleneck / source_total (0-1)
    hops: int
    non_producer_intermediaries: List[str]   # countries that are neither
                                             # known producers nor processors
    is_circumvention_candidate: bool         # True if any non-producer in path
    # Note: "circumvention candidate" means the path passes through countries
    # with no legitimate production/processing role for this commodity.
    # Processor countries (South Korea, Japan for graphite anode) are in
    # _KNOWN_PRODUCERS and are NOT flagged — they add genuine value.
    # Shell company hubs (UAE, Switzerland, Singapore for non-processed goods)
    # ARE flagged as circumvention candidates.


@dataclass
class ReroutingSignal:
    """Statistical test result for rerouting through a hub after a restriction event."""
    hub: str
    event_year: int
    metric: str                     # "dom_to_hub" or "hub_to_world"
    pre_mean_t: float
    post_mean_t: float
    pct_change: float
    t_stat: float
    p_value: float
    is_significant: bool            # p < 0.10
    n_pre: int
    n_post: int


@dataclass
class CircumventionEstimate:
    """Summary of estimated circumvention for a set of restriction events."""
    event_years: List[int]
    nominal_restriction_t: float    # total restricted volume (tonnes)
    detected_rerouted_t: float      # volume detected flowing through non-producer hubs
    circumvention_rate: float       # detected_rerouted / nominal_restricted
    circumvention_rate_ci: Tuple[float, float]  # 95% bootstrap CI
    significant_hubs: List[str]     # hubs with p < 0.10 rerouting signal
    notes: List[str]


class TransshipmentAnalyzer:
    """
    Detect trade circumvention and transshipment in bilateral commodity flows.

    Parameters
    ----------
    df : pd.DataFrame
        CEPII bilateral trade data with columns:
        year, exporter, importer, value_kusd, quantity_tonnes
    commodity : str
        Commodity name (e.g. "graphite", "lithium").
    dominant_exporter : str
        The country subject to export restrictions (e.g. "China").
    known_producers : set[str] | None
        Override for known domestic producers. If None, uses built-in list.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        commodity: str = "graphite",
        dominant_exporter: str = "China",
        known_producers: Optional[Set[str]] = None,
    ):
        # Normalize country codes before any analysis
        self.df = normalize_country_names(df.copy())
        self.commodity = commodity
        self.dominant_exporter = dominant_exporter
        self.known_producers: Set[str] = (
            known_producers
            if known_producers is not None
            else _KNOWN_PRODUCERS.get(commodity, set())
        )
        self._net_cache: Dict[int, nx.DiGraph] = {}

    # ------------------------------------------------------------------
    # 1. Annual trade network
    # ------------------------------------------------------------------

    def build_annual_network(self, year: int) -> nx.DiGraph:
        """
        Build a directed trade graph for a single year.

        Nodes = countries.  Edges = (exporter → importer) with attributes:
            weight      — quantity_tonnes
            value_kusd  — trade value
        """
        if year in self._net_cache:
            return self._net_cache[year]

        sub = self.df[self.df["year"] == year]
        G = nx.DiGraph()
        for _, row in sub.iterrows():
            exp = str(row["exporter"])
            imp = str(row["importer"])
            qt = float(row.get("quantity_tonnes", 0) or 0)
            vk = float(row.get("value_kusd", 0) or 0)
            if G.has_edge(exp, imp):
                G[exp][imp]["weight"] += qt
                G[exp][imp]["value_kusd"] += vk
            else:
                G.add_edge(exp, imp, weight=qt, value_kusd=vk)

        self._net_cache[year] = G
        return G

    # ------------------------------------------------------------------
    # 2. Trace downstream: where does source's product actually go?
    # ------------------------------------------------------------------

    def _flow_pruned_paths(
        self,
        G: nx.DiGraph,
        source: str,
        destination: Optional[str],
        source_total: float,
        max_hops: int,
        min_flow_pct: float,
        top_k: int = 200,
    ) -> List[dict]:
        """
        Flow-pruned BFS path finder.  Replaces nx.all_simple_paths which is
        exponential on dense trade networks.

        At each BFS frontier node we only expand edges whose flow exceeds the
        min_flow_pct threshold (relative to source total), and we keep at most
        top_k paths total sorted by bottleneck descending.  This makes the
        algorithm O(nodes × max_hops) in practice.

        Args:
            G:            Directed trade graph (edges have 'weight' attr).
            source:       Starting country.
            destination:  End country to match, or None for all destinations.
            source_total: Total outbound flow from source (for pct normalisation).
            max_hops:     Maximum path length.
            min_flow_pct: Minimum bottleneck as fraction of source_total.
            top_k:        Maximum number of paths to return.

        Returns:
            List of path dicts, sorted by bottleneck_t descending.
        """
        min_t = source_total * min_flow_pct
        results = []

        # State: (current_node, path_so_far, flows_so_far, value_so_far, visited)
        # Use a stack for DFS (avoids recursion depth limits)
        stack = [(source, [source], [], [], {source})]

        while stack and len(results) < top_k * 10:
            node, path, flows, values, visited = stack.pop()

            # Record this path if it ends at destination (or any node if dest=None)
            if len(path) > 1:
                bottleneck = min(flows) if flows else 0.0
                if bottleneck >= min_t:
                    end = path[-1]
                    if destination is None or end == destination:
                        results.append({
                            "path": path,
                            "flows_t": flows,
                            "flows_kusd": values,
                            "bottleneck_t": bottleneck,
                        })

            # Expand if under max_hops
            if len(path) - 1 < max_hops:
                for nb in G.successors(node):
                    if nb in visited:
                        continue
                    edge_flow = G[node][nb].get("weight", 0.0)
                    edge_val = G[node][nb].get("value_kusd", 0.0)
                    new_bottleneck = min(min(flows), edge_flow) if flows else edge_flow
                    if new_bottleneck < min_t:
                        continue  # prune this branch
                    stack.append((
                        nb,
                        path + [nb],
                        flows + [edge_flow],
                        values + [edge_val],
                        visited | {nb},
                    ))

        # Sort by bottleneck descending, keep top_k
        results.sort(key=lambda x: x["bottleneck_t"], reverse=True)
        return results[:top_k]

    def trace_downstream(
        self,
        source: str,
        year: int,
        max_hops: int = 4,
        min_flow_pct: float = 0.001,
        top_k: int = 100,
    ) -> pd.DataFrame:
        """
        Trace all destinations reachable from ``source`` within ``max_hops``.

        Uses a flow-pruned BFS (not nx.all_simple_paths) so it runs in
        milliseconds even on dense networks.

        Args:
            source:        Country whose exports to trace (e.g. "China").
            year:          Year of trade data to use.
            max_hops:      Maximum chain length to follow.
            min_flow_pct:  Minimum bottleneck as fraction of source total to
                           include; filters noise routes.
            top_k:         Maximum number of paths to return.

        Returns:
            DataFrame with columns:
                path, hops, intermediaries, final_destination,
                bottleneck_t, pct_of_source_exports,
                non_producer_intermediaries, is_circumvention_candidate
        """
        G = self.build_annual_network(year)
        if source not in G:
            return pd.DataFrame()

        source_total = sum(
            G[source][nb].get("weight", 0.0) for nb in G.successors(source)
        )
        if source_total == 0:
            return pd.DataFrame()

        raw = self._flow_pruned_paths(
            G, source, None, source_total, max_hops, min_flow_pct, top_k=top_k
        )

        rows = []
        for r in raw:
            path = r["path"]
            intermediaries = path[1:-1]
            non_prod = [c for c in intermediaries if c not in self.known_producers]
            rows.append({
                "path": " → ".join(path),
                "path_list": path,
                "hops": len(path) - 1,
                "intermediaries": intermediaries,
                "final_destination": path[-1],
                "bottleneck_t": r["bottleneck_t"],
                "pct_of_source_exports": r["bottleneck_t"] / source_total,
                "non_producer_intermediaries": non_prod,
                "is_circumvention_candidate": len(non_prod) > 0,
            })

        if not rows:
            return pd.DataFrame()

        return (
            pd.DataFrame(rows)
            .sort_values("bottleneck_t", ascending=False)
            .reset_index(drop=True)
        )

    # ------------------------------------------------------------------
    # 3. Trace a specific source → destination path
    # ------------------------------------------------------------------

    def trace_paths(
        self,
        source: str,
        destination: str,
        year: int,
        max_hops: int = 4,
        min_flow_pct: float = 0.0001,
        top_k: int = 50,
    ) -> List[PathTrace]:
        """
        Find the highest-flow paths source → destination in ``year``.

        Uses a flow-pruned BFS so it completes in milliseconds even on
        dense networks (replaces the exponential nx.all_simple_paths).

        Args:
            source:        Origin country (e.g. "China").
            destination:   Final destination (e.g. "USA").
            year:          Year of trade data.
            max_hops:      Maximum chain length.
            min_flow_pct:  Minimum bottleneck as fraction of source total.
                           Increase to filter out negligible routes.
            top_k:         Maximum number of paths to return.

        Returns:
            List of PathTrace objects sorted by bottleneck_t descending.
        """
        G = self.build_annual_network(year)
        if source not in G or destination not in G:
            return []

        source_total = sum(
            G[source][nb].get("weight", 0.0) for nb in G.successors(source)
        )

        raw = self._flow_pruned_paths(
            G, source, destination, source_total, max_hops, min_flow_pct, top_k=top_k
        )

        traces: List[PathTrace] = []
        for r in raw:
            path = r["path"]
            intermediaries = path[1:-1]
            non_prod = [c for c in intermediaries if c not in self.known_producers]
            traces.append(PathTrace(
                path=path,
                flows_t=r["flows_t"],
                flows_kusd=r["flows_kusd"],
                bottleneck_t=r["bottleneck_t"],
                source_total_t=source_total,
                pct_of_source=r["bottleneck_t"] / source_total if source_total > 0 else 0.0,
                hops=len(path) - 1,
                non_producer_intermediaries=non_prod,
                is_circumvention_candidate=len(non_prod) > 0,
            ))

        return traces

    # ------------------------------------------------------------------
    # 4. Mirror statistics discrepancy
    # ------------------------------------------------------------------

    def mirror_discrepancy(
        self,
        year_range: Optional[Tuple[int, int]] = None,
    ) -> pd.DataFrame:
        """
        Compare each bilateral flow's reported exports vs. reported imports.

        For pair (A, B):
            A reports exporting X tonnes to B
            B reports importing Y tonnes from A
            discrepancy = Y − X

        Positive discrepancy (B received more than A says it sent) can
        indicate: A underreports exports (tax/quota evasion), goods
        transshipped through unreported hops, or statistical noise.

        Negative discrepancy (A says it sent more than B acknowledges)
        can indicate: A overreports exports (subsidies), goods diverted
        en route, or smuggling.

        Args:
            year_range: (start_year, end_year) inclusive.  None = all years.

        Returns:
            DataFrame with columns: year, country_a, country_b,
            a_reports_export_t, b_reports_import_t, discrepancy_t,
            discrepancy_pct, abs_discrepancy_t

        Important: CEPII BACI data is pre-reconciled — both sides of each
        bilateral flow are averaged before publication, so discrepancies are
        near-zero by construction. This method is most useful with raw
        UN Comtrade data where the two sides of a flow are independently
        reported and may differ substantially.  On BACI data it primarily
        detects asymmetric coverage (one side reporting, other not).
        """
        df = self.df.copy()
        if year_range:
            df = df[df["year"].between(year_range[0], year_range[1])]

        # A's exports to B
        exp = (
            df.groupby(["year", "exporter", "importer"])["quantity_tonnes"]
            .sum()
            .reset_index()
            .rename(columns={
                "exporter": "country_a",
                "importer": "country_b",
                "quantity_tonnes": "a_reports_export_t",
            })
        )

        # B's imports from A (same physical flow, reported by the other side)
        imp = (
            df.groupby(["year", "importer", "exporter"])["quantity_tonnes"]
            .sum()
            .reset_index()
            .rename(columns={
                "importer": "country_b",
                "exporter": "country_a",
                "quantity_tonnes": "b_reports_import_t",
            })
        )

        merged = exp.merge(imp, on=["year", "country_a", "country_b"], how="outer").fillna(0.0)
        merged["discrepancy_t"] = (
            merged["b_reports_import_t"] - merged["a_reports_export_t"]
        )
        merged["abs_discrepancy_t"] = merged["discrepancy_t"].abs()
        merged["discrepancy_pct"] = merged["discrepancy_t"] / (
            merged["a_reports_export_t"].replace(0, np.nan)
        )
        return merged.sort_values("abs_discrepancy_t", ascending=False).reset_index(drop=True)

    # ------------------------------------------------------------------
    # 5. Detect rerouting: event study around restriction events
    # ------------------------------------------------------------------

    def _candidate_hubs(self, min_imports_from_dominant: float = 0.0) -> List[str]:
        """
        Return countries that:
        (a) import from the dominant exporter AND
        (b) are not known producers (so their exports must largely be re-exports).
        """
        dom_exportees = set(
            row["importer"]
            for _, row in self.df[self.df["exporter"] == self.dominant_exporter].iterrows()
        )
        return [
            c for c in dom_exportees
            if c not in self.known_producers and c != self.dominant_exporter
        ]

    def detect_rerouting(
        self,
        event_years: List[int],
        pre_window: int = 3,
        post_window: int = 3,
        min_annual_flow_t: float = 100.0,
        significance_level: float = 0.10,
    ) -> pd.DataFrame:
        """
        Event study: after each restriction event, do flows through non-producer
        hub countries increase significantly?

        Tests two signals per hub per event:
        1. ``dom_to_hub``  — dominant_exporter → hub flow (does China send more
           to Singapore after imposing restrictions, suggesting rerouting prep?)
        2. ``hub_to_world`` — hub → rest-of-world flow (does Singapore export
           more shortly after, suggesting it is re-exporting Chinese goods?)

        A hub is flagged as a likely transshipment channel if BOTH signals are
        significantly positive.

        Args:
            event_years:          Years when restrictions were imposed.
            pre_window:           Years of pre-event data (baseline).
            post_window:          Years of post-event data (treatment).
            min_annual_flow_t:    Minimum average annual flow to include a hub.
            significance_level:   p-value threshold for flagging significance.

        Returns:
            DataFrame of ReroutingSignal records; one row per (hub, event, metric).
        """
        hubs = self._candidate_hubs()
        annual = (
            self.df.groupby(["year", "exporter", "importer"])["quantity_tonnes"]
            .sum()
            .reset_index()
        )

        results: List[Dict] = []

        for event_year in event_years:
            pre_yrs = list(range(event_year - pre_window, event_year))
            post_yrs = list(range(event_year, event_year + post_window + 1))

            for hub in hubs:
                # ---- Signal 1: dominant_exporter → hub ----
                def _flows(exporter, importer, years):
                    mask = (
                        (annual["exporter"] == exporter) &
                        (annual["importer"] == importer) &
                        (annual["year"].isin(years))
                    )
                    return annual[mask]["quantity_tonnes"].values

                dom_hub_pre = _flows(self.dominant_exporter, hub, pre_yrs)
                dom_hub_post = _flows(self.dominant_exporter, hub, post_yrs)

                for metric, pre_vals, post_vals in [
                    ("dom_to_hub", dom_hub_pre, dom_hub_post),
                    ("hub_to_world",
                     self._hub_to_world(hub, pre_yrs, annual),
                     self._hub_to_world(hub, post_yrs, annual)),
                ]:
                    if len(pre_vals) == 0 and len(post_vals) == 0:
                        continue
                    pre_mean = float(np.mean(pre_vals)) if len(pre_vals) > 0 else 0.0
                    post_mean = float(np.mean(post_vals)) if len(post_vals) > 0 else 0.0

                    # Skip tiny flows
                    if max(pre_mean, post_mean) < min_annual_flow_t:
                        continue

                    pct_change = (
                        (post_mean - pre_mean) / pre_mean
                        if pre_mean > 1 else float("nan")
                    )

                    # Welch t-test (unequal variance, small samples)
                    if len(pre_vals) >= 2 and len(post_vals) >= 2:
                        t_stat, p_val = stats.ttest_ind(
                            post_vals, pre_vals, equal_var=False
                        )
                    else:
                        t_stat, p_val = float("nan"), 1.0

                    results.append({
                        "hub": hub,
                        "event_year": event_year,
                        "metric": metric,
                        "pre_mean_t": pre_mean,
                        "post_mean_t": post_mean,
                        "pct_change": pct_change,
                        "t_stat": t_stat,
                        "p_value": p_val,
                        "is_significant": (
                            p_val < significance_level and pct_change > 0
                        ),
                        "n_pre": len(pre_vals),
                        "n_post": len(post_vals),
                    })

        if not results:
            return pd.DataFrame()

        df_out = pd.DataFrame(results)
        return df_out.sort_values(["event_year", "p_value"]).reset_index(drop=True)

    def _hub_to_world(
        self,
        hub: str,
        years: List[int],
        annual: pd.DataFrame,
    ) -> np.ndarray:
        """Sum of hub's exports to all countries (excluding dominant_exporter) per year."""
        mask = (
            (annual["exporter"] == hub) &
            (annual["importer"] != self.dominant_exporter) &
            (annual["year"].isin(years))
        )
        per_year = (
            annual[mask]
            .groupby("year")["quantity_tonnes"]
            .sum()
            .reindex(years, fill_value=0.0)
        )
        return per_year.values

    # ------------------------------------------------------------------
    # 6. Circumvention rate estimation
    # ------------------------------------------------------------------

    def _block_bootstrap_ci(
        self,
        event_years: List[int],
        pre_window: int,
        sig_hubs: List[str],
        nominal_restricted_t: float,
        n_bootstrap: int = 500,
        confidence: float = 0.95,
    ) -> Tuple[float, float]:
        """
        Circular block bootstrap CI for the circumvention rate.

        Resampling individual event years is degenerate with a single
        restriction event.  Instead this method bootstraps over the annual
        hub-flow time series, preserving temporal autocorrelation.

        Block length follows the T^(1/3) rule of thumb (Politis & Romano 1994).
        Circular wrapping avoids boundary bias.

        Args:
            event_years:         Restriction event years.
            pre_window:          Pre-event window used for baseline.
            sig_hubs:            Hubs with significant rerouting signals.
            nominal_restricted_t: Denominator for the circumvention rate.
            n_bootstrap:         Number of bootstrap replicates.
            confidence:          Confidence level (default 0.95).

        Returns:
            (ci_lo, ci_hi) tuple.
        """
        if not sig_hubs or nominal_restricted_t <= 0:
            return (0.0, 0.0)

        # Build annual dom→hub flow series for significant hubs
        annual = (
            self.df.groupby(["year", "exporter", "importer"])["quantity_tonnes"]
            .sum()
            .reset_index()
        )
        all_years = sorted(self.df["year"].unique())
        T = len(all_years)
        if T < 4:
            return (0.0, 0.0)

        block_len = max(2, int(np.ceil(T ** (1.0 / 3.0))))
        rng = np.random.default_rng(42)

        # Build matrix: rows = years, cols = significant hubs
        hub_series = {}
        for hub in sig_hubs:
            mask = (annual["exporter"] == self.dominant_exporter) & (annual["importer"] == hub)
            yr_flow = annual[mask].set_index("year")["quantity_tonnes"]
            hub_series[hub] = np.array([float(yr_flow.get(y, 0.0)) for y in all_years])

        # Event year indices
        event_idxs = [all_years.index(ev) for ev in event_years if ev in all_years]
        pre_idxs_sets = [
            [all_years.index(y) for y in range(ev - pre_window, ev) if y in all_years]
            for ev in event_years
        ]

        boot_rates = []
        for _ in range(n_bootstrap):
            # Draw circular blocks to cover T years
            n_blocks = int(np.ceil(T / block_len))
            starts = rng.integers(0, T, size=n_blocks)
            indices = []
            for s in starts:
                indices.extend([(s + j) % T for j in range(block_len)])
            indices = indices[:T]

            # Compute rerouted_t under the resampled series
            b_detected = 0.0
            for hub, series in hub_series.items():
                boot_series = series[indices]
                # Map original indices back to resample for pre/post
                post_vals = np.array([boot_series[i] for i in event_idxs if i < T])
                pre_vals = np.concatenate([
                    np.array([boot_series[i] for i in pre_set if i < T])
                    for pre_set in pre_idxs_sets
                ])
                if len(pre_vals) == 0:
                    continue
                pre_mean = float(np.mean(pre_vals))
                post_mean = float(np.mean(post_vals)) if len(post_vals) > 0 else pre_mean
                b_detected += max(0.0, post_mean - pre_mean)

            boot_rates.append(min(b_detected / nominal_restricted_t, 1.0))

        alpha = 1.0 - confidence
        ci_lo = float(np.percentile(boot_rates, alpha / 2 * 100))
        ci_hi = float(np.percentile(boot_rates, (1 - alpha / 2) * 100))
        return (ci_lo, ci_hi)

    def estimate_circumvention_rate(
        self,
        event_years: List[int],
        nominal_restriction: float = 1.0,
        pre_window: int = 3,
        significance_level: float = 0.10,
        n_bootstrap: int = 500,
    ) -> CircumventionEstimate:
        """
        Estimate the fraction of a nominal export restriction that is
        actually evaded via transshipment through third countries.

        Method:
          For each significant hub (rerouting detected), compute the
          incremental flow through that hub after the restriction:
              rerouted_t = max(0, post_mean_t − pre_baseline_t)
          Aggregate across hubs and normalise by the nominal restricted volume.

          Bootstrap CI is computed by resampling annual observations.

        Args:
            event_years:         Years of restriction events.
            nominal_restriction: Fraction of dominant_exporter supply nominally
                                 restricted (e.g. 0.30 for a 30% quota).
                                 Used to compute the denominator.
            pre_window:          Years before event for baseline.
            significance_level:  p-value threshold for hub inclusion.
            n_bootstrap:         Resamples for CI.

        Returns:
            CircumventionEstimate with rate, CI, and significant hubs.
        """
        rerouting = self.detect_rerouting(
            event_years=event_years,
            pre_window=pre_window,
            significance_level=significance_level,
        )

        # Dominant exporter's baseline supply (pre-restriction)
        pre_years = [
            y for ev in event_years
            for y in range(ev - pre_window, ev)
        ]
        dom_baseline = (
            self.df[
                (self.df["exporter"] == self.dominant_exporter) &
                (self.df["year"].isin(pre_years))
            ]["quantity_tonnes"]
            .sum() / max(len(pre_years), 1)  # annual average
        )
        nominal_restricted_t = dom_baseline * nominal_restriction

        sig_hubs: List[str] = []
        detected_t = 0.0

        if not rerouting.empty:
            # Only count "dom_to_hub" signal (direct evidence of rerouting from source)
            dom_sig = rerouting[
                (rerouting["metric"] == "dom_to_hub") &
                (rerouting["is_significant"])
            ]
            sig_hubs = list(dom_sig["hub"].unique())
            # Incremental rerouted volume = sum of (post_mean - pre_mean) for significant hubs
            detected_t = float(
                (dom_sig["post_mean_t"] - dom_sig["pre_mean_t"])
                .clip(lower=0)
                .sum()
            )

        rate = detected_t / nominal_restricted_t if nominal_restricted_t > 0 else 0.0
        rate = min(rate, 1.0)  # cap at 100%

        # Circular block bootstrap CI on the hub flow time series.
        #
        # Resampling over event_years is degenerate when there is only one
        # restriction event (the same year is always resampled, giving a
        # zero-width CI).  Instead we bootstrap over the annual time series
        # of dom→hub flows, preserving temporal autocorrelation via blocks
        # of length L = ceil(T^(1/3)) (standard rule of thumb).
        ci_lo, ci_hi = self._block_bootstrap_ci(
            event_years=event_years,
            pre_window=pre_window,
            sig_hubs=sig_hubs,
            nominal_restricted_t=nominal_restricted_t,
            n_bootstrap=n_bootstrap,
        )

        notes = []
        if nominal_restricted_t < 1:
            notes.append("Nominal restricted volume near zero — check dominant_exporter and event_years.")
        if not sig_hubs:
            notes.append(
                "No statistically significant rerouting hubs detected. "
                "Possible causes: short pre/post windows, data sparsity, "
                "or circumvention occurs below detection threshold. "
                "Use literature prior: ~0.20-0.40 for China restriction events."
            )
        if rate > 0.5:
            notes.append(
                f"High circumvention rate ({rate:.0%}) — verify hub list and "
                "ensure significant hubs are not legitimate producers."
            )

        return CircumventionEstimate(
            event_years=event_years,
            nominal_restriction_t=nominal_restricted_t,
            detected_rerouted_t=detected_t,
            circumvention_rate=rate,
            circumvention_rate_ci=(ci_lo, ci_hi),
            significant_hubs=sig_hubs,
            notes=notes,
        )

    # ------------------------------------------------------------------
    # 7. Corrected supply series (for parameter fitting)
    # ------------------------------------------------------------------

    def corrected_dom_supply(
        self,
        event_years: List[int],
        circumvention_rate: Optional[float] = None,
        pre_window: int = 3,
    ) -> pd.DataFrame:
        """
        Return a corrected annual supply series for the dominant exporter,
        adjusting reported exports upward to account for circumvented flows.

        The idea: if China's 2023 export controls were 30% nominal but 25%
        actually circumvented, China's *effective* supply reduction was only
        5%.  Reported CEPII flows show China exporting less (the nominal cut),
        but the circumvented volume reappears as Vietnamese/Singaporean exports.

        Correction:
            corrected_supply_t = reported_supply_t + detected_rerouted_t

        This produces a more accurate instrument for IV estimation of eta_D.

        Args:
            event_years:         Restriction event years.
            circumvention_rate:  Override rate. If None, estimated from data.
            pre_window:          Pre-event window for baseline.

        Returns:
            DataFrame: year, reported_supply_t, rerouted_adjustment_t,
                       corrected_supply_t, circumvention_applied
        """
        annual_dom = (
            self.df[self.df["exporter"] == self.dominant_exporter]
            .groupby("year")["quantity_tonnes"]
            .sum()
            .reset_index()
            .rename(columns={"quantity_tonnes": "reported_supply_t"})
        )

        if circumvention_rate is None:
            est = self.estimate_circumvention_rate(
                event_years=event_years,
                pre_window=pre_window,
            )
            circumvention_rate = est.circumvention_rate
            logger.info(
                f"Estimated circumvention rate: {circumvention_rate:.1%} "
                f"(hubs: {est.significant_hubs})"
            )

        # Baseline supply (pre-restriction)
        all_pre = [y for ev in event_years for y in range(ev - pre_window, ev)]
        pre_df = annual_dom[annual_dom["year"].isin(all_pre)]
        baseline_t = float(pre_df["reported_supply_t"].mean()) if len(pre_df) > 0 else 0.0

        def _adjustment(row):
            if row["year"] in event_years or any(
                row["year"] > ev for ev in event_years
            ):
                # Reported supply is suppressed; add back the circumvented fraction
                gap = max(0.0, baseline_t - row["reported_supply_t"])
                return gap * circumvention_rate
            return 0.0

        annual_dom["rerouted_adjustment_t"] = annual_dom.apply(_adjustment, axis=1)
        annual_dom["corrected_supply_t"] = (
            annual_dom["reported_supply_t"] + annual_dom["rerouted_adjustment_t"]
        )
        annual_dom["circumvention_applied"] = circumvention_rate

        return annual_dom

    # ------------------------------------------------------------------
    # 8. Summary report
    # ------------------------------------------------------------------

    def summary_report(
        self,
        destination: str,
        event_years: List[int],
        year: Optional[int] = None,
        max_hops: int = 4,
        nominal_restriction: float = 0.30,
    ) -> str:
        """
        Human-readable report: where does the dominant exporter's product
        flow to ``destination``, and how much goes via transshipment?

        Args:
            nominal_restriction: Fraction of dominant_exporter supply nominally
                restricted (e.g. 0.30 for a 30% quota).  Used as the
                circumvention rate denominator.  Defaults to 0.30.
        """
        year = year or (event_years[-1] if event_years else self.df["year"].max())
        # min_flow_pct=0.005 (0.5% of source exports) filters network noise:
        # paths below this threshold are real bilateral flows but too small
        # to represent meaningful commercial routing decisions.
        paths = self.trace_paths(
            self.dominant_exporter, destination, year=year, max_hops=max_hops,
            min_flow_pct=0.005,
        )
        ce = self.estimate_circumvention_rate(event_years, nominal_restriction=nominal_restriction)

        lines = [
            f"=== Transshipment Report: {self.dominant_exporter} → {destination} ({year}) ===",
            f"Commodity: {self.commodity}",
            "",
            f"Top trade routes ({self.dominant_exporter} → {destination}):",
        ]
        if not paths:
            lines.append(f"  No paths found within {max_hops} hops.")
        for i, pt in enumerate(paths[:8], 1):
            flag = " [TRANSSHIPMENT CANDIDATE]" if pt.is_circumvention_candidate else ""
            lines.append(
                f"  {i}. {' → '.join(pt.path)}{flag}"
            )
            lines.append(
                f"     Bottleneck: {pt.bottleneck_t:,.0f}t  "
                f"({pt.pct_of_source*100:.1f}% of source exports)"
            )
            if pt.non_producer_intermediaries:
                lines.append(
                    f"     Non-producer intermediaries: {pt.non_producer_intermediaries}"
                )

        lines += [
            "",
            f"Circumvention estimate (events: {event_years}):",
            f"  Detected circumvention rate: {ce.circumvention_rate:.1%}  "
            f"CI [{ce.circumvention_rate_ci[0]:.1%}, {ce.circumvention_rate_ci[1]:.1%}]",
            f"  Significant rerouting hubs: {ce.significant_hubs or 'none detected'}",
            f"  Nominal restricted volume: {ce.nominal_restriction_t:,.0f}t/yr",
            f"  Detected rerouted volume:  {ce.detected_rerouted_t:,.0f}t/yr",
        ]
        for note in ce.notes:
            lines.append(f"  Note: {note}")

        return "\n".join(lines)
