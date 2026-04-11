"""
Multi-layer supply chain network for critical minerals.

Builds country-level trade graphs from bilateral data and supports:
- Centrality (critical nodes)
- Alternative paths when nodes are blocked
- Shock propagation (direct/indirect)

Integration with causal model:
  Simple DAG:  ExportPolicy → Supply → Shortage → Price
  With network: ExportPolicy_China →
    ├→ Supply_China→USA (direct)
    ├→ Supply_China→Mexico → Supply_Mexico→USA (indirect)
    └→ Supply_China→Japan → embedded supply → USA

  Causal query: P(USA_Shortage | do(China_ExportPolicy))
    = P_direct + P_indirect + P_embedded (network propagation)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import pandas as pd


class GlobalSupplyChainNetwork:
    """
    Multi-layer supply chain network for critical minerals.
    One directed graph per mineral (countries = nodes, trade flows = edges).
    """

    def __init__(self) -> None:
        self.networks: Dict[str, nx.DiGraph] = {
            "graphite": nx.DiGraph(),
            "lithium": nx.DiGraph(),
            "rare_earths": nx.DiGraph(),
            "copper": nx.DiGraph(),
        }
        self.trade_data: Dict[str, pd.DataFrame] = {}

    def load_trade_data(
        self,
        mineral: str,
        trade_csv: str,
        reporter_col: str = "reporter",
        partner_col: str = "partner",
        year_col: str = "year",
        value_col: str = "trade_value",
        quantity_col: Optional[str] = "quantity",
    ) -> nx.DiGraph:
        """
        Load global trade matrix into the mineral's network.

        CSV format (column names configurable):
          reporter, partner, year, trade_value, quantity
          CHN, USA, 2023, 50000000, 12000
          ...
        """
        df = pd.read_csv(trade_csv)

        # Allow alternate column names
        if value_col not in df.columns and "trade_value_usd" in df.columns:
            value_col = "trade_value_usd"
        if quantity_col is not None and quantity_col not in df.columns:
            quantity_col = None

        if mineral not in self.networks:
            self.networks[mineral] = nx.DiGraph()

        G = self.networks[mineral]

        # Node columns
        rcol = reporter_col if reporter_col in df.columns else df.columns[0]
        pcol = partner_col if partner_col in df.columns else df.columns[1]
        countries = set(df[rcol].dropna().astype(str)) | set(df[pcol].dropna().astype(str))
        G.add_nodes_from(countries)

        ycol = year_col if year_col in df.columns else None
        qcol = quantity_col

        for _, row in df.iterrows():
            u, v = str(row[rcol]), str(row[pcol])
            weight = float(row[value_col]) if row[value_col] == row[value_col] else 0.0
            edge_data: Dict[str, Any] = {"weight": weight}
            if ycol is not None:
                edge_data["year"] = int(row[ycol])
            if qcol and qcol in df.columns:
                edge_data["quantity"] = row[qcol]
            G.add_edge(u, v, **edge_data)

        self.trade_data[mineral] = df

        print(f"✅ Loaded {mineral} network:")
        print(f"   Nodes (countries): {G.number_of_nodes()}")
        print(f"   Edges (trade flows): {G.number_of_edges()}")

        return G

    def analyze_centrality(self, mineral: str) -> Dict[str, Dict[str, float]]:
        """Find critical nodes (degree, betweenness, pagerank, eigenvector)."""
        G = self.networks[mineral]
        if G.number_of_edges() == 0:
            return {"degree": {}, "betweenness": {}, "pagerank": {}, "eigenvector": {}}

        weight = "weight" if nx.get_edge_attributes(G, "weight") else None

        centrality: Dict[str, Dict[str, float]] = {
            "degree": dict(nx.degree_centrality(G)),
            "betweenness": dict(nx.betweenness_centrality(G, weight=weight)),
            "pagerank": dict(nx.pagerank(G, weight=weight)),
        }

        try:
            centrality["eigenvector"] = dict(
                nx.eigenvector_centrality(G, weight=weight, max_iter=500)
            )
        except (nx.PowerIterationFailedConvergence, nx.NetworkXError):
            centrality["eigenvector"] = {n: 0.0 for n in G.nodes()}

        return centrality

    def find_alternative_paths(
        self,
        mineral: str,
        source: str,
        target: str,
        blocked_nodes: Optional[List[str]] = None,
        cutoff: int = 3,
    ) -> List[Tuple[List[str], float]]:
        """
        Find alternative supply routes if key country(ies) are blocked.

        Example: If China is blocked, what are USA's alternative paths to supply?
        Returns list of (path, min_capacity) sorted by capacity descending.
        """
        G = self.networks[mineral].copy()
        if blocked_nodes:
            G.remove_nodes_from(blocked_nodes)

        if source not in G or target not in G:
            return []

        try:
            paths = list(nx.all_simple_paths(G, source, target, cutoff=cutoff))
        except nx.NetworkXNoPath:
            return []

        path_capacities: List[Tuple[List[str], float]] = []
        for path in paths:
            try:
                capacity = min(
                    G[path[i]][path[i + 1]].get("weight", 0.0)
                    for i in range(len(path) - 1)
                )
                path_capacities.append((path, capacity))
            except (KeyError, IndexError):
                continue

        path_capacities.sort(key=lambda x: x[1], reverse=True)
        return path_capacities

    def simulate_shock(
        self,
        mineral: str,
        shock_country: str,
        reduction_pct: float,
        cascade: bool = True,
        max_hops: int = 4,
    ) -> List[Tuple[str, str, float, float]]:
        """
        Simulate a supply shock propagating from one country through the network.

        Direct effect: The shock_country's exports are reduced by reduction_pct.

        Cascading effect (cascade=True): Countries that import from shock_country
        receive less supply and proportionally reduce their own exports downstream.
        The cascading reduction for a downstream node D is:

            cascade_reduction(D) = upstream_reduction(D) × (import_from_upstream / total_imports_of_D)

        This models supply-chain dependency: if Japan gets 60% of its graphite from
        China and China cuts exports 40%, Japan's supply drops by 24%, and Japan
        then cuts its own downstream exports by 24%.

        Returns list of (source, target, original_weight, new_weight) for all
        affected edges, with hop distance in a separate dict available via
        the returned list (hop=0 for direct, hop=1+ for cascaded).
        """
        G = self.networks[mineral]
        affected_edges: List[Tuple[str, str, float, float]] = []

        # node -> fractional reduction applied to its exports (0..1)
        node_reduction: dict[str, float] = {shock_country: reduction_pct}

        # BFS through the network
        from collections import deque
        visited: set[str] = {shock_country}
        queue: deque = deque([(shock_country, 0)])

        while queue:
            current, hop = queue.popleft()
            if hop >= max_hops:
                continue
            current_red = node_reduction[current]

            for u, v, data in G.edges(data=True):
                if u != current:
                    continue
                original = data.get("weight", 0.0)
                new_value = original * (1.0 - current_red)
                affected_edges.append((u, v, original, new_value))

                if not cascade or v in visited:
                    continue

                # Compute v's cascade reduction:
                # fraction of v's total imports that came from u, times u's reduction
                total_imports_v = sum(
                    G[src][v].get("weight", 0.0)
                    for src in G.predecessors(v)
                )
                import_from_u = original
                if total_imports_v > 0:
                    dependency = import_from_u / total_imports_v
                    cascade_red = current_red * dependency
                else:
                    cascade_red = 0.0

                if cascade_red > 0.001:  # ignore negligible cascades
                    node_reduction[v] = max(node_reduction.get(v, 0.0), cascade_red)
                    visited.add(v)
                    queue.append((v, hop + 1))

        return affected_edges

    def causal_network_effect(
        self,
        mineral: str,
        shock_country: str,
        target_country: str,
        reduction_pct: float,
        dag=None,
        cutoff: int = 4,
    ) -> Dict:
        """
        Compute P(target_shortage | do(shock_country_policy = reduction_pct)).

        Combines the trade network (who imports from whom, with what weights)
        with optional DAG identifiability to give the full causal picture:

          Total effect = direct path effect + sum of indirect path effects

        Each path's contribution is weighted by the minimum edge capacity
        (bottleneck) along the path, normalised by the target's total imports.

        Args:
            mineral:        Mineral network to query (e.g. "graphite").
            shock_country:  Country applying the export restriction (e.g. "China").
            target_country: Country whose supply shortage we care about (e.g. "USA").
            reduction_pct:  Fraction of shock_country's exports cut (0..1).
            dag:            Optional CausalDAG (GraphiteSupplyChainDAG) for
                            do-calculus identifiability metadata.
            cutoff:         Max path length to consider.

        Returns:
            Dict with keys:
              direct_effect        – fractional shortage from direct trade
              indirect_effect      – fractional shortage via intermediaries
              total_effect         – direct + indirect (fraction of target imports)
              total_effect_pct     – total_effect * 100
              paths                – list of (path, weight, contribution) tuples
              identifiability      – IdentificationResult (if dag provided)
              alternative_paths    – paths still available after shock
        """
        G = self.networks.get(mineral)
        if G is None or G.number_of_nodes() == 0:
            return {
                "direct_effect": 0.0, "indirect_effect": 0.0,
                "total_effect": 0.0, "total_effect_pct": 0.0,
                "paths": [], "identifiability": None, "alternative_paths": [],
            }

        # Total imports of target (denominator for normalisation)
        total_imports = sum(
            G[src][target_country].get("weight", 0.0)
            for src in G.predecessors(target_country)
        ) if target_country in G else 0.0

        # All simple paths from shock_country → target_country within cutoff
        try:
            all_paths = list(nx.all_simple_paths(G, shock_country, target_country, cutoff=cutoff))
        except (nx.NetworkXNoPath, nx.NodeNotFound, nx.NetworkXError):
            all_paths = []

        path_details = []
        direct_effect = 0.0
        indirect_effect = 0.0

        for path in all_paths:
            # Bottleneck capacity along this path
            capacity = min(
                G[path[i]][path[i + 1]].get("weight", 0.0)
                for i in range(len(path) - 1)
            )
            # Contribution = reduction_pct * (path capacity / total imports)
            contribution = (reduction_pct * capacity / total_imports) if total_imports > 0 else 0.0
            path_details.append((path, capacity, contribution))

            if len(path) == 2:  # direct: shock_country → target_country
                direct_effect += contribution
            else:               # indirect: through intermediary countries
                indirect_effect += contribution

        total_effect = direct_effect + indirect_effect

        # Remaining direct suppliers of target_country that bypass shock_country
        alt_paths: List[Tuple[List[str], float]] = []
        if target_country in G:
            for src in G.predecessors(target_country):
                if src == shock_country:
                    continue
                w = G[src][target_country].get("weight", 0.0)
                alt_paths.append(([src, target_country], w))
            alt_paths.sort(key=lambda x: -x[1])

        # DAG identifiability check
        id_result = None
        if dag is not None:
            try:
                id_result = dag.is_identifiable("ExportPolicy", "Price")
            except Exception:
                pass

        return {
            "direct_effect": direct_effect,
            "indirect_effect": indirect_effect,
            "total_effect": total_effect,
            "total_effect_pct": total_effect * 100,
            "paths": sorted(path_details, key=lambda x: -x[2]),
            "identifiability": id_result,
            "alternative_paths": alt_paths,
        }

    def visualize(
        self,
        mineral: str,
        highlight_country: Optional[str] = None,
        output_path: Optional[str] = None,
    ) -> None:
        """Visualize supply chain network (requires matplotlib)."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("⚠️  matplotlib required for visualize()")
            return

        G = self.networks[mineral]
        if G.number_of_nodes() == 0:
            print("⚠️  Empty graph, nothing to draw")
            return

        pos = nx.spring_layout(G, k=0.5, iterations=50, seed=42)

        node_sizes = []
        for node in G.nodes():
            total = sum(
                G[u][v].get("weight", 0.0)
                for u, v in G.edges() if u == node or v == node
            )
            node_sizes.append(max(total / 1e6, 100))

        out = output_path or f"outputs/{mineral}_network.png"
        Path(out).parent.mkdir(parents=True, exist_ok=True)

        plt.figure(figsize=(16, 12))

        nx.draw_networkx_nodes(
            G, pos,
            node_size=node_sizes,
            node_color="lightblue",
            alpha=0.7,
        )

        if highlight_country and highlight_country in G:
            nx.draw_networkx_nodes(
                G, pos,
                nodelist=[highlight_country],
                node_size=1200,
                node_color="red",
            )

        edge_widths = [data.get("weight", 0) / 1e7 for _, _, data in G.edges(data=True)]
        nx.draw_networkx_edges(
            G, pos,
            width=[max(w, 0.1) for w in edge_widths],
            alpha=0.3,
            arrows=True,
            arrowsize=10,
        )

        nx.draw_networkx_labels(G, pos, font_size=8)

        plt.title(f"{mineral.replace('_', ' ').title()} Global Supply Chain Network")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(out, dpi=150)
        plt.close()
        print(f"✅ Saved {out}")


# Main execution
if __name__ == "__main__":
    print("=" * 70)
    print("GLOBAL SUPPLY CHAIN NETWORK ANALYSIS")
    print("=" * 70)

    data_path = Path("data/canonical/comtrade_network_graphite.csv")

    if not data_path.exists():
        print(f"❌ Data file not found: {data_path}")
        print("   Run the Comtrade download script first!")
        raise SystemExit(1)

    net = GlobalSupplyChainNetwork()

    net.load_trade_data(
        "graphite",
        str(data_path),
        value_col="trade_value_usd",
    )

    centrality = net.analyze_centrality("graphite")
    if centrality.get("degree"):
        print("\n📊 Top nodes by degree centrality:")
        top = sorted(centrality["degree"].items(), key=lambda x: -x[1])[:5]
        for node, val in top:
            print(f"   {node}: {val:.3f}")

    G = net.networks["graphite"]
    if "China" in G and "USA" in G:
        paths = net.find_alternative_paths("graphite", "China", "USA", cutoff=4)
        print(f"\n🔀 Alternative paths China → USA: {len(paths)}")
        for path, cap in paths[:3]:
            print(f"   {' → '.join(path)} (min capacity: {cap:,.0f})")
    else:
        print("\n⚠️  China or USA not in network (or no edges between them)")
        nodes = list(G.nodes())[:15]
        print(f"   Sample nodes: {', '.join(sorted(nodes))}")

    if "China" in G:
        affected = net.simulate_shock("graphite", "China", 0.4)
        print(f"\n⚡ China 40% export reduction: {len(affected)} edges affected")

    print("\n✅ Network analysis complete!")
