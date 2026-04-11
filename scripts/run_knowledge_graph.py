#!/usr/bin/env python3
"""Run the critical minerals knowledge graph: build KG, print summary, optional shock propagation."""

import argparse
import sys
from pathlib import Path

# Project root on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _build_kg_dag_layout(kg, max_nodes: int | None):
    """Build the DAG graph, layout positions, and node categories. Returns (G, pos, node_categories, category_colors)."""
    import networkx as nx
    from src.minerals.knowledge_graph import EntityType

    dag = kg.to_causal_dag()
    G_full = dag.graph
    if G_full.number_of_nodes() == 0:
        return None

    G = G_full
    if max_nodes and G_full.number_of_nodes() > max_nodes:
        try:
            gens = list(nx.topological_generations(G_full))
            n_keep_bottom = 4
            keep = set()
            for layer in gens[-n_keep_bottom:]:
                keep.update(layer)
            rest = [n for layer in gens[:-n_keep_bottom] for n in layer]
            out_deg = dict(G_full.out_degree())
            rest.sort(key=lambda n: -out_deg.get(n, 0))
            for n in rest:
                if len(keep) >= max_nodes:
                    break
                keep.add(n)
            G = G_full.subgraph(keep).copy()
        except Exception:
            G = G_full

    category_colors = {
        "policy_event": "#E74C3C",
        "economic": "#F39C12",
        "commodity": "#3498DB",
        "industry": "#2ECC71",
        "other": "#95A5A6",
    }
    type_to_category = {
        EntityType.POLICY: "policy_event",
        EntityType.EVENT: "policy_event",
        EntityType.ECONOMIC_INDICATOR: "economic",
        EntityType.COMMODITY: "commodity",
        EntityType.INDUSTRY: "industry",
        EntityType.TECHNOLOGY: "industry",
    }
    node_categories = {}
    for node in G.nodes():
        entity = kg.get_entity(node)
        node_categories[node] = type_to_category.get(entity.entity_type, "other") if entity else "other"

    try:
        generations = list(nx.topological_generations(G))
    except nx.NetworkXError:
        generations = None

    if generations:
        pos = {}
        max_layer_width = max(len(layer) for layer in generations)
        n_layers = len(generations)
        h_spacing, v_spacing = 1.0, 1.2
        stagger_offset, min_node_gap = 0.25, 0.75
        x_scale = max(max_layer_width * h_spacing, max_layer_width * min_node_gap)
        for layer_idx, layer in enumerate(generations):
            y_base = (n_layers - 1 - layer_idx) * v_spacing
            sorted_layer = sorted(layer)
            n = len(sorted_layer)
            effective_spacing = max(min_node_gap, x_scale / max(n, 1))
            layer_width = n * effective_spacing
            x_start = (x_scale - layer_width) / 2
            for i, node in enumerate(sorted_layer):
                x = x_start + (i + 0.5) * effective_spacing
                y = y_base + (stagger_offset if (i % 2 == 1 and n > 8) else 0)
                pos[node] = (x, y)
    else:
        pos = nx.kamada_kawai_layout(G)

    return G, pos, node_categories, category_colors


def get_kg_dag_interactive_data(kg, max_nodes: int = 42) -> dict | None:
    """Return graph data for the interactive viewer: nodes (id, label, x, y, color) and edges (from, to)."""
    result = _build_kg_dag_layout(kg, max_nodes)
    if result is None:
        return None
    G, pos, node_categories, category_colors = result
    # Scale layout to ~600x500 viewport; vis-network uses pixels-like units when fixed
    xs = [pos[n][0] for n in G.nodes()]
    ys = [pos[n][1] for n in G.nodes()]
    scale = 120
    ox = (max(xs) + min(xs)) / 2 if xs else 0
    oy = (max(ys) + min(ys)) / 2 if ys else 0
    nodes = []
    for n in G.nodes():
        x, y = pos[n]
        nodes.append({
            "id": n,
            "label": n.replace("_", " ")[:30] + ("…" if len(n) > 30 else ""),
            "x": (x - ox) * scale,
            "y": (y - oy) * scale,
            "color": category_colors[node_categories[n]],
        })
    edges = [{"from": u, "to": v} for u, v in G.edges()]
    return {"nodes": nodes, "edges": edges}


def visualize_kg_dag(kg, output_path: str, max_nodes: int = 42) -> None:
    """Render the KG-derived causal DAG. If max_nodes is set, show a simplified subgraph for legibility."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import networkx as nx

    result = _build_kg_dag_layout(kg, max_nodes)
    if result is None:
        print("Empty graph, nothing to draw")
        return
    G, pos, node_categories, category_colors = result

    # --- Figure sizing (scale up for legibility) ---
    n_nodes = G.number_of_nodes()
    fig_w = max(48, n_nodes * 1.4)
    fig_h = max(32, n_nodes * 0.9)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_facecolor("#FAFAFA")
    fig.patch.set_facecolor("#FAFAFA")

    node_size = 2400
    node_colors = [category_colors[node_categories[n]] for n in G.nodes()]

    # --- Edges ---
    edge_colors = [
        category_colors[node_categories[u]] for u, _v in G.edges()
    ]
    nx.draw_networkx_edges(
        G, pos, ax=ax,
        edge_color=edge_colors,
        alpha=0.5,
        arrows=True,
        arrowsize=22,
        arrowstyle="-|>",
        connectionstyle="arc3,rad=0.08",
        width=1.6,
        min_source_margin=24,
        min_target_margin=24,
    )

    # --- Nodes ---
    nx.draw_networkx_nodes(
        G, pos, ax=ax,
        node_color=node_colors,
        node_size=node_size,
        alpha=0.92,
        edgecolors="#2c3e50",
        linewidths=1.5,
    )

    # --- Labels: abbreviate long names for legibility ---
    max_label_len = 24
    def _label_text(name: str) -> str:
        s = name.replace("_", " ")
        if len(s) > max_label_len:
            return s[: max_label_len - 1].rstrip() + "…"
        return s

    label_pos = {k: (x, y - 0.04) for k, (x, y) in pos.items()}
    labels = {n: _label_text(n) for n in G.nodes()}
    nx.draw_networkx_labels(
        G, label_pos, labels, ax=ax,
        font_size=13,
        font_weight="bold",
        font_family="sans-serif",
        verticalalignment="top",
    )

    # --- Legend (larger font) ---
    legend_items = [
        mpatches.Patch(color=category_colors["policy_event"], label="Policy / Event"),
        mpatches.Patch(color=category_colors["economic"], label="Economic"),
        mpatches.Patch(color=category_colors["commodity"], label="Commodity"),
        mpatches.Patch(color=category_colors["industry"], label="Industry / Tech"),
        mpatches.Patch(color=category_colors["other"], label="Other"),
    ]
    ax.legend(handles=legend_items, loc="upper left", fontsize=16, framealpha=0.95)
    ax.set_title("Critical Minerals Causal Knowledge Graph", fontsize=24, fontweight="bold", pad=24)
    ax.margins(0.1)
    ax.axis("off")
    plt.tight_layout()

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out), bbox_inches="tight", dpi=280, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"DAG saved to {out.resolve()}")


def main():
    parser = argparse.ArgumentParser(
        description="Build and run the critical minerals knowledge graph (summary, shock propagation, DAG)."
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        default=True,
        help="Print KG summary (default: True).",
    )
    parser.add_argument(
        "--no-summary",
        action="store_false",
        dest="summary",
        help="Skip summary.",
    )
    parser.add_argument(
        "--shock",
        type=str,
        metavar="ORIGIN_ID",
        help="Propagate shock from this entity (e.g. china_export_controls, graphite).",
    )
    parser.add_argument(
        "--dag-edges",
        action="store_true",
        help="Print KG-derived causal DAG edges.",
    )
    parser.add_argument(
        "--dag-image",
        type=str,
        metavar="PATH",
        help="Write KG-derived DAG visualization to PNG path (e.g. kg_dag.png).",
    )
    parser.add_argument(
        "--list-origins",
        action="store_true",
        help="List entity IDs that can be used as shock origins.",
    )
    args = parser.parse_args()

    from src.minerals.knowledge_graph import build_critical_minerals_kg

    kg = build_critical_minerals_kg()
    print("Built critical minerals knowledge graph.\n")

    if args.summary:
        print(kg.summary())
        print()

    if args.list_origins:
        origins = kg.get_shock_origin_candidates()
        print("Shock origin candidates (entity IDs):")
        for o in origins:
            print(f"  - {o}")
        print()

    if args.shock:
        trace = kg.propagate_shock(
            args.shock.strip(), initial_magnitude=1.0, decay=0.5, max_depth=5
        )
        print(f"Shock origin: {trace.origin}")
        print(f"Affected entities ({len(trace.affected)}):")
        for eid, mag in sorted(trace.affected.items(), key=lambda x: -x[1]):
            path = trace.paths.get(eid, [])
            path_str = " -> ".join(path) if path else eid
            print(f"  - {eid} (impact {mag:.3f}): path {path_str}")
        print()

    if args.dag_edges:
        dag = kg.to_causal_dag()
        edges = list(dag.graph.edges())
        print("KG-derived causal DAG edges (cause -> effect):")
        for u, v in sorted(edges):
            print(f"  {u} -> {v}")
        print()

    if args.dag_image:
        visualize_kg_dag(kg, args.dag_image)


if __name__ == "__main__":
    main()
