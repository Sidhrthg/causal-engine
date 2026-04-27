#!/usr/bin/env python3
"""
Knowledge graph scenario visualiser.

For each scenario:
  1. Query HippoRAG → retrieve top-k document chunks
  2. Scan chunk text for KG entity mentions → focal nodes (no hardcoded lists)
  3. query_at_time(year)       → year-specific edge shares
  4. propagate_shock(origin)   → impact magnitude per node
  5. effective_control_at()    → binding constraint annotation
  6. Render subgraph PNG

Usage
-----
  # All validation episodes (historical)
  python scripts/run_knowledge_graph.py --enriched --validation

  # All predictive scenarios (2027-2035)
  python scripts/run_knowledge_graph.py --enriched --predictive

  # Everything
  python scripts/run_knowledge_graph.py --enriched --all-scenarios

  # One episode
  python scripts/run_knowledge_graph.py --enriched --scenario graphite_2022

  # Full KG overview PNG
  python scripts/run_knowledge_graph.py --enriched --dag-image outputs/kg_full.png
"""

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ── Scenario definitions ──────────────────────────────────────────────────────
# hipporag_query: what to ask HippoRAG — drives which document chunks are
#   retrieved and therefore which KG entities appear in the subgraph.
# shock_origin / commodity: always included as anchors regardless of retrieval.

VALIDATION_SCENARIOS = {
    "graphite_2008":    {"year": 2008, "shock_origin": "china",     "commodity": "graphite",
                         "title": "Graphite 2008 — China Export Controls (Pre-EV Era)"},
    "graphite_2022":    {"year": 2022, "shock_origin": "china",     "commodity": "graphite",
                         "title": "Graphite 2022 — China Anode Processing Dominance"},
    "rare_earths_2010": {"year": 2010, "shock_origin": "china",     "commodity": "rare_earths",
                         "title": "Rare Earths 2010 — China Export Quota Crisis"},
    "lithium_2016":     {"year": 2016, "shock_origin": "chile",     "commodity": "lithium",
                         "title": "Lithium 2016 — Atacama Supply Surge"},
    "lithium_2022":     {"year": 2022, "shock_origin": "china",     "commodity": "lithium",
                         "title": "Lithium 2022 — China Battery-Grade Processing Lock-in"},
    "cobalt_2016":      {"year": 2016, "shock_origin": "drc",       "commodity": "cobalt",
                         "title": "Cobalt 2016 — DRC Artisanal Mining Concentration"},
    "cobalt_2022":      {"year": 2022, "shock_origin": "drc",       "commodity": "cobalt",
                         "title": "Cobalt 2022 — Post-COVID Supply Recovery"},
    "nickel_2006":      {"year": 2006, "shock_origin": "russia",    "commodity": "nickel",
                         "title": "Nickel 2006 — Norilsk / LME Squeeze"},
    "nickel_2022":      {"year": 2022, "shock_origin": "indonesia", "commodity": "nickel",
                         "title": "Nickel 2022 — Indonesia Ore Ban + HPAL Processing"},
    "uranium_2007":     {"year": 2007, "shock_origin": "canada",    "commodity": "strategic_mineral",
                         "title": "Uranium 2007 — Cigar Lake Flood (τ_K ≈ 14–20yr)"},
}

PREDICTIVE_SCENARIOS = {
    "pred_graphite_ban_2027":       {"year": 2027, "shock_origin": "china",     "commodity": "graphite",
                                     "title": "PREDICTIVE: China Graphite Full Ban (2027)"},
    "pred_ree_sweep_2028":          {"year": 2028, "shock_origin": "china",     "commodity": "rare_earths",
                                     "title": "PREDICTIVE: China Rare Earth Export Sweep (2028)"},
    "pred_cobalt_instability_2027": {"year": 2027, "shock_origin": "drc",       "commodity": "cobalt",
                                     "title": "PREDICTIVE: DRC Cobalt Political Instability (2027)"},
    "pred_indonesia_nickel_2028":   {"year": 2028, "shock_origin": "indonesia", "commodity": "nickel",
                                     "title": "PREDICTIVE: Indonesia Nickel Escalation (2028)"},
    "pred_us_vulnerability_2030":   {"year": 2030, "shock_origin": "usa",       "commodity": "strategic_mineral",
                                     "title": "PREDICTIVE: US Critical Mineral Import Vulnerability (2030)"},
    "pred_china_sweep_2030":        {"year": 2030, "shock_origin": "china",     "commodity": "graphite",
                                     "title": "PREDICTIVE: China Full Critical Minerals Sweep (2030)"},
}

ALL_SCENARIOS = {**VALIDATION_SCENARIOS, **PREDICTIVE_SCENARIOS}

# ── Temporal comparison series ────────────────────────────────────────────────
# Year-by-year snapshots per commodity. Each series shows how the supply chain
# shifts across structurally meaningful years (pre / during / post regime
# break). Reuses existing validation entries where possible.
#
# Schema: TEMPORAL_SCENARIOS[commodity] = [(scenario_id, year), ...] in chronological
# order. The PNG paths resolve to outputs/kg_scenarios/temporal/<scenario_id>.png
# unless the same scenario_id already exists in validation/ or predictive/, in
# which case we reuse that PNG.

TEMPORAL_SERIES_DEFS = {
    "graphite":    [("graphite_2008",       2008, "china",     "Graphite 2008 — Pre-EV Industrial Cycle"),
                    ("graphite_2015",       2015, "china",     "Graphite 2015 — EV Transition Begins"),
                    ("graphite_2022",       2022, "china",     "Graphite 2022 — China Anode Processing Dominance")],
    "rare_earths": [("rare_earths_2008",    2008, "china",     "Rare Earths 2008 — Pre-Quota Baseline"),
                    ("rare_earths_2010",    2010, "china",     "Rare Earths 2010 — China Export Quota Crisis"),
                    ("rare_earths_2014",    2014, "china",     "Rare Earths 2014 — Post-WTO Supply Flood")],
    "cobalt":      [("cobalt_2010",         2010, "drc",       "Cobalt 2010 — Pre-EV DRC Concentration"),
                    ("cobalt_2016",         2016, "drc",       "Cobalt 2016 — DRC Artisanal Mining Concentration"),
                    ("cobalt_2022",         2022, "drc",       "Cobalt 2022 — Post-COVID Supply Recovery")],
    "lithium":     [("lithium_2014",        2014, "chile",     "Lithium 2014 — Pre-EV Brine Era"),
                    ("lithium_2016",        2016, "chile",     "Lithium 2016 — Atacama Supply Surge"),
                    ("lithium_2022",        2022, "china",     "Lithium 2022 — China Battery-Grade Processing Lock-in")],
    "nickel":      [("nickel_2006",         2006, "russia",    "Nickel 2006 — Norilsk / LME Squeeze"),
                    ("nickel_2014",         2014, "indonesia", "Nickel 2014 — Indonesia First Ore Ban"),
                    ("nickel_2022",         2022, "indonesia", "Nickel 2022 — Indonesia Ore Ban + HPAL Processing")],
    "uranium":     [("uranium_2003",        2003, "canada",    "Uranium 2003 — Pre-Renaissance Canadian Supply"),
                    ("uranium_2007",        2007, "canada",    "Uranium 2007 — Cigar Lake Flood (τ_K ≈ 14–20yr)"),
                    ("uranium_2022",        2022, "russia",    "Uranium 2022 — Russia Sanctions / PRIA")],
}

# Flatten into a {scenario_id: scenario_dict} mapping, taking the first
# occurrence (validation entries win over freshly-defined ones).
TEMPORAL_SCENARIOS = {}
for _commodity, _entries in TEMPORAL_SERIES_DEFS.items():
    for _sid, _year, _origin, _title in _entries:
        if _sid not in TEMPORAL_SCENARIOS and _sid not in VALIDATION_SCENARIOS:
            TEMPORAL_SCENARIOS[_sid] = {
                "year": _year,
                "shock_origin": _origin,
                "commodity": _commodity if _commodity != "rare_earths" else "rare_earths",
                "title": _title,
            }

# ── Colour palette ────────────────────────────────────────────────────────────

_CATEGORY_COLORS = {
    "policy_event": "#E74C3C",
    "economic":     "#F39C12",
    "commodity":    "#3498DB",
    "industry":     "#2ECC71",
    "country":      "#9B59B6",
    "other":        "#95A5A6",
}

_TYPE_TO_CATEGORY = {
    "policy":             "policy_event",
    "event":              "policy_event",
    "economic_indicator": "economic",
    "commodity":          "commodity",
    "industry":           "industry",
    "technology":         "industry",
    "country":            "country",
    "region":             "country",
}


def _node_category(entity_type: str) -> str:
    return _TYPE_TO_CATEGORY.get(entity_type, "other")


# ── Claude + HippoRAG → focal nodes ──────────────────────────────────────────

_QUERY_GEN_PROMPT = """\
You are an expert in critical minerals supply chains and commodity markets.

Given this scenario:
  Title: {title}
  Year: {year}
  Shock origin: {shock_origin}
  Commodity: {commodity}

Generate a single, specific search query (max 20 words) to retrieve the most
relevant document passages about this supply shock from a corpus of USGS Mineral
Commodity Summaries, IEA Critical Minerals reports, and trade databases.

Reply with ONLY the query string, no explanation."""


_QUERY_CACHE_PATH = Path("outputs/kg_scenarios/_query_cache.json")


def _load_query_cache() -> dict:
    try:
        if _QUERY_CACHE_PATH.exists():
            return json.loads(_QUERY_CACHE_PATH.read_text())
    except Exception:
        pass
    return {}


def _save_query_cache(cache: dict) -> None:
    try:
        _QUERY_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        _QUERY_CACHE_PATH.write_text(json.dumps(cache, indent=2))
    except Exception as exc:
        print(f"    [warn] could not write query cache: {exc}")


def _template_query(scenario: dict) -> str:
    """Deterministic fallback query, good enough for HippoRAG retrieval."""
    return (f"{scenario['shock_origin']} {scenario['commodity']} "
            f"{scenario.get('title', '')} {scenario['year']} "
            f"supply chain disruption mineral commodity").strip()


def _generate_hipporag_query(scenario: dict, cache_key: str = "") -> str:
    """
    Return a HippoRAG retrieval query. Order of preference:
      1. Disk cache (keyed by scenario id).
      2. Claude (if API is up and has credits).
      3. Deterministic template built from scenario fields.
    """
    from src.llm.chat import chat_completion, is_chat_available

    cache = _load_query_cache()
    if cache_key and cache_key in cache:
        return cache[cache_key]

    if not is_chat_available():
        return _template_query(scenario)

    prompt = _QUERY_GEN_PROMPT.format(
        title=scenario["title"],
        year=scenario["year"],
        shock_origin=scenario["shock_origin"],
        commodity=scenario["commodity"],
    )
    try:
        query = chat_completion([{"role": "user", "content": prompt}],
                                max_tokens=60).strip().strip('"')
    except Exception as exc:
        print(f"    [warn] LLM query generation failed ({type(exc).__name__}); "
              f"using template fallback.")
        query = _template_query(scenario)

    if cache_key:
        cache[cache_key] = query
        _save_query_cache(cache)
    return query


def _focal_nodes_from_hipporag(extractor, pipeline, scenario: dict,
                                kg_entity_ids: set, top_k: int = 6,
                                scenario_id: str = "") -> tuple:
    """
    1. Claude generates a targeted HippoRAG query from the scenario metadata.
    2. HippoRAG retrieves top-k document chunks.
    3. KGExtractor.extract_from_text() (Claude) extracts triples from each chunk.
    4. Triple subjects/objects are matched against KG entity IDs → focal nodes.

    Returns (focal_node_set, query_used).
    """
    query = _generate_hipporag_query(scenario, cache_key=scenario_id)

    try:
        chunks = pipeline.retrieve(query, top_k=top_k)
    except Exception as exc:
        print(f"    [warn] HippoRAG retrieval failed: {exc}")
        return set(), query

    # Extract triples from each chunk via Claude (best-effort; skip on API failure)
    all_triples = []
    for chunk in chunks:
        text = chunk.get("text", "")
        if not text:
            continue
        try:
            triples = extractor.extract_from_text(
                text,
                source=chunk.get("metadata", {}).get("source", ""),
                year=str(scenario["year"]),
            )
            all_triples.extend(triples)
        except Exception as exc:
            print(f"    [warn] triple extraction failed ({type(exc).__name__}); "
                  f"continuing with remaining chunks.")
            break

    # Resolve triple subjects/objects to KG entity IDs
    focal = set()
    kg_ids_lower = {eid.lower(): eid for eid in kg_entity_ids if len(eid) >= 4}
    for t in all_triples:
        for field in (t.get("subject", ""), t.get("object", "")):
            f_lower = field.lower().strip()
            # Exact match
            if f_lower in kg_ids_lower:
                focal.add(kg_ids_lower[f_lower])
                continue
            # Partial match: KG entity ID appears in the triple text
            for kid_lower, kid in kg_ids_lower.items():
                if re.search(r"\b" + re.escape(kid_lower) + r"\b", f_lower):
                    focal.add(kid)
                    break

    return focal, query


# ── Subgraph builder ──────────────────────────────────────────────────────────

def _build_subgraph(snap_data: dict, focal_nodes: set, max_hops: int = 1):
    import networkx as nx

    G_full = nx.DiGraph()
    for e in snap_data["entities"]:
        G_full.add_node(e["id"], entity_type=e.get("entity_type", "other"))
    for r in snap_data["relationships"]:
        props = r.get("properties", {})
        G_full.add_edge(
            r["source_id"], r["target_id"],
            relation_type=r.get("relation_type", ""),
            share=props.get("share"),
        )

    keep = set(focal_nodes) & set(G_full.nodes)
    for _ in range(max_hops):
        neighbours = set()
        for n in keep:
            if n in G_full:
                neighbours |= set(G_full.successors(n)) | set(G_full.predecessors(n))
        keep |= neighbours

    sub = G_full.subgraph(keep).copy()
    isolates = [n for n in sub.nodes if sub.degree(n) == 0]
    sub.remove_nodes_from(isolates)
    return sub


# ── Scenario renderer ─────────────────────────────────────────────────────────

def _render_scenario(kg_obj, scenario_id: str, scenario: dict,
                     output_path: str, pipeline, extractor,
                     enriched: bool = False) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import matplotlib.colors as mcolors
    import networkx as nx

    year         = scenario["year"]
    shock_origin = scenario["shock_origin"]
    commodity    = scenario["commodity"]
    title        = scenario["title"]

    # ── 1. Claude generates query → HippoRAG retrieves → Claude extracts triples → focal nodes
    all_entity_ids = {e["id"] for e in kg_obj.to_dict()["entities"]}
    hippo_focal, query = _focal_nodes_from_hipporag(
        extractor, pipeline, scenario, all_entity_ids, top_k=6,
        scenario_id=scenario_id)
    focal = {shock_origin, commodity} | hippo_focal
    print(f"    Query (Claude): {query!r}")
    print(f"    Focal nodes ({len(hippo_focal)}): "
          f"{sorted(hippo_focal)[:8]}{'...' if len(hippo_focal) > 8 else ''}")

    # ── 2. Year-specific KG snapshot ─────────────────────────────────────────
    snap = kg_obj.query_at_time(year)
    snap_data = snap.to_dict()

    G = _build_subgraph(snap_data, focal, max_hops=1)

    # Keep most-connected nodes if graph is too dense for readability
    if G.number_of_nodes() > 80:
        deg = dict(G.degree())
        must_keep = {shock_origin, commodity}
        ranked = sorted((n for n in G.nodes() if n not in must_keep),
                        key=lambda n: -deg[n])
        keep = must_keep | set(ranked[:80 - len(must_keep)])
        G    = G.subgraph(keep).copy()

    if G.number_of_nodes() == 0:
        print(f"  [skip] {scenario_id}: no nodes in snapshot @ {year}")
        return {
            "scenario_id": scenario_id,
            "output_path": str(output_path),
            "node_count": 0,
            "focal_count": len(hippo_focal),
            "query": query,
            "skipped": True,
        }

    # ── 3. Shock propagation ──────────────────────────────────────────────────
    try:
        origin_id = snap.resolve_id(shock_origin)
        trace     = snap.propagate_shock(origin_id, initial_magnitude=1.0,
                                         decay=0.6, max_depth=4)
        impact     = trace.affected
        prop_paths = trace.paths
    except Exception:
        impact, prop_paths = {}, {}

    # ── 4. Effective control annotation ──────────────────────────────────────
    try:
        ctrl      = kg_obj.effective_control_at(shock_origin, commodity, year)
        eff_share = ctrl.get("effective_share")
        binding   = ctrl.get("binding", "unknown")
    except Exception:
        eff_share, binding = None, "unknown"

    # ── Layout ────────────────────────────────────────────────────────────────
    pos = nx.spring_layout(G, k=3.0 / max(G.number_of_nodes() ** 0.5, 1),
                           seed=42, iterations=150)

    # ── Edge styling ──────────────────────────────────────────────────────────
    hot_edges = set()
    for path in prop_paths.values():
        for i in range(len(path) - 1):
            hot_edges.add((path[i], path[i + 1]))

    edge_colors, edge_widths = [], []
    for u, v in G.edges():
        share = G[u][v].get("share")
        w = 1.0 + (float(share) * 5.0 if share else 0.0)
        edge_widths.append(min(w, 8.0))
        edge_colors.append("#C0392B" if (u, v) in hot_edges else "#AAAAAA")

    # ── Node styling: heat = impact magnitude ─────────────────────────────────
    impact_cmap = plt.cm.YlOrRd
    norm        = mcolors.Normalize(vmin=0.0, vmax=1.0)

    node_colors, node_sizes = [], []
    for n in G.nodes():
        if n == shock_origin:
            node_colors.append("#8B0000")
            node_sizes.append(3000)
        elif n == commodity:
            mag = impact.get(n, 0.0)
            node_colors.append(impact_cmap(norm(max(mag, 0.3))))
            node_sizes.append(2600)
        else:
            mag = impact.get(n, 0.0)
            if mag > 0.01:
                node_colors.append(impact_cmap(norm(mag)))
                node_sizes.append(800 + int(mag * 1400))
            else:
                etype = G.nodes[n].get("entity_type", "other")
                node_colors.append(_CATEGORY_COLORS[_node_category(etype)])
                node_sizes.append(900)

    # ── Draw ──────────────────────────────────────────────────────────────────
    n_nodes = G.number_of_nodes()
    fig_w = min(22, max(14, n_nodes * 0.35))
    fig_h = min(17, max(10, n_nodes * 0.28))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_facecolor("#F8F9FA")
    fig.patch.set_facecolor("#F8F9FA")

    nx.draw_networkx_edges(G, pos, ax=ax,
                           edge_color=edge_colors, width=edge_widths,
                           arrows=True, arrowsize=16, arrowstyle="-|>",
                           connectionstyle="arc3,rad=0.07",
                           min_source_margin=18, min_target_margin=18)
    nx.draw_networkx_nodes(G, pos, ax=ax,
                           node_color=node_colors, node_size=node_sizes,
                           alpha=0.93, edgecolors="#2c3e50", linewidths=1.3)

    font_sz = max(6, min(9, 170 // max(n_nodes, 1)))
    labels  = {n: n.replace("_", "\n")[:26] for n in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels=labels, ax=ax,
                            font_size=font_sz, font_weight="bold")

    # Share labels on PRODUCES/PROCESSES edges
    share_labels = {}
    for u, v in G.edges():
        s  = G[u][v].get("share")
        rt = G[u][v].get("relation_type", "").lower()
        if s and rt in ("produces", "processes"):
            share_labels[(u, v)] = f"{rt[:3].upper()} {float(s):.0%}"
    if share_labels:
        nx.draw_networkx_edge_labels(G, pos, edge_labels=share_labels, ax=ax,
                                     font_size=6, font_color="#1a5276",
                                     bbox=dict(boxstyle="round,pad=0.12",
                                               fc="white", alpha=0.7))

    # Colourbar
    sm = plt.cm.ScalarMappable(cmap=impact_cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("Shock impact magnitude", fontsize=9)

    # Effective control box
    if eff_share is not None:
        ctrl_text = (f"Effective control ({year})\n"
                     f"{shock_origin.upper()} → {commodity}: "
                     f"{eff_share:.0%}  [{binding}]")
        ax.text(0.98, 0.02, ctrl_text, transform=ax.transAxes,
                fontsize=10, va="bottom", ha="right",
                bbox=dict(boxstyle="round,pad=0.4", fc="#EBF5FB", alpha=0.9))

    ax.text(0.01, 0.01,
            f'Claude query → HippoRAG: "{query[:72]}{"…" if len(query) > 72 else ""}"',
            transform=ax.transAxes, fontsize=6, va="bottom", ha="left",
            color="#666666")

    legend_items = [
        mpatches.Patch(color="#8B0000",                        label=f"Shock origin: {shock_origin}"),
        mpatches.Patch(color=impact_cmap(0.6),                 label=f"Focal commodity: {commodity}"),
        mpatches.Patch(color=_CATEGORY_COLORS["country"],      label="Country / Region"),
        mpatches.Patch(color=_CATEGORY_COLORS["policy_event"], label="Policy / Event"),
        mpatches.Patch(color=_CATEGORY_COLORS["industry"],     label="Industry / Tech"),
        mpatches.Patch(color="#AAAAAA",                        label="Edge: structural"),
        mpatches.Patch(color="#C0392B",                        label="Edge: propagation path"),
    ]
    ax.legend(handles=legend_items, loc="upper left", fontsize=9, framealpha=0.95)

    src_label = "Enriched KG" if enriched else "Seed KG"
    is_pred   = scenario_id.startswith("pred_")
    kind      = "Predictive" if is_pred else "Validation"
    ax.set_title(
        f"[{kind}] {title}\n"
        f"{src_label} snapshot @ {year}  |  {n_nodes} nodes  |  "
        f"HippoRAG focal: {len(hippo_focal)}",
        fontsize=12, fontweight="bold", pad=12,
    )
    ax.axis("off")
    plt.tight_layout()

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out), bbox_inches="tight", dpi=200,
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved: {out.resolve()}  "
          f"({n_nodes} nodes, impact={len(impact)}, eff={eff_share})")

    return {
        "scenario_id": scenario_id,
        "output_path": str(out),
        "node_count": int(n_nodes),
        "focal_count": int(len(hippo_focal)),
        "edge_count": int(G.number_of_edges()),
        "impact_count": int(len(impact)),
        "effective_share": float(eff_share) if eff_share is not None else None,
        "binding": binding,
        "query": query,
        "skipped": False,
    }


# ── Full-KG DAG visualisation (--dag-image) ───────────────────────────────────

def visualize_kg_dag(kg_data: dict, output_path: str, max_nodes: int = 60) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import networkx as nx

    G_full = nx.DiGraph()
    for e in kg_data["entities"]:
        G_full.add_node(e["id"], entity_type=e.get("entity_type", "other"))
    for r in kg_data["relationships"]:
        G_full.add_edge(r["source_id"], r["target_id"],
                        relation_type=r.get("relation_type", ""))

    G = G_full
    if max_nodes and G_full.number_of_nodes() > max_nodes:
        deg = dict(G_full.degree())
        top = sorted(deg, key=lambda n: -deg[n])[:max_nodes]
        G   = G_full.subgraph(top).copy()

    node_categories = {n: _node_category(G.nodes[n].get("entity_type", "other"))
                       for n in G.nodes()}
    pos = nx.spring_layout(G, k=2.5 / max(G.number_of_nodes() ** 0.5, 1),
                           seed=42, iterations=100)

    n_nodes    = G.number_of_nodes()
    node_size  = max(600, 2400 - n_nodes * 12)
    node_colors = [_CATEGORY_COLORS[node_categories[n]] for n in G.nodes()]
    edge_colors = [_CATEGORY_COLORS[node_categories[u]] for u, _ in G.edges()]

    fig_w = min(24, max(16, n_nodes * 0.28))
    fig_h = min(18, max(12, n_nodes * 0.20))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_facecolor("#FAFAFA")
    fig.patch.set_facecolor("#FAFAFA")

    nx.draw_networkx_edges(G, pos, ax=ax, edge_color=edge_colors, alpha=0.5,
                           arrows=True, arrowsize=20, arrowstyle="-|>",
                           connectionstyle="arc3,rad=0.08", width=1.5,
                           min_source_margin=22, min_target_margin=22)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors,
                           node_size=node_size, alpha=0.92,
                           edgecolors="#2c3e50", linewidths=1.5)

    font_sz = max(6, min(11, 220 // max(n_nodes, 1)))
    labels = {n: (n.replace("_", " ")[:23] + "…" if len(n) > 24
                  else n.replace("_", " ")) for n in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels=labels, ax=ax,
                            font_size=font_sz, font_weight="bold")

    legend_items = [
        mpatches.Patch(color=_CATEGORY_COLORS[c], label=l) for c, l in [
            ("commodity",    "Commodity"),
            ("country",      "Country / Region"),
            ("policy_event", "Policy / Event"),
            ("industry",     "Industry / Tech"),
            ("economic",     "Economic Indicator"),
            ("other",        "Other"),
        ]
    ]
    ax.legend(handles=legend_items, loc="upper left", fontsize=14, framealpha=0.95)
    ax.set_title("Critical Minerals Causal Knowledge Graph",
                 fontsize=22, fontweight="bold", pad=20)
    ax.margins(0.1)
    ax.axis("off")
    plt.tight_layout()

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out), bbox_inches="tight", dpi=280, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"DAG saved → {out.resolve()}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--enriched", action="store_true",
                        help="Load enriched KG (472 entities) instead of seed KG.")
    parser.add_argument("--scenario", type=str, metavar="ID",
                        choices=list(ALL_SCENARIOS.keys()),
                        help="Generate one scenario PNG.")
    parser.add_argument("--validation", action="store_true",
                        help="Generate all 10 validation episode PNGs.")
    parser.add_argument("--predictive", action="store_true",
                        help="Generate all 6 predictive scenario PNGs.")
    parser.add_argument("--all-scenarios", action="store_true",
                        help="Generate all 16 scenario PNGs.")
    parser.add_argument("--temporal", action="store_true",
                        help="Generate the temporal comparison series (year-by-year per commodity).")
    parser.add_argument("--scenario-dir", type=str, default="outputs/kg_scenarios",
                        help="Output directory (default: outputs/kg_scenarios/).")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip scenarios whose output PNG already exists.")
    parser.add_argument("--dag-image", type=str, metavar="PATH",
                        help="Write full KG overview PNG.")
    parser.add_argument("--summary", action="store_true", default=True)
    parser.add_argument("--no-summary", action="store_false", dest="summary")
    parser.add_argument("--shock", type=str, metavar="ORIGIN_ID",
                        help="Print shock propagation trace.")
    parser.add_argument("--dag-edges", action="store_true")
    parser.add_argument("--list-origins", action="store_true")
    args = parser.parse_args()

    from src.minerals.knowledge_graph import build_critical_minerals_kg, CausalKnowledgeGraph

    # ── Load KG ───────────────────────────────────────────────────────────────
    if args.enriched:
        enriched_path = Path("data/canonical/enriched_kg.json")
        if not enriched_path.exists():
            print(f"ERROR: {enriched_path} not found.")
            sys.exit(1)
        kg_obj  = CausalKnowledgeGraph.load(str(enriched_path))
        kg_data = kg_obj.to_dict()
        print(f"Loaded enriched KG: {len(kg_data['entities'])} entities, "
              f"{len(kg_data['relationships'])} relationships\n")
    else:
        kg_obj  = build_critical_minerals_kg()
        kg_data = kg_obj.to_dict()
        print("Built seed KG.\n")

    if args.summary:
        print(f"Entities: {len(kg_data['entities'])}  |  "
              f"Relationships: {len(kg_data['relationships'])}\n")

    # ── Init HippoRAG + KGExtractor (once — expensive) ───────────────────────
    needs_hipporag = (args.scenario or args.validation or
                      args.predictive or args.all_scenarios or args.temporal)
    pipeline  = None
    extractor = None
    if needs_hipporag:
        print("Initialising HippoRAG pipeline + KGExtractor (Claude backend)...")
        from src.minerals.rag_pipeline import RAGPipeline
        from src.minerals.kg_extractor import KGExtractor
        pipeline  = RAGPipeline(backend="hipporag")
        extractor = KGExtractor(pipeline=pipeline)
        print(f"  RAG backend:  {pipeline.backend_name}")
        from src.llm.chat import is_chat_available, _backend
        print(f"  LLM backend:  {_backend()} (available={is_chat_available()})\n")

    # ── Utility flags ─────────────────────────────────────────────────────────
    if args.list_origins:
        for o in kg_obj.get_shock_origin_candidates():
            print(f"  - {o}")
        print()

    if args.shock:
        trace = kg_obj.propagate_shock(args.shock.strip(), initial_magnitude=1.0,
                                       decay=0.5, max_depth=5)
        print(f"Shock origin: {trace.origin}")
        for eid, mag in sorted(trace.affected.items(), key=lambda x: -x[1]):
            path = " -> ".join(trace.paths.get(eid, [eid]))
            print(f"  {eid} (impact {mag:.3f}): {path}")
        print()

    if args.dag_edges:
        dag = kg_obj.to_causal_dag()
        for u, v in sorted(dag.graph.edges()):
            print(f"  {u} -> {v}")
        print()

    if args.dag_image:
        visualize_kg_dag(kg_data, args.dag_image)

    # ── Scenario rendering ────────────────────────────────────────────────────
    to_run = {}
    if args.scenario:
        to_run = {args.scenario: ALL_SCENARIOS[args.scenario]}
    elif args.all_scenarios:
        to_run = ALL_SCENARIOS
    elif args.validation:
        to_run = VALIDATION_SCENARIOS
    elif args.predictive:
        to_run = PREDICTIVE_SCENARIOS
    elif args.temporal:
        to_run = TEMPORAL_SCENARIOS

    if to_run:
        print(f"Generating {len(to_run)} scenario PNGs → {args.scenario_dir}/")
        failures = []
        for sid, s in to_run.items():
            if sid.startswith("pred_"):
                subdir = "predictive"
            elif sid in TEMPORAL_SCENARIOS:
                subdir = "temporal"
            else:
                subdir = "validation"
            out = Path(args.scenario_dir) / subdir / f"{sid}.png"
            if args.skip_existing and out.exists():
                print(f"\n[{sid}]  skipped (PNG already exists)")
                continue
            print(f"\n[{sid}]  year={s['year']}  origin={s['shock_origin']}")
            try:
                _render_scenario(kg_obj, sid, s, str(out),
                                 pipeline=pipeline, extractor=extractor,
                                 enriched=args.enriched)
            except Exception as exc:
                print(f"  [error] {sid} failed: {type(exc).__name__}: {exc}")
                failures.append(sid)
        if failures:
            print(f"\nDone with {len(failures)} failure(s): {failures}")
        else:
            print("\nDone.")


if __name__ == "__main__":
    main()
