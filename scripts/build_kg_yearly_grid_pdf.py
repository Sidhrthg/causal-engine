#!/usr/bin/env python3
"""
Render KG snapshots at even intervals (every 2 years) for each commodity from
earliest to latest yearly_share data. One KG per page.

Run after the seed KG changes:
    python scripts/build_kg_yearly_grid_pdf.py

Outputs:
    outputs/kg_scenarios/yearly_grid/<commodity>_<origin>_<year>.png   (cached)
    outputs/kg_scenarios/kg_yearly_grid_appendix.pdf                   (final)
"""

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

KG_DIR = Path("outputs/kg_scenarios")
GRID_DIR = KG_DIR / "yearly_grid"
PDF_OUT = KG_DIR / "kg_yearly_grid_appendix.pdf"

# Dominant supplier per commodity (used as shock_origin for all snapshots in
# that commodity's grid — consistent within a series so the visualisation
# shows how that supplier's footprint shifts over time).
COMMODITY_ORIGIN = {
    "graphite": "china",
    "rare_earths": "china",
    "cobalt": "drc",
    "lithium": "china",
    "nickel": "indonesia",
    "uranium": "russia",
}

INTERVAL = 2  # years between snapshots


def _year_range_for(kg, commodity_id: str) -> tuple[int, int]:
    """Find min/max year from yearly_share keys on PRODUCES/PROCESSES edges
    pointing into the commodity."""
    from src.minerals.knowledge_graph import RelationType
    years: set[int] = set()
    for u, v, data in kg._graph.edges(data=True):
        if v != commodity_id:
            continue
        rel = data["relationship"]
        if rel.relation_type not in (RelationType.PRODUCES, RelationType.PROCESSES):
            continue
        yearly = (rel.properties or {}).get("yearly_share") or {}
        for y in yearly.keys():
            years.add(int(y))
    if not years:
        return (2010, 2022)
    return (min(years), max(years))


def _render_one(kg_obj, commodity: str, origin: str, year: int, out_path: Path) -> dict:
    from scripts.run_knowledge_graph import _render_scenario as render_fn
    sid = f"{commodity}_{origin}_{year}"
    scenario = {
        "year": int(year), "shock_origin": origin, "commodity": commodity,
        "title": f"{commodity.title()} {year} — {origin.upper()} snapshot",
    }
    return render_fn(
        kg_obj=kg_obj, scenario_id=sid, scenario=scenario,
        output_path=str(out_path),
        pipeline=None, extractor=None,  # fast path: no HippoRAG
        enriched=True,
    )


def main() -> None:
    GRID_DIR.mkdir(parents=True, exist_ok=True)

    from src.minerals.knowledge_graph import CausalKnowledgeGraph
    kg_obj = CausalKnowledgeGraph.load("data/canonical/enriched_kg.json")

    # Plan: list of (commodity, origin, year, png_path, control_dict)
    plan: list[tuple[str, str, int, Path, dict | None]] = []
    for commodity, origin in COMMODITY_ORIGIN.items():
        cid = kg_obj.resolve_id(commodity)
        if cid is None:
            print(f"  [skip] {commodity}: not in KG")
            continue
        y_min, y_max = _year_range_for(kg_obj, cid)
        years = list(range(y_min, y_max + 1, INTERVAL))
        # Always include the last year if it isn't already
        if years[-1] != y_max:
            years.append(y_max)
        print(f"  {commodity}: {y_min}–{y_max}, {len(years)} snapshots")
        for year in years:
            png_path = GRID_DIR / f"{commodity}_{origin}_{year}.png"
            ctrl = None
            try:
                c = kg_obj.effective_control_at(origin, commodity, year)
                if c:
                    ctrl = {
                        "produces": c.get("produces_share"),
                        "processes": c.get("processes_share"),
                        "effective": c.get("effective_share"),
                        "binding": c.get("binding"),
                    }
            except Exception:
                pass
            plan.append((commodity, origin, year, png_path, ctrl))

    # Render every snapshot that doesn't already exist on disk
    rendered = skipped = 0
    for commodity, origin, year, png_path, _ctrl in plan:
        if png_path.exists():
            skipped += 1
            continue
        try:
            _render_one(kg_obj, commodity, origin, year, png_path)
            rendered += 1
        except Exception as exc:
            print(f"  [error] {commodity} {year}: {type(exc).__name__}: {exc}")
    print(f"\nRendered {rendered} new, skipped {skipped} existing.")

    # Build PDF (1 KG/page, landscape A4)
    print(f"\nBuilding PDF → {PDF_OUT}")
    with PdfPages(PDF_OUT) as pdf:
        # Title page
        fig = plt.figure(figsize=(11, 8.5))
        fig.text(0.5, 0.62, "Critical Minerals Causal Engine", ha="center",
                 fontsize=22, fontweight="bold")
        fig.text(0.5, 0.55, "Appendix — Yearly Grid (every 2 years)", ha="center",
                 fontsize=14, color="#475569")
        fig.text(0.5, 0.40,
                 "One KG render per page, every 2 years, per commodity.\n"
                 "Shock origin = dominant supplier (most recent year): china for graphite\n"
                 "and rare earths, drc for cobalt, china for lithium (processing-bound),\n"
                 "indonesia for nickel, russia for uranium.\n\n"
                 "Year-to-year visual differences are subtle (only edge share % labels and\n"
                 "the effective control box change); the grid documents the gradual drift.\n"
                 "For sharper structural breaks, see the Curated Snapshots PDF.",
                 ha="center", fontsize=9, color="#64748b")
        toc_y = 0.18
        fig.text(0.30, toc_y, "Year ranges by commodity:", fontsize=10, fontweight="bold")
        toc_y -= 0.020
        from collections import defaultdict
        by_cmd: dict[str, list[int]] = defaultdict(list)
        for commodity, _origin, year, _path, _ctrl in plan:
            by_cmd[commodity].append(year)
        for cmd in COMMODITY_ORIGIN.keys():
            ys = by_cmd.get(cmd, [])
            if not ys:
                continue
            label = f"  {cmd.replace('_', ' ').title()}: {ys[0]}–{ys[-1]} ({len(ys)} snapshots)"
            fig.text(0.30, toc_y, label, fontsize=8, color="#475569")
            toc_y -= 0.018
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # One KG per page
        for commodity, origin, year, png_path, ctrl in plan:
            if not png_path.exists():
                continue
            fig = plt.figure(figsize=(11, 8.5))
            ax_img = fig.add_axes([0.05, 0.10, 0.90, 0.82])
            try:
                img = plt.imread(str(png_path))
                ax_img.imshow(img)
            except Exception as exc:
                ax_img.text(0.5, 0.5, f"Render failed: {exc}", ha="center", va="center")
            ax_img.set_axis_off()

            # Title above
            fig.text(0.05, 0.95, f"{commodity.replace('_', ' ').title()} · {year}",
                     fontsize=14, fontweight="bold", color="#0f172a")
            fig.text(0.05, 0.925, f"shock origin: {origin.upper()}",
                     fontsize=9, color="#64748b", style="italic")

            # Stats footer
            if ctrl:
                produces = ctrl.get("produces")
                processes = ctrl.get("processes")
                effective = ctrl.get("effective")
                binding = ctrl.get("binding") or "—"

                def _fmt(s):
                    return f"{s * 100:.0f}%" if s is not None else "—"

                footer = (f"Mining (PRODUCES): {_fmt(produces)}     "
                          f"Processing (PROCESSES): {_fmt(processes)}     "
                          f"Effective control: {_fmt(effective)} ({binding}-bound)")
                fig.text(0.05, 0.05, footer, fontsize=9, color="#1e293b", family="monospace")

            pdf.savefig(fig, bbox_inches="tight", dpi=140)
            plt.close(fig)

    size_mb = PDF_OUT.stat().st_size / 1024 / 1024
    print(f"Wrote {PDF_OUT} ({size_mb:.1f} MB, {len(plan) + 1} pages including title).")


if __name__ == "__main__":
    main()
