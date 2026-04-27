#!/usr/bin/env python3
"""
Bundle every pre-rendered KG snapshot PNG into a multi-page PDF for the
thesis appendix (2 KG renders per page, organised by commodity).

Run after adding new scenarios:
    python scripts/build_kg_snapshots_pdf.py

Outputs to outputs/kg_scenarios/kg_snapshots_appendix.pdf — committed to git
and served by FastAPI at /api/kg/snapshots-export.

Generated locally (not at request time) because bundling 24 PNGs blows past
the 2GB Fly memory limit.
"""

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

KG_DIR = Path("outputs/kg_scenarios")
OUT_PATH = KG_DIR / "kg_snapshots_appendix.pdf"

COMMODITIES = {
    "Graphite": [
        ("graphite_2008", 2008, "historical", "China export quota — pre-EV cycle"),
        ("graphite_2015", 2015, "temporal", "EV transition begins"),
        ("graphite_2022", 2022, "historical", "China anode processing 95% lock-in"),
        ("pred_graphite_ban_2027", 2027, "predictive", "PREDICTIVE: full ban scenario"),
    ],
    "Rare Earths": [
        ("rare_earths_2008", 2008, "temporal", "Pre-quota baseline"),
        ("rare_earths_2010", 2010, "historical", "China export quota crisis"),
        ("rare_earths_2014", 2014, "temporal", "Post-WTO supply flood"),
        ("pred_ree_sweep_2028", 2028, "predictive", "PREDICTIVE: export sweep"),
    ],
    "Cobalt": [
        ("cobalt_2010", 2010, "temporal", "Pre-EV DRC concentration"),
        ("cobalt_2016", 2016, "historical", "DRC artisanal concentration"),
        ("cobalt_2022", 2022, "historical", "Post-COVID recovery"),
        ("pred_cobalt_instability_2027", 2027, "predictive", "PREDICTIVE: DRC instability"),
    ],
    "Lithium": [
        ("lithium_2014", 2014, "temporal", "Pre-EV brine era"),
        ("lithium_2016", 2016, "historical", "Atacama supply surge"),
        ("lithium_2022", 2022, "historical", "China processing lock-in"),
    ],
    "Nickel": [
        ("nickel_2006", 2006, "historical", "Norilsk / LME squeeze"),
        ("nickel_2014", 2014, "temporal", "Indonesia first ore ban"),
        ("nickel_2022", 2022, "historical", "Indonesia ban + HPAL"),
        ("pred_indonesia_nickel_2028", 2028, "predictive", "PREDICTIVE: escalation"),
    ],
    "Uranium": [
        ("uranium_2003", 2003, "temporal", "Pre-renaissance Canadian supply"),
        ("uranium_2007", 2007, "historical", "Cigar Lake flood"),
        ("uranium_2022", 2022, "temporal", "Russia sanctions / PRIA"),
    ],
    "Cross-cutting predictive": [
        ("pred_us_vulnerability_2030", 2030, "predictive", "PREDICTIVE: US import vulnerability"),
        ("pred_china_sweep_2030", 2030, "predictive", "PREDICTIVE: China full-sweep"),
    ],
}

KIND_BADGE = {
    "historical": ("Historical episode", "#1d4ed8"),
    "temporal": ("Temporal snapshot", "#0e7490"),
    "predictive": ("Predictive scenario", "#a16207"),
}


def _resolve(sid: str) -> Path | None:
    for sub in ("validation", "temporal", "predictive"):
        p = KG_DIR / sub / f"{sid}.png"
        if p.exists():
            return p
    return None


def main() -> None:
    KG_DIR.mkdir(parents=True, exist_ok=True)

    with PdfPages(OUT_PATH) as pdf:
        # Title page
        fig = plt.figure(figsize=(11, 8.5))
        fig.text(0.5, 0.62, "Critical Minerals Causal Engine",
                 ha="center", fontsize=22, fontweight="bold")
        fig.text(0.5, 0.55, "Appendix — Knowledge Graph Renders",
                 ha="center", fontsize=14, color="#475569")
        fig.text(0.5, 0.42,
                 "Each render: enriched KG snapshot at the indicated year, with shock\n"
                 "origin (dark red node), focal commodity, and 1-hop subgraph.\n"
                 "Edges annotated with year-specific PRODUCES / PROCESSES shares.\n"
                 "Effective control box (bottom-right) shows the binding stage and percentage.",
                 ha="center", fontsize=10, color="#64748b")
        toc_y = 0.30
        fig.text(0.30, toc_y, "Contents", fontsize=11, fontweight="bold", color="#1e293b")
        toc_y -= 0.025
        for cmd, items in COMMODITIES.items():
            fig.text(0.30, toc_y, f"{cmd} ({len(items)} renders)", fontsize=9, color="#475569")
            toc_y -= 0.022
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # Per-commodity pages, 2 KGs/page
        for commodity_name, items in COMMODITIES.items():
            entries = [(sid, year, kind, subtitle, _resolve(sid))
                       for sid, year, kind, subtitle in items
                       if _resolve(sid) is not None]
            for i in range(0, len(entries), 2):
                fig, axes = plt.subplots(2, 1, figsize=(11, 8.5),
                                         gridspec_kw={"hspace": 0.18})
                pair = entries[i:i + 2]
                for ax_idx, (sid, year, kind, subtitle, p) in enumerate(pair):
                    ax = axes[ax_idx]
                    img = plt.imread(str(p))
                    ax.imshow(img)
                    ax.set_axis_off()
                    label, color = KIND_BADGE.get(kind, ("", "#64748b"))
                    ax.set_title(f"{commodity_name} · {year}", fontsize=11, fontweight="bold",
                                 loc="left", pad=4)
                    ax.text(0.0, -0.02, f"{label} — {subtitle}",
                            fontsize=8, color=color, transform=ax.transAxes,
                            ha="left", va="top", style="italic")
                if len(pair) == 1:
                    axes[1].set_axis_off()
                pdf.savefig(fig, bbox_inches="tight", dpi=150)
                plt.close(fig)

    size_mb = OUT_PATH.stat().st_size / 1024 / 1024
    print(f"Wrote {OUT_PATH} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
