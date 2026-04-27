#!/usr/bin/env python3
"""
Bundle every pre-rendered KG snapshot PNG into a multi-page PDF for the
thesis appendix (2 KG renders per page, organised by commodity).

Run after adding new scenarios:
    python scripts/build_kg_snapshots_pdf.py

Outputs to outputs/kg_scenarios/kg_snapshots_appendix.pdf — committed to git
and served by FastAPI at /api/kg/snapshots-export.

Uses PIL for assembly so the source PNGs (rendered at dpi=200) are embedded
at native resolution. Earlier matplotlib-imshow version blurred them at
savefig DPI.
"""

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image, ImageDraw, ImageFont

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
    "historical": ("Historical episode", (29, 78, 216)),
    "temporal": ("Temporal snapshot", (14, 116, 144)),
    "predictive": ("Predictive scenario", (161, 98, 7)),
}

DPI = 200
PAGE_W = int(11.0 * DPI)
PAGE_H = int(8.5 * DPI)
MARGIN = int(0.4 * DPI)
HALF_H = (PAGE_H - 2 * MARGIN) // 2


def _resolve(sid: str) -> Path | None:
    for sub in ("validation", "temporal", "predictive"):
        p = KG_DIR / sub / f"{sid}.png"
        if p.exists():
            return p
    return None


def _font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = (
        ("/System/Library/Fonts/Helvetica.ttc", 1 if bold else 0),
        ("/Library/Fonts/Arial.ttf", 0) if not bold else ("/Library/Fonts/Arial Bold.ttf", 0),
        ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold
         else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 0),
    )
    for path, idx in candidates:
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size, index=idx)
            except Exception:
                try:
                    return ImageFont.truetype(path, size)
                except Exception:
                    continue
    return ImageFont.load_default()


def _draw_kg_panel(canvas: Image.Image, draw: ImageDraw.ImageDraw,
                   commodity: str, year: int, kind: str, subtitle: str,
                   png_path: Path, top_y: int) -> None:
    """Draw one KG panel onto an existing canvas, anchored at (MARGIN, top_y)."""
    title_font = _font(24, bold=True)
    badge_font = _font(16)

    # Header line
    draw.text((MARGIN, top_y), f"{commodity} · {year}",
              fill=(15, 23, 42), font=title_font)
    badge_text, badge_color = KIND_BADGE.get(kind, ("", (100, 116, 139)))
    draw.text((MARGIN, top_y + 36), f"{badge_text} — {subtitle}",
              fill=badge_color, font=badge_font)

    # Image area below header
    img_top = top_y + 70
    img_max_h = HALF_H - 90
    img_max_w = PAGE_W - 2 * MARGIN
    if not png_path.exists():
        draw.text((MARGIN, img_top + 100), f"[render missing: {png_path.name}]",
                  fill=(220, 38, 38), font=badge_font)
        return
    with Image.open(png_path) as img:
        img.load()
        sw, sh = img.size
        scale = min(img_max_w / sw, img_max_h / sh)
        if scale < 1.0:
            img = img.resize((int(sw * scale), int(sh * scale)), Image.LANCZOS)
        if img.mode != "RGB":
            img = img.convert("RGB")
        x = MARGIN + (img_max_w - img.width) // 2
        canvas.paste(img, (x, img_top))


def _build_title_page() -> Image.Image:
    page = Image.new("RGB", (PAGE_W, PAGE_H), "white")
    draw = ImageDraw.Draw(page)
    draw.text((PAGE_W // 2 - 380, 360), "Critical Minerals Causal Engine",
              fill=(15, 23, 42), font=_font(56, bold=True))
    draw.text((PAGE_W // 2 - 240, 440), "Appendix — Knowledge Graph Renders",
              fill=(71, 85, 105), font=_font(28))
    body_font = _font(18)
    bold_font = _font(18, bold=True)
    body = (
        "Each render: enriched KG snapshot at the indicated year, with shock",
        "origin (dark red node), focal commodity, and 1-hop subgraph.",
        "Edges annotated with year-specific PRODUCES / PROCESSES shares.",
        "Effective control box (bottom-right) shows binding stage and percentage.",
    )
    y = 520
    for line in body:
        draw.text((MARGIN + 100, y), line, fill=(100, 116, 139), font=body_font)
        y += 26
    y += 20
    draw.text((MARGIN + 100, y), "Contents:", fill=(15, 23, 42), font=bold_font)
    y += 30
    for cmd, items in COMMODITIES.items():
        line = f"   {cmd} ({len(items)} renders)"
        draw.text((MARGIN + 100, y), line, fill=(71, 85, 105), font=body_font)
        y += 24
    return page


def main() -> None:
    pages: list[Image.Image] = [_build_title_page()]

    for commodity_name, items in COMMODITIES.items():
        entries = [(sid, year, kind, subtitle, _resolve(sid))
                   for sid, year, kind, subtitle in items
                   if _resolve(sid) is not None]
        for i in range(0, len(entries), 2):
            page = Image.new("RGB", (PAGE_W, PAGE_H), "white")
            draw = ImageDraw.Draw(page)
            pair = entries[i:i + 2]
            for ax_idx, (_sid, year, kind, subtitle, p) in enumerate(pair):
                top_y = MARGIN + ax_idx * HALF_H
                _draw_kg_panel(page, draw, commodity_name, year, kind, subtitle, p, top_y)
            pages.append(page)

    first, *rest = pages
    first.save(OUT_PATH, "PDF", resolution=DPI, save_all=True, append_images=rest)
    size_mb = OUT_PATH.stat().st_size / 1024 / 1024
    print(f"Wrote {OUT_PATH} ({size_mb:.1f} MB, {len(pages)} pages).")


if __name__ == "__main__":
    main()
