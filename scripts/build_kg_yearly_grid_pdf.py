#!/usr/bin/env python3
"""
Render KG snapshots at even intervals (every 2 years) for each commodity from
earliest to latest yearly_share data. One KG per page.

Run after the seed KG changes:
    python scripts/build_kg_yearly_grid_pdf.py

Outputs:
    outputs/kg_scenarios/yearly_grid/<commodity>_<origin>_<year>.png   (cached)
    outputs/kg_scenarios/kg_yearly_grid_appendix.pdf                   (final)

Uses PIL for the PDF assembly so the source PNGs (saved at dpi=200 by the
underlying renderer) keep their full resolution. matplotlib's imshow
re-rasterizes at savefig DPI and produces fuzzy output unless dpi >= source.
"""

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image, ImageDraw, ImageFont

KG_DIR = Path("outputs/kg_scenarios")
GRID_DIR = KG_DIR / "yearly_grid"
PDF_OUT = KG_DIR / "kg_yearly_grid_appendix.pdf"

COMMODITY_ORIGIN = {
    "graphite": "china",
    "rare_earths": "china",
    "cobalt": "drc",
    "lithium": "china",
    "nickel": "indonesia",
    "uranium": "russia",
}

INTERVAL = 2  # years between snapshots

# Page geometry (landscape A4 at 200 DPI)
DPI = 200
PAGE_W = int(11.0 * DPI)   # 2200
PAGE_H = int(8.5 * DPI)    # 1700
MARGIN = int(0.5 * DPI)    # 100
TITLE_BAND_H = int(0.6 * DPI)
FOOTER_BAND_H = int(0.4 * DPI)
IMAGE_AREA_W = PAGE_W - 2 * MARGIN
IMAGE_AREA_H = PAGE_H - TITLE_BAND_H - FOOTER_BAND_H - 2 * MARGIN


def _font(size: int) -> ImageFont.ImageFont:
    # Try common system fonts; fall back to default bitmap font.
    for path in ("/System/Library/Fonts/Helvetica.ttc",
                 "/System/Library/Fonts/Supplemental/Arial.ttf",
                 "/Library/Fonts/Arial.ttf",
                 "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                continue
    return ImageFont.load_default()


def _font_bold(size: int) -> ImageFont.ImageFont:
    for path in ("/System/Library/Fonts/Helvetica.ttc",
                 "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
                 "/Library/Fonts/Arial Bold.ttf",
                 "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"):
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size, index=0)
            except Exception:
                try:
                    return ImageFont.truetype(path, size)
                except Exception:
                    continue
    return _font(size)


def _year_range_for(kg, commodity_id: str) -> tuple[int, int]:
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
    return (min(years), max(years)) if years else (2010, 2022)


def _render_one(kg_obj, commodity: str, origin: str, year: int, out_path: Path):
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


def _build_page(commodity: str, origin: str, year: int,
                png_path: Path, ctrl: dict | None) -> Image.Image:
    """Compose one page: title band + KG image (preserved at native resolution
    via PIL.Image.thumbnail with LANCZOS) + footer band."""
    page = Image.new("RGB", (PAGE_W, PAGE_H), "white")
    draw = ImageDraw.Draw(page)

    # Title band
    title_font = _font_bold(40)
    subtitle_font = _font(20)
    draw.text(
        (MARGIN, MARGIN),
        f"{commodity.replace('_', ' ').title()} · {year}",
        fill=(15, 23, 42), font=title_font,
    )
    draw.text(
        (MARGIN, MARGIN + 50),
        f"shock origin: {origin.upper()}",
        fill=(100, 116, 139), font=subtitle_font,
    )

    # KG image — keep aspect ratio, fit into image area, no upscaling beyond source
    if png_path.exists():
        with Image.open(png_path) as img:
            img.load()
            # Compute the largest resize that fits the image area, preserving aspect
            src_w, src_h = img.size
            scale = min(IMAGE_AREA_W / src_w, IMAGE_AREA_H / src_h)
            target_w = int(src_w * scale)
            target_h = int(src_h * scale)
            if scale < 1.0:
                resized = img.resize((target_w, target_h), Image.LANCZOS)
            else:
                resized = img
            # Center inside image area
            x = MARGIN + (IMAGE_AREA_W - target_w) // 2
            y = MARGIN + TITLE_BAND_H + (IMAGE_AREA_H - target_h) // 2
            if resized.mode != "RGB":
                resized = resized.convert("RGB")
            page.paste(resized, (x, y))
    else:
        draw.text((MARGIN, MARGIN + TITLE_BAND_H + 200),
                  f"[render failed: {png_path.name}]",
                  fill=(220, 38, 38), font=_font(24))

    # Footer band — stats line
    footer_font = _font(22)
    label_font = _font_bold(20)

    def _fmt(s):
        return f"{s * 100:.0f}%" if s is not None else "—"

    if ctrl:
        produces = ctrl.get("produces")
        processes = ctrl.get("processes")
        effective = ctrl.get("effective")
        binding = (ctrl.get("binding") or "—")
        line_y = PAGE_H - MARGIN - 30
        x = MARGIN
        for label, value in [
            ("Mining:", _fmt(produces)),
            ("Processing:", _fmt(processes)),
            ("Effective control:", f"{_fmt(effective)} ({binding}-bound)"),
        ]:
            draw.text((x, line_y), label, fill=(30, 41, 59), font=label_font)
            x += int(draw.textlength(label, font=label_font)) + 8
            draw.text((x, line_y), value, fill=(15, 23, 42), font=footer_font)
            x += int(draw.textlength(value, font=footer_font)) + 30

    return page


def _build_title_page(plan_summary: dict[str, list[int]]) -> Image.Image:
    page = Image.new("RGB", (PAGE_W, PAGE_H), "white")
    draw = ImageDraw.Draw(page)
    h_font = _font_bold(56)
    sub_font = _font(28)
    body_font = _font(20)
    bold_body = _font_bold(20)

    draw.text((PAGE_W // 2 - 380, 360), "Critical Minerals Causal Engine",
              fill=(15, 23, 42), font=h_font)
    draw.text((PAGE_W // 2 - 280, 440), "Appendix — Yearly Grid (every 2 years)",
              fill=(71, 85, 105), font=sub_font)

    body = (
        "One KG render per page, every 2 years, per commodity.",
        "Shock origin = dominant supplier (most recent year):",
        "  graphite/REE: china   cobalt: drc   lithium: china",
        "  nickel: indonesia     uranium: russia",
        "",
        "Year-to-year visual differences are subtle; the grid documents",
        "the gradual drift in PRODUCES / PROCESSES share annotations and",
        "the effective control box. For sharper structural breaks, see the",
        "Curated Snapshots PDF.",
    )
    y = 520
    for line in body:
        draw.text((MARGIN + 100, y), line, fill=(100, 116, 139), font=body_font)
        y += 28

    y += 40
    draw.text((MARGIN + 100, y), "Year ranges by commodity:", fill=(15, 23, 42), font=bold_body)
    y += 32
    for cmd, ys in plan_summary.items():
        if not ys:
            continue
        line = f"   {cmd.replace('_', ' ').title()}: {ys[0]}–{ys[-1]} ({len(ys)} snapshots)"
        draw.text((MARGIN + 100, y), line, fill=(71, 85, 105), font=body_font)
        y += 26

    return page


def main() -> None:
    GRID_DIR.mkdir(parents=True, exist_ok=True)

    from src.minerals.knowledge_graph import CausalKnowledgeGraph
    kg_obj = CausalKnowledgeGraph.load("data/canonical/enriched_kg.json")

    # Plan
    plan: list[tuple[str, str, int, Path, dict | None]] = []
    plan_summary: dict[str, list[int]] = {}
    for commodity, origin in COMMODITY_ORIGIN.items():
        cid = kg_obj.resolve_id(commodity)
        if cid is None:
            print(f"  [skip] {commodity}: not in KG")
            continue
        y_min, y_max = _year_range_for(kg_obj, cid)
        years = list(range(y_min, y_max + 1, INTERVAL))
        if years[-1] != y_max:
            years.append(y_max)
        print(f"  {commodity}: {y_min}–{y_max}, {len(years)} snapshots")
        plan_summary[commodity] = years
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

    # Render any missing PNGs
    rendered = skipped = 0
    for commodity, origin, year, png_path, _ in plan:
        if png_path.exists():
            skipped += 1
            continue
        try:
            _render_one(kg_obj, commodity, origin, year, png_path)
            rendered += 1
        except Exception as exc:
            print(f"  [error] {commodity} {year}: {type(exc).__name__}: {exc}")
    print(f"\nRendered {rendered} new, skipped {skipped} existing.")

    # Compose pages
    print(f"\nBuilding PDF → {PDF_OUT}")
    pages: list[Image.Image] = [_build_title_page(plan_summary)]
    for commodity, origin, year, png_path, ctrl in plan:
        pages.append(_build_page(commodity, origin, year, png_path, ctrl))

    # Save as multi-page PDF. PIL embeds via flate (lossless) by default.
    first, *rest = pages
    first.save(PDF_OUT, "PDF", resolution=DPI, save_all=True, append_images=rest)

    size_mb = PDF_OUT.stat().st_size / 1024 / 1024
    print(f"Wrote {PDF_OUT} ({size_mb:.1f} MB, {len(pages)} pages).")


if __name__ == "__main__":
    main()
