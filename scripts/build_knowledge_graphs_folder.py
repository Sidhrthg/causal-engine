#!/usr/bin/env python3
"""
Build a top-level "Knowledge Graphs/" folder at the project root, one
subfolder per critical mineral, containing:
  - one PNG per year (every year in the seed yearly_share range)
  - a per-commodity PDF (1 KG per page) bundling all years for thesis use

Run after seed KG / yearly_share changes:
    python scripts/build_knowledge_graphs_folder.py

Output structure:
    Knowledge Graphs/
        graphite/
            1995.png ... 2024.png
            graphite_kg.pdf
        rare_earths/...
        cobalt/...
        lithium/...
        nickel/...
        uranium/...

PNGs are rendered fast-path (no HippoRAG), so each takes ~3-5s. Fully
running takes ~10 minutes for ~140 KGs total.
"""

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
OUT_ROOT = ROOT / "Knowledge Graphs"

# Dominant supplier per commodity (kept consistent across years for series view)
COMMODITY_ORIGIN = {
    "graphite": "china",
    "rare_earths": "china",
    "cobalt": "drc",
    "lithium": "china",
    "nickel": "indonesia",
    "uranium": "russia",
    "copper": "china",     # processing-bound (chile dominates mining, china refining)
    "gallium": "china",    # ~85% PRODUCES + ~80% PROCESSES
    "germanium": "china",  # ~62% PRODUCES from zinc smelting byproduct
}

# Page geometry for per-commodity PDFs (landscape A4 at 200 DPI)
DPI = 200
PAGE_W = int(11.0 * DPI)
PAGE_H = int(8.5 * DPI)
MARGIN = int(0.4 * DPI)


def _font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = ["/System/Library/Fonts/Helvetica.ttc"]
    for path in candidates:
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size, index=1 if bold else 0)
            except Exception:
                try:
                    return ImageFont.truetype(path, size)
                except Exception:
                    continue
    return ImageFont.load_default()


def _year_range(kg, commodity_id: str) -> tuple[int, int]:
    from src.minerals.knowledge_graph import RelationType
    years: set[int] = set()
    for u, v, data in kg._graph.edges(data=True):
        if v != commodity_id:
            continue
        rel = data["relationship"]
        if rel.relation_type not in (RelationType.PRODUCES, RelationType.PROCESSES):
            continue
        for y in (rel.properties or {}).get("yearly_share", {}).keys():
            years.add(int(y))
    return (min(years), max(years)) if years else (2010, 2022)


def _render_year(kg_obj, commodity: str, origin: str, year: int, out_path: Path) -> None:
    from scripts.run_knowledge_graph import _render_scenario as render_fn
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sid = f"{commodity}_{origin}_{year}"
    scenario = {
        "year": int(year), "shock_origin": origin, "commodity": commodity,
        "title": f"{commodity.replace('_', ' ').title()} {year} — {origin.upper()} snapshot",
    }
    render_fn(
        kg_obj=kg_obj, scenario_id=sid, scenario=scenario,
        output_path=str(out_path),
        pipeline=None, extractor=None,
        enriched=True,
    )


def _control(kg_obj, commodity: str, origin: str, year: int) -> dict | None:
    try:
        c = kg_obj.effective_control_at(origin, commodity, year)
        if not c:
            return None
        return {
            "produces": c.get("produces_share"),
            "processes": c.get("processes_share"),
            "effective": c.get("effective_share"),
            "binding": c.get("binding"),
        }
    except Exception:
        return None


def _build_page(commodity: str, origin: str, year: int, png_path: Path,
                ctrl: dict | None) -> Image.Image:
    page = Image.new("RGB", (PAGE_W, PAGE_H), "white")
    draw = ImageDraw.Draw(page)

    title_font = _font(40, bold=True)
    sub_font = _font(20)
    label_font = _font(20, bold=True)
    value_font = _font(22)

    draw.text((MARGIN, MARGIN),
              f"{commodity.replace('_', ' ').title()} · {year}",
              fill=(15, 23, 42), font=title_font)
    draw.text((MARGIN, MARGIN + 50),
              f"shock origin: {origin.upper()}",
              fill=(100, 116, 139), font=sub_font)

    # KG image
    img_top = MARGIN + 110
    footer_h = 80
    img_max_w = PAGE_W - 2 * MARGIN
    img_max_h = PAGE_H - img_top - footer_h - MARGIN
    if png_path.exists():
        with Image.open(png_path) as img:
            img.load()
            sw, sh = img.size
            scale = min(img_max_w / sw, img_max_h / sh)
            if scale < 1.0:
                img = img.resize((int(sw * scale), int(sh * scale)), Image.LANCZOS)
            if img.mode != "RGB":
                img = img.convert("RGB")
            x = MARGIN + (img_max_w - img.width) // 2
            page.paste(img, (x, img_top))

    # Footer stats
    if ctrl:
        def _fmt(s):
            return f"{s * 100:.0f}%" if s is not None else "—"
        line_y = PAGE_H - MARGIN - 30
        x = MARGIN
        for label, value in [
            ("Mining:", _fmt(ctrl.get("produces"))),
            ("Processing:", _fmt(ctrl.get("processes"))),
            ("Effective control:",
             f"{_fmt(ctrl.get('effective'))} ({(ctrl.get('binding') or '—')}-bound)"),
        ]:
            draw.text((x, line_y), label, fill=(30, 41, 59), font=label_font)
            x += int(draw.textlength(label, font=label_font)) + 8
            draw.text((x, line_y), value, fill=(15, 23, 42), font=value_font)
            x += int(draw.textlength(value, font=value_font)) + 30

    return page


def _build_title_page(commodity: str, origin: str, years: list[int]) -> Image.Image:
    page = Image.new("RGB", (PAGE_W, PAGE_H), "white")
    draw = ImageDraw.Draw(page)
    draw.text((PAGE_W // 2 - 380, 320),
              f"{commodity.replace('_', ' ').title()} — Knowledge Graphs",
              fill=(15, 23, 42), font=_font(56, bold=True))
    draw.text((PAGE_W // 2 - 200, 410),
              f"Year-by-year supply-chain renders",
              fill=(71, 85, 105), font=_font(24))
    body_font = _font(20)
    body = (
        f"{len(years)} pages, one KG render per year from {years[0]} to {years[-1]}.",
        f"Shock origin: {origin.upper()}.",
        "",
        "Each page: KG snapshot at year T with year-specific PRODUCES /",
        "PROCESSES share annotations on edges, plus the dominant supplier's",
        "effective control percentage and binding stage in the footer.",
    )
    y = 500
    for line in body:
        draw.text((MARGIN + 100, y), line, fill=(71, 85, 105), font=body_font)
        y += 28
    return page


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    from src.minerals.knowledge_graph import CausalKnowledgeGraph
    kg_obj = CausalKnowledgeGraph.load("data/canonical/enriched_kg.json")

    summary: dict[str, dict] = {}

    for commodity, origin in COMMODITY_ORIGIN.items():
        cid = kg_obj.resolve_id(commodity)
        if cid is None:
            print(f"[skip] {commodity}: not in KG")
            continue
        y_min, y_max = _year_range(kg_obj, cid)
        years = list(range(y_min, y_max + 1))
        commodity_dir = OUT_ROOT / commodity
        commodity_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{commodity}: {y_min}–{y_max} ({len(years)} years)")
        rendered = skipped = 0
        png_paths: list[tuple[int, Path, dict | None]] = []
        for year in years:
            png_path = commodity_dir / f"{year}.png"
            if not png_path.exists():
                try:
                    _render_year(kg_obj, commodity, origin, year, png_path)
                    rendered += 1
                except Exception as exc:
                    print(f"  [error] {year}: {type(exc).__name__}: {exc}")
                    continue
            else:
                skipped += 1
            png_paths.append((year, png_path, _control(kg_obj, commodity, origin, year)))
        print(f"  rendered {rendered} new, skipped {skipped} existing")

        # Build per-commodity PDF
        pdf_path = commodity_dir / f"{commodity}_kg.pdf"
        pages = [_build_title_page(commodity, origin, [p[0] for p in png_paths])]
        for year, png_path, ctrl in png_paths:
            pages.append(_build_page(commodity, origin, year, png_path, ctrl))
        if pages:
            first, *rest = pages
            first.save(pdf_path, "PDF", resolution=DPI,
                       save_all=True, append_images=rest)
            size_mb = pdf_path.stat().st_size / 1024 / 1024
            print(f"  → {pdf_path.relative_to(ROOT)} ({size_mb:.1f} MB, {len(pages)} pages)")
            summary[commodity] = {
                "years": [p[0] for p in png_paths],
                "pdf": str(pdf_path.relative_to(ROOT)),
                "size_mb": round(size_mb, 2),
            }

    # Write a top-level README pointing to all the PDFs
    readme = OUT_ROOT / "README.md"
    lines = [
        "# Knowledge Graphs",
        "",
        "Year-by-year KG renders per critical mineral. One PNG per year,",
        "plus a bundled PDF per commodity.",
        "",
        "## Per-commodity PDFs",
        "",
    ]
    for commodity, info in summary.items():
        # info['pdf'] is "Knowledge Graphs/<commodity>/<commodity>_kg.pdf";
        # README sits at "Knowledge Graphs/README.md", so the link is
        # "<commodity>/<commodity>_kg.pdf".
        rel_link = f"{commodity}/{Path(info['pdf']).name}"
        lines.append(f"- **{commodity.replace('_', ' ').title()}**: "
                     f"[{Path(info['pdf']).name}]({rel_link}) "
                     f"({info['size_mb']} MB, {info['years'][0]}–{info['years'][-1]}, "
                     f"{len(info['years'])} pages)")
    lines.extend([
        "",
        "## Generation",
        "",
        "Rebuild after seed KG changes:",
        "",
        "```",
        "python scripts/build_knowledge_graphs_folder.py",
        "```",
    ])
    readme.write_text("\n".join(lines) + "\n")
    print(f"\nWrote {readme.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
