# Knowledge Graphs

All thesis-ready KG figures and PDFs in one place.

## Per-commodity PDFs (1 KG per year, 1 KG per page)

Drop the matching PDF into the appendix of each commodity chapter and
reference specific year-pages from the body text.

| Commodity | PDF | Year range | Pages | Size |
|-----------|-----|-----------|-------|------|
| Graphite | [graphite/graphite_kg.pdf](graphite/graphite_kg.pdf) | 1995–2024 | 31 | 5.5 MB |
| Lithium | [lithium/lithium_kg.pdf](lithium/lithium_kg.pdf) | 1995–2024 | 31 | 5.3 MB |
| Rare Earths | [rare_earths/rare_earths_kg.pdf](rare_earths/rare_earths_kg.pdf) | 2005–2022 | 19 | 3.6 MB |
| Cobalt | [cobalt/cobalt_kg.pdf](cobalt/cobalt_kg.pdf) | 2005–2024 | 21 | 3.2 MB |
| Nickel | [nickel/nickel_kg.pdf](nickel/nickel_kg.pdf) | 2005–2024 | 21 | 3.3 MB |
| Uranium | [uranium/uranium_kg.pdf](uranium/uranium_kg.pdf) | 2003–2024 | 23 | 3.4 MB |
| Copper | [copper/copper_kg.pdf](copper/copper_kg.pdf) | 2000–2024 | 26 | 5.0 MB |
| Gallium | [gallium/gallium_kg.pdf](gallium/gallium_kg.pdf) | 2005–2024 | 21 | 4.0 MB |
| Germanium | [germanium/germanium_kg.pdf](germanium/germanium_kg.pdf) | 2005–2024 | 21 | 4.0 MB |

Each per-commodity PDF page: title (commodity · year), KG image, footer
line with Mining / Processing / Effective control percentages.

## Cross-commodity appendix PDFs

| File | What it is | Pages | Size |
|------|-----------|-------|------|
| [_share_trajectories.pdf](_share_trajectories.pdf) | Line-chart of every supplier's PRODUCES (dashed) and PROCESSES (solid) share for all 6 main commodities. 2 charts/page, landscape. | 4 | 42 KB |
| [_curated_snapshots.pdf](_curated_snapshots.pdf) | Curated structurally-significant snapshots (10 validation + 6 predictive + 8 temporal = 24 KGs), 2 KGs per page, organised by commodity. | 13 | 2.0 MB |
| [_yearly_grid_every_2yr.pdf](_yearly_grid_every_2yr.pdf) | Every-2-year KG grid for the 6 main commodities. 1 KG per page. | 77 | 13 MB |

## Per-year PNG access

Each commodity subfolder contains every-year PNGs (e.g.
`graphite/2008.png`, `graphite/2014.png`, `graphite/2022.png`). PNGs are
rendered at 200 dpi, gitignored to save space (~114 MB total), and
regen-able from the script.

## Live API access

The same PDFs are served live by the FastAPI backend:

```
GET /api/kg/commodity-pdf?commodity=graphite      # per-commodity PDF
GET /api/kg/snapshots-export                       # curated snapshots
GET /api/kg/yearly-grid-export                     # every-2-year grid
GET /api/kg/trajectory-export                      # share-trajectory charts
```

The frontend `/temporal-comparison` page has download buttons for all of
these in the header.

## Regeneration

After seed KG / yearly_share / commodity additions:

```bash
# Render every-year PNGs + per-commodity PDFs (10 min, ~140 PNGs)
python scripts/build_knowledge_graphs_folder.py

# Cross-commodity appendix PDFs
python scripts/build_kg_snapshots_pdf.py        # → outputs/kg_scenarios/kg_snapshots_appendix.pdf
python scripts/build_kg_yearly_grid_pdf.py      # → outputs/kg_scenarios/kg_yearly_grid_appendix.pdf

# Then re-copy into this folder for the consolidated view:
cp outputs/kg_scenarios/kg_snapshots_appendix.pdf "Knowledge Graphs/_curated_snapshots.pdf"
cp outputs/kg_scenarios/kg_yearly_grid_appendix.pdf "Knowledge Graphs/_yearly_grid_every_2yr.pdf"
curl -s https://causal-engine.fly.dev/api/kg/trajectory-export -o "Knowledge Graphs/_share_trajectories.pdf"
```
