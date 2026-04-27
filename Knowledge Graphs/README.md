# Knowledge Graphs

Year-by-year KG renders per critical mineral. One PNG per year, plus a
bundled PDF per commodity (1 KG per page).

## Per-commodity PDFs

- **Graphite** — [graphite_kg.pdf](graphite/graphite_kg.pdf) (1995–2024, 30 years, 5.5 MB)
- **Rare Earths** — [rare_earths_kg.pdf](rare_earths/rare_earths_kg.pdf) (2005–2022, 18 years, 3.6 MB)
- **Cobalt** — [cobalt_kg.pdf](cobalt/cobalt_kg.pdf) (2005–2024, 20 years, 3.2 MB)
- **Lithium** — [lithium_kg.pdf](lithium/lithium_kg.pdf) (1995–2024, 30 years, 5.3 MB)
- **Nickel** — [nickel_kg.pdf](nickel/nickel_kg.pdf) (2005–2024, 20 years, 3.3 MB)
- **Uranium** — [uranium_kg.pdf](uranium/uranium_kg.pdf) (2003–2024, 22 years, 3.4 MB)

## Folder structure

```
Knowledge Graphs/
├── graphite/
│   ├── 1995.png … 2024.png      (30 PNGs, gitignored — regen via script)
│   └── graphite_kg.pdf          (committed)
├── rare_earths/
├── cobalt/
├── lithium/
├── nickel/
└── uranium/
```

PNGs are gitignored (~114 MB total) and regenerated on demand. Each PNG
is 1.5 MB at dpi=200 — render fast-path (no HippoRAG) takes ~3 s each.

## Regeneration

After seed KG / yearly_share changes:

```
python scripts/build_knowledge_graphs_folder.py
```

This:
1. Re-renders any missing PNGs (skips existing ones — delete to force regen)
2. Rebuilds each commodity's `*_kg.pdf` from its current PNG set

## API access

The PDFs are served live at:

```
GET /api/kg/commodity-pdf?commodity=graphite
GET /api/kg/commodity-pdf?commodity=rare_earths
GET /api/kg/commodity-pdf?commodity=cobalt
GET /api/kg/commodity-pdf?commodity=lithium
GET /api/kg/commodity-pdf?commodity=nickel
GET /api/kg/commodity-pdf?commodity=uranium
```

The frontend `/temporal-comparison` page has a download button that
points at the currently-selected commodity's PDF.
