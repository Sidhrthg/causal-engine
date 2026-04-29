# Presentation Day Checklist

## Pre-presentation (do this 24-48hr before)

### 1. Backend deploy
The live Fly.io backend at `https://causal-engine.fly.dev` is BEHIND main.
Live `/api/commodities` returns 6 items; main has 8 (added Ge/Ga).
Live deployment also lacks the rare_earths_2014 OOS pair.

```bash
flyctl deploy   # from repo root; uses Dockerfile
# Verify after deploy:
curl https://causal-engine.fly.dev/api/commodities | jq .commodities
# Should show: graphite, lithium, cobalt, nickel, copper, soybeans,
# rare_earths, uranium, germanium, gallium  (10 items including soybeans)
```

If `flyctl` is not installed: `brew install flyctl && flyctl auth login`.

### 2. Frontend build verification
```bash
cd frontend
npm install
npm run build       # catches type errors
npm run dev         # spot-check pages render at http://localhost:3000
```
If `npm run build` fails, send the error and stop here.

### 3. Vercel deploy
If Vercel is auto-deploying from `main`, the frontend updates already.
Verify at your Vercel project URL (or trigger manually if needed).

### 4. Smoke test live (do this RIGHT before presenting)
Visit each page in the deployed app and click the primary action:

- [ ] **Landing** — 8 commodity cards visible, hero text mentions all 8
- [ ] **Knowledge Query** — ask "why is graphite critical?" → returns answer + sources
- [ ] **Knowledge Graph** — select graphite → entities/edges render
- [ ] **Transshipment** — China → USA, graphite, 2022 → routes table populates
- [ ] **Counterfactual** — pick a scenario, click Run → trajectory chart renders
- [ ] **Shock Extractor** — paste "China imposed 30% restriction on graphite" → shocks parsed
- [ ] **Scenario Builder** — pick a preset → KG image renders
- [ ] **Temporal Comparison** — select graphite → 3 snapshot images render
- [ ] **KG Enrich** — single query mode (skip batch — slow)

If any step shows a 404 or error, the live backend hasn't deployed.

## During presentation — recommended demo path

1. **Open with the landing page** — sets context (8 commodities, Pearl L1/L2/L3)
2. **Knowledge Graph** — show graphite → demonstrates the structured supply-chain registry
3. **Temporal Comparison** — graphite 2008 → 2015 → 2022 — show how China's processing share evolved (the regime break that drives Group A/B distinction)
4. **Scenario Builder** — pick a preset → show the do-calculus interventions
5. **Counterfactual** — Pearl L3 in action; explain abduction step
6. **Transshipment** — graphite China → USA → show 6% circumvention rate finding
7. **Shock Extractor** — paste a recent news headline → shocks parse → predict price trajectory
8. **Knowledge Query** — natural-language Q&A as a coda

Skip kg-enrich during presentation — batch run takes 3-5 min, single run takes ~30s. Mention it exists; don't run it live.

## What's intentionally NOT in the frontend (for now)

- Predictive hazard model (v1 results) — lives on `predictive-experimental` branch only. Mentioned in defense_sections.md if examiners ask. Not appropriate for live demo because the result is a null finding.
- Gallium/germanium forward scenarios — backend supports it via `/api/scenario/run` if you specify germanium/gallium as commodity, but no dedicated frontend page yet.

## If something breaks live

- **Frontend looks broken:** check browser DevTools network tab — usually a 4xx from the rewrite to fly.dev. Either fly is down (`fly status`) or backend missed the deploy.
- **Single page errors:** `/api/health` to confirm backend up; specific page might use an endpoint that's not on the deployed image yet.
- **Fallback:** screenshots/recordings of each page work as backup.

## Known cosmetic items (not blockers)

- Landing page KG entity/edge counts (3,132 / 31,916) are hardcoded; could be live-fetched but not critical
- Some pages don't have rare_earths/uranium/Ge/Ga in their dropdowns yet (the audit fixed query, kg-enrich, landing — but counterfactual/shock-extractor/scenario-builder dropdowns weren't audited individually)
- KG enrich progress indicator could be more granular for the 3-5 min batch run

These are polish items, not presentation blockers.

## Branch state at presentation time

| Branch | What it is | Where it should be deployed |
|---|---|---|
| `main` | Frontend + backend with 8 commodities | Vercel (frontend) + Fly.io (backend) |
| `predictive-experimental` | Hazard model lab work | Not deployed; for thesis defense Q&A only |

Make sure you're on `main` for the demo (`git checkout main`). The presentation app should serve from main only.
