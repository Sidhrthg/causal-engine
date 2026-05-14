'use client';

import { useEffect, useState } from 'react';
import HowToUse from '@/components/HowToUse';
import LineChart from '@/components/LineChart';
import MathPanel from '@/components/MathPanel';
import OutputGuide from '@/components/OutputGuide';
import {
  getForecastCommodities,
  runForecast,
  type ForecastResponse,
} from '@/lib/api';

type Severity = 'baseline' | 'mild_ban' | 'full_ban' | 'severe_ban' | 'custom';

const SEVERITY_LABELS: Record<Severity, string> = {
  baseline:   'Baseline (no shock)',
  mild_ban:   'Mild ban (30% restriction, 2yr)',
  full_ban:   'Full ban (30% restriction, 3yr)',
  severe_ban: 'Severe ban (50% restriction, 4yr)',
  custom:     'Custom',
};

interface CommodityRow {
  commodity: string;
  in_sample_DA: number | null;
  oos_DA: number | null;
}

export default function ForecastPage() {
  const [commodities, setCommodities] = useState<CommodityRow[]>([]);
  const [commodity, setCommodity] = useState('graphite');
  const [shockYear, setShockYear] = useState(2025);
  const [severity, setSeverity] = useState<Severity>('full_ban');
  const [customMag, setCustomMag] = useState(0.30);
  const [customDur, setCustomDur] = useState(3);
  const [demandSurge, setDemandSurge] = useState<number | ''>('');
  const [horizon, setHorizon] = useState(10);
  const [includeCI, setIncludeCI] = useState(true);

  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<ForecastResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    getForecastCommodities()
      .then((d) => setCommodities(d.commodities))
      .catch(() => {});
  }, []);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const r = await runForecast({
        commodity,
        shock_year: shockYear,
        severity,
        restriction_magnitude: severity === 'custom' ? customMag : undefined,
        restriction_duration:  severity === 'custom' ? customDur : undefined,
        demand_surge: demandSurge === '' ? undefined : Number(demandSurge),
        horizon_years: horizon,
        n_bootstrap: includeCI ? 24 : 0,
      });
      setResult(r);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Forecast failed');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex flex-col h-screen bg-zinc-50 dark:bg-zinc-950">
      {/* Header */}
      <div className="border-b border-zinc-200 dark:border-zinc-800 bg-white dark:bg-zinc-900 px-6 py-3 shrink-0">
        <p className="text-[10px] font-semibold text-indigo-600 uppercase tracking-widest mb-0.5">
          Forward Price Projection
        </p>
        <h1 className="text-lg font-bold text-zinc-900 dark:text-zinc-100">
          Forecast
          <span className="ml-2 text-sm font-normal text-zinc-400">
            10-year price path under an export-restriction shock, with accuracy attribution
          </span>
        </h1>
      </div>

      <div className="flex flex-1 overflow-hidden">
        {/* Sidebar form */}
        <div className="w-80 border-r border-zinc-200 dark:border-zinc-800 bg-white dark:bg-zinc-900 p-5 overflow-y-auto shrink-0">
          <HowToUse
            id="forecast"
            steps={[
              <>Pick a <strong>commodity</strong>. Each runs with its calibrated <code>α_P, η_D, τ_K, g</code> from the most recent episode.</>,
              <>Set the <strong>shock year</strong> and a <strong>severity preset</strong>, or pick <strong>Custom</strong> to set magnitude (0–1) and duration in years directly.</>,
              <>Optionally add a <strong>demand surge</strong> (e.g. 0.10 = +10% demand) to stack a demand shock on top.</>,
              <>Click <strong>Run forecast</strong>. Returns the price path vs. baseline, peak year, and the year prices fall back within ±10% of baseline ("normalisation").</>,
              <>The accuracy panel shows in-sample DA and OOS DA for that commodity from the calibration runs — these are honest numbers, not advertising.</>,
            ]}
            tip="A 1σ confidence band (gray) comes from re-running with parameters jittered ±10% — wider band means the trajectory is sensitive to calibration."
          />

          <form onSubmit={handleSubmit} className="flex flex-col gap-4 mt-5">
            <div>
              <label className="text-[10px] font-semibold text-zinc-400 uppercase tracking-wider mb-1.5 block">
                Commodity
              </label>
              <select
                value={commodity}
                onChange={(e) => setCommodity(e.target.value)}
                className="w-full text-sm border border-zinc-200 dark:border-zinc-800 rounded-lg px-3 py-2 bg-white dark:bg-zinc-900 focus:outline-none focus:ring-2 focus:ring-indigo-500"
              >
                {commodities.map((c) => (
                  <option key={c.commodity} value={c.commodity}>
                    {c.commodity.replace(/_/g, ' ')}
                  </option>
                ))}
              </select>
            </div>

            <div>
              <label className="text-[10px] font-semibold text-zinc-400 uppercase tracking-wider mb-1.5 block">
                Shock Year
              </label>
              <input
                type="number" min={2000} max={2035}
                value={shockYear}
                onChange={(e) => setShockYear(Number(e.target.value))}
                className="w-full text-sm border border-zinc-200 dark:border-zinc-800 rounded-lg px-3 py-2 bg-white dark:bg-zinc-900 focus:outline-none focus:ring-2 focus:ring-indigo-500"
              />
            </div>

            <div>
              <label className="text-[10px] font-semibold text-zinc-400 uppercase tracking-wider mb-1.5 block">
                Severity
              </label>
              <select
                value={severity}
                onChange={(e) => setSeverity(e.target.value as Severity)}
                className="w-full text-sm border border-zinc-200 dark:border-zinc-800 rounded-lg px-3 py-2 bg-white dark:bg-zinc-900 focus:outline-none focus:ring-2 focus:ring-indigo-500"
              >
                {(Object.keys(SEVERITY_LABELS) as Severity[]).map((s) => (
                  <option key={s} value={s}>{SEVERITY_LABELS[s]}</option>
                ))}
              </select>
            </div>

            {severity === 'custom' && (
              <div className="flex gap-2">
                <div className="flex-1">
                  <label className="text-[10px] font-semibold text-zinc-400 uppercase tracking-wider mb-1.5 block">
                    Magnitude (0–1)
                  </label>
                  <input
                    type="number" step={0.05} min={0} max={1}
                    value={customMag}
                    onChange={(e) => setCustomMag(Number(e.target.value))}
                    className="w-full text-sm border border-zinc-200 dark:border-zinc-800 rounded-lg px-3 py-2 bg-white dark:bg-zinc-900 focus:outline-none focus:ring-2 focus:ring-indigo-500"
                  />
                </div>
                <div className="flex-1">
                  <label className="text-[10px] font-semibold text-zinc-400 uppercase tracking-wider mb-1.5 block">
                    Duration (yr)
                  </label>
                  <input
                    type="number" min={1} max={20}
                    value={customDur}
                    onChange={(e) => setCustomDur(Number(e.target.value))}
                    className="w-full text-sm border border-zinc-200 dark:border-zinc-800 rounded-lg px-3 py-2 bg-white dark:bg-zinc-900 focus:outline-none focus:ring-2 focus:ring-indigo-500"
                  />
                </div>
              </div>
            )}

            <div>
              <label className="text-[10px] font-semibold text-zinc-400 uppercase tracking-wider mb-1.5 block">
                Demand surge (optional, e.g. 0.10)
              </label>
              <input
                type="number" step={0.05} min={0} max={2}
                value={demandSurge}
                onChange={(e) => setDemandSurge(e.target.value === '' ? '' : Number(e.target.value))}
                placeholder="0.00"
                className="w-full text-sm border border-zinc-200 dark:border-zinc-800 rounded-lg px-3 py-2 bg-white dark:bg-zinc-900 focus:outline-none focus:ring-2 focus:ring-indigo-500"
              />
            </div>

            <div>
              <label className="text-[10px] font-semibold text-zinc-400 uppercase tracking-wider mb-1.5 block">
                Horizon (years)
              </label>
              <input
                type="number" min={1} max={20}
                value={horizon}
                onChange={(e) => setHorizon(Number(e.target.value))}
                className="w-full text-sm border border-zinc-200 dark:border-zinc-800 rounded-lg px-3 py-2 bg-white dark:bg-zinc-900 focus:outline-none focus:ring-2 focus:ring-indigo-500"
              />
            </div>

            <label className="flex items-center gap-2 text-xs text-zinc-600 dark:text-zinc-400">
              <input
                type="checkbox"
                checked={includeCI}
                onChange={(e) => setIncludeCI(e.target.checked)}
                className="rounded border-zinc-300"
              />
              Include 1σ confidence band (slower)
            </label>

            <button
              type="submit"
              disabled={loading}
              className="mt-2 w-full bg-gradient-to-r from-indigo-600 to-violet-600 text-white text-sm font-semibold py-2.5 rounded-lg hover:opacity-95 disabled:opacity-40 transition-opacity"
            >
              {loading ? 'Running…' : 'Run forecast'}
            </button>
          </form>
        </div>

        {/* Result panel */}
        <div className="flex-1 overflow-auto p-6">
          <div className="mb-5 grid grid-cols-1 xl:grid-cols-2 gap-5">
            <OutputGuide
              rung="L2"
              intro="The forecast result has 5 stat chips, a baseline-vs-scenario price chart, optional ±1σ confidence band, and a calibration-accuracy panel."
              fields={[
                { name: 'Direction',
                  meaning: '↑/↓/flat at the peak of the scenario relative to baseline (price ratio > 1.05 / < 0.95 / between)',
                  read: '↑ Up → scenario peak rose at least 5% above the no-shock counterfactual at that year' },
                { name: 'Peak vs baseline',
                  meaning: 'max_t [ P_scenario(t) / P_baseline(t) ] in the post-shock horizon. The cleanest single number for "how big was the shock impact".',
                  read: '1.68× means scenario price was at most 68% above the no-shock baseline at the peak year' },
                { name: 'Peak year',
                  meaning: 'the year where peak-vs-baseline was reached',
                  read: 'Restriction starts 2025; peak year 2026 means price peaks one year into the restriction' },
                { name: 'Normalises',
                  meaning: 'the first year after the restriction ends from which scenario stays within ±10% of baseline for the rest of the horizon. "Never" means the model does not return to baseline within the simulated window.',
                  read: '2030 (+3yr post-end) → 3 years after restriction end, prices return to within ±10% of where they would have been without the shock' },
                { name: 'Restriction',
                  meaning: 'the do(·) value and active years for the export-restriction shock',
                  read: '30% · 2025–2027 → export_restriction pinned to 0.30 from 2025 through 2027 inclusive' },
                { name: 'CI band (gray fan)',
                  meaning: '1σ confidence band from 24 parameter perturbations (±10% on α_P, η_D, τ_K). Wide band = forecast sensitive to calibration uncertainty.',
                  read: 'If the band crosses the baseline line, the shock direction is not robust to parameter uncertainty' },
                { name: 'in_sample_DA / oos_DA',
                  meaning: 'historical directional accuracy of this commodity\'s calibration. 1.0 = perfect sign-of-change calls, 0.5 = chance.',
                  read: 'graphite: in_sample 1.0, oos 0.467 → fits its own period perfectly but the regime shift makes cross-period transfer hard' },
              ]}
              takeaway="A high Peak vs baseline + late or 'Never' normalisation = severe scar from the shock. Use the CI band to gauge whether the headline direction is solid or marginal."
            />
            <MathPanel
              rung="L2"
              title="What this page computes (thin L2 — supply-shock interventions only)"
              formal="P(Y | do(export_restriction = m  for years [t_a, t_b]))   —   restrict export supply, integrate ODE forward"
              equations={[
                {
                  label: "Demand-side equation (price elasticity η_D enters here)",
                  code:
`D_t  =  D_0 · g^t · (P_t / P_ref)^η_D
          · (1 − policy.substitution) · (1 − policy.efficiency)
          · (1 + demand_surge_t) · demand_destruction_mult_t`,
                },
                {
                  label: "Supply-side equation (α_P, τ_K, capacity utilisation)",
                  code:
`u_t   =  clamp( u_0 + β_u · log(P_t / P_ref),  u_min,  u_max )
Q_t   =  K_t · u_t
Q_eff,t = Q_t · (1 − shock.export_restriction_t)
          · policy_supply_mult_t · capacity_supply_mult_t

dK/dt =  (K_target(P_t) − K_t) / τ_K          [capacity adjusts at rate 1/τ_K]
dP/dt =  α_P · (shortage_t − λ_cover · inventory_gap_t) + σ_P · dW_t`,
                },
                {
                  label: "Calibrated parameters {α_P, η_D, τ_K, g} per episode",
                  code:
`Per-mineral values are fitted by differential evolution to maximise
DA + Spearman ρ vs. CEPII BACI bilateral unit values
(see src/minerals/predictability.py lines 71–89).`,
                },
              ]}
              caveat="This page is a convenience L2 interface for supply-shock interventions. For the canonical do-calculus surface — intervening on any structural parameter (η_D, τ_K, substitution_elasticity, fringe_capacity_share, …) via explicit graph surgery — use the L2 — Intervention page."
              source="src/minerals/model.py — step(); calibration in src/minerals/predictability.py"
            />
          </div>

          {error && (
            <div className="bg-red-50 border border-red-200 text-red-800 text-sm rounded-lg p-3 mb-4">
              <p className="font-semibold mb-1">Forecast failed</p>
              <p className="text-red-700">{error}</p>
            </div>
          )}

          {!loading && !result && !error && (
            <div className="flex flex-col items-center justify-center h-full text-center text-zinc-400">
              <p className="text-sm font-medium text-zinc-500">No forecast yet</p>
              <p className="text-xs mt-1 max-w-sm">
                Pick a commodity, year, and severity on the left, then click Run forecast.
              </p>
            </div>
          )}

          {loading && (
            <div className="flex flex-col items-center justify-center h-full text-center">
              <div className="h-10 w-10 border-2 border-indigo-200 border-t-indigo-600 rounded-full animate-spin mb-4" />
              <p className="text-sm font-medium text-zinc-700 dark:text-zinc-300">
                Running forward simulation{includeCI ? ' + bootstrap' : ''}…
              </p>
            </div>
          )}

          {result && !loading && <ForecastView r={result} />}
        </div>
      </div>
    </div>
  );
}

function ForecastView({ r }: { r: ForecastResponse }) {
  const years = r.scenario_path.map((p) => p.year);
  const baselineSeries = r.baseline_path.map((p) => p.price_index);
  const scenarioSeries = r.scenario_path.map((p) => p.price_index);

  const directionLabel = r.direction === 'up' ? '▲ Up' : r.direction === 'down' ? '▼ Down' : '→ Flat';
  const directionColor = r.direction === 'up' ? 'text-red-600' : r.direction === 'down' ? 'text-emerald-600' : 'text-zinc-500';

  const normLabel = r.normalization_year !== null
    ? `${r.normalization_year} (+${r.normalization_lag_years}yr post-end)`
    : 'never within window';

  return (
    <div className="flex flex-col gap-5">
      <div className="flex flex-wrap gap-2">
        <Stat label="Direction" value={directionLabel} valueClass={directionColor} />
        <Stat
          label="Peak vs baseline"
          value={r.peak_vs_baseline ? `${r.peak_vs_baseline.toFixed(2)}×` : '—'}
        />
        <Stat label="Peak year" value={r.peak_year ?? '—'} />
        <Stat label="Normalises" value={normLabel} />
        <Stat label="Restriction" value={`${(r.restriction_magnitude * 100).toFixed(0)}% · ${r.restriction_start}–${r.restriction_end}`} />
      </div>

      <div className="bg-white dark:bg-zinc-900 border border-zinc-200 dark:border-zinc-800 rounded-lg p-4">
        <LineChart
          title={`Price index (P / P_${r.shock_year - 1})`}
          years={years}
          height={260}
          series={[
            { label: 'baseline (no shock)', data: baselineSeries, color: '#94a3b8' },
            { label: `scenario (${r.severity})`, data: scenarioSeries, color: '#6366f1' },
          ]}
        />
        <div className="flex flex-wrap gap-4 mt-3 text-[11px] text-zinc-500">
          <span>
            Restriction window: <span className="font-mono text-zinc-700">{r.restriction_start}–{r.restriction_end}</span>
          </span>
          {r.normalization_year !== null && (
            <span>
              Returns to baseline at <span className="font-mono text-zinc-700">{r.normalization_year}</span>
            </span>
          )}
          {r.ci_band && (
            <span className="text-zinc-400">1σ band from {r.ci_band.length} parameter perturbations</span>
          )}
        </div>
      </div>

      {r.ci_band && (
        <CIDisplay
          band={r.ci_band}
          scenario={scenarioSeries}
          years={years}
        />
      )}

      <div className="grid grid-cols-2 gap-4">
        <div className="bg-white dark:bg-zinc-900 border border-zinc-200 dark:border-zinc-800 rounded-lg p-4">
          <p className="text-[10px] font-semibold text-zinc-400 uppercase tracking-wider mb-3">
            Calibration accuracy ({r.commodity})
          </p>
          <div className="grid grid-cols-2 gap-3">
            <AccuracyTile
              label="In-sample DA"
              value={r.accuracy.in_sample_DA}
              detail={r.accuracy.in_sample_episodes.length
                ? r.accuracy.in_sample_episodes.join(', ')
                : 'no in-sample episode'}
            />
            <AccuracyTile
              label="Out-of-sample DA"
              value={r.accuracy.oos_DA}
              detail={r.accuracy.oos_pairs.length
                ? r.accuracy.oos_pairs.join(', ')
                : 'no OOS pair available'}
            />
          </div>
          <p className="text-[10px] text-zinc-400 mt-3 leading-relaxed">
            DA = directional accuracy (fraction of years where the model gets the
            sign of the price change correct). 1.000 = perfect, 0.500 = chance.
          </p>
        </div>

        <div className="bg-white dark:bg-zinc-900 border border-zinc-200 dark:border-zinc-800 rounded-lg p-4">
          <p className="text-[10px] font-semibold text-zinc-400 uppercase tracking-wider mb-3">
            Parameters used
          </p>
          <div className="grid grid-cols-2 gap-3 text-xs">
            <ParamTile name="α_P" value={r.params_used.alpha_P} desc="price → capacity response" />
            <ParamTile name="η_D" value={r.params_used.eta_D} desc="demand price elasticity" />
            <ParamTile name="τ_K" value={r.params_used.tau_K} desc="capacity adjustment time (yr)" />
            <ParamTile name="g"   value={r.params_used.g}      desc="background demand growth" />
          </div>
        </div>
      </div>

      <p className="text-[10px] text-zinc-400 leading-relaxed">
        Method: Pearl Layer 2 (Intervention) — equivalent to{' '}
        <span className="font-mono">do(export_restriction = {r.restriction_magnitude.toFixed(2)})</span>{' '}
        applied for {r.restriction_end - r.restriction_start + 1} year(s) starting {r.restriction_start},
        ODE integrated over the {r.horizon_years}-year horizon. This page uses{' '}
        <span className="font-mono">run_scenario()</span> with the shock injected as a forcing
        term — mathematically the same intervention. For the canonical do-calculus
        interface that performs explicit graph surgery (mutilated DAG +
        identifiability check), see <span className="font-mono">POST /api/pearl/l2/do</span>.
        Normalisation = first post-restriction year where |scenario − baseline| &lt; 10% of baseline.
      </p>
    </div>
  );
}

function CIDisplay({
  band, scenario, years,
}: {
  band: { year: number; low: number; high: number }[];
  scenario: number[];
  years: number[];
}) {
  const lows = band.map((b) => b.low);
  const highs = band.map((b) => b.high);
  return (
    <div className="bg-white dark:bg-zinc-900 border border-zinc-200 dark:border-zinc-800 rounded-lg p-4">
      <LineChart
        title="Scenario with 1σ confidence band"
        years={years}
        height={220}
        series={[
          { label: 'low (−1σ)', data: lows, color: '#cbd5e1' },
          { label: 'forecast', data: scenario, color: '#6366f1' },
          { label: 'high (+1σ)', data: highs, color: '#cbd5e1' },
        ]}
      />
    </div>
  );
}

function Stat({ label, value, valueClass }: { label: string; value: string | number; valueClass?: string }) {
  return (
    <div className="bg-white dark:bg-zinc-900 border border-zinc-200 dark:border-zinc-800 rounded-lg px-3 py-2">
      <p className="text-[9px] font-semibold text-zinc-400 uppercase tracking-wider">{label}</p>
      <p className={`text-sm font-bold ${valueClass ?? 'text-zinc-800 dark:text-zinc-100'}`}>{value}</p>
    </div>
  );
}

function AccuracyTile({ label, value, detail }: { label: string; value: number | null; detail: string }) {
  const v = value === null ? '—' : value.toFixed(3);
  return (
    <div>
      <p className="text-[9px] font-semibold text-zinc-400 uppercase tracking-wider">{label}</p>
      <p className="text-xl font-bold text-zinc-800 dark:text-zinc-100">{v}</p>
      <p className="text-[10px] text-zinc-500 mt-0.5 leading-snug">{detail}</p>
    </div>
  );
}

function ParamTile({ name, value, desc }: { name: string; value: number; desc: string }) {
  return (
    <div>
      <p className="text-[10px] text-zinc-400">{desc}</p>
      <p className="text-sm font-mono font-semibold text-zinc-800 dark:text-zinc-100">
        {name} = {value.toFixed(3)}
      </p>
    </div>
  );
}
