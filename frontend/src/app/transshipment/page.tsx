'use client';

import { useState, useCallback, Suspense } from 'react';
import { useSearchParams } from 'next/navigation';
import RouteCard from '@/components/RouteCard';
import CircumventionCard from '@/components/CircumventionCard';
import HowToUse from '@/components/HowToUse';
import { analyzeTransshipment } from '@/lib/api';
import type { TransshipmentResponse } from '@/lib/types';

// ─── Demo data (shown when backend is offline) ────────────────────────────────
const DEMO: TransshipmentResponse = {
  commodity: 'graphite',
  source: 'China',
  destination: 'USA',
  year: 2024,
  routes: [
    { rank: 1, path: ['China', 'USA'], bottleneck_t: 24549, pct_of_source: 0.165, is_circumvention: false, non_producer_intermediaries: [], hops: 1 },
    { rank: 2, path: ['China', 'Canada', 'USA'], bottleneck_t: 2766, pct_of_source: 0.019, is_circumvention: false, non_producer_intermediaries: [], hops: 2 },
    { rank: 3, path: ['China', 'South Korea', 'USA'], bottleneck_t: 1401, pct_of_source: 0.009, is_circumvention: true, non_producer_intermediaries: ['South Korea'], hops: 2 },
    { rank: 4, path: ['China', 'Mexico', 'USA'], bottleneck_t: 289, pct_of_source: 0.002, is_circumvention: true, non_producer_intermediaries: ['Mexico'], hops: 2 },
    { rank: 5, path: ['China', 'Germany', 'Netherlands', 'USA'], bottleneck_t: 200, pct_of_source: 0.001, is_circumvention: true, non_producer_intermediaries: ['Germany', 'Netherlands'], hops: 3 },
  ],
  circumvention_rate: 0.06,
  circumvention_rate_ci: [0.06, 0.06],
  nominal_restriction_t: 58246,
  detected_rerouted_t: 3498,
  significant_hubs: ['Poland'],
  notes: ['Circular block bootstrap CI (block = T^1/3 = 3). Welch t-test at α=0.10.', 'Poland signal reflects EU battery demand growth, not evasion — interpret with caution.'],
  summary: 'Demo data — connect the FastAPI backend to run live analysis.',
};

// ─── Preset scenarios ─────────────────────────────────────────────────────────
const PRESETS = [
  { label: 'China → USA  Graphite', commodity: 'graphite', source: 'China', destination: 'USA', year: 2024 },
  { label: 'China → Germany  Lithium', commodity: 'lithium', source: 'China', destination: 'Germany', year: 2023 },
  { label: 'DRC → USA  Cobalt', commodity: 'cobalt', source: 'DRC', destination: 'USA', year: 2023 },
  { label: 'Indonesia → Japan  Nickel', commodity: 'nickel', source: 'Indonesia', destination: 'Japan', year: 2023 },
];

const YEARS = [2024, 2023, 2022, 2021, 2020, 2019, 2018];
const COMMODITIES = ['graphite', 'lithium', 'cobalt', 'nickel', 'copper'];

// ─── Helpers ──────────────────────────────────────────────────────────────────
function circumventionInterpretation(rate: number, hubs: string[]): { text: string; color: string } {
  if (rate < 0.02) return { text: 'No significant circumvention signal detected. The statistical test finds no hubs with elevated post-restriction flows.', color: 'text-emerald-700 bg-emerald-50 border-emerald-200' };
  if (rate < 0.10) return { text: `Low-to-moderate signal (${(rate * 100).toFixed(0)}%). A small fraction of restricted volume may be rerouted. Statistical confidence is limited — treat as indicative.`, color: 'text-amber-700 bg-amber-50 border-amber-200' };
  if (rate < 0.25) return { text: `Moderate circumvention detected (${(rate * 100).toFixed(0)}%)${hubs.length ? ` via ${hubs.join(', ')}` : ''}. Meaningful rerouting is likely occurring through intermediary hubs.`, color: 'text-orange-700 bg-orange-50 border-orange-200' };
  return { text: `High circumvention signal (${(rate * 100).toFixed(0)}%)${hubs.length ? ` — ${hubs.join(', ')} showing statistically significant post-restriction flow increases` : ''}. Substantial rerouting is strongly indicated.`, color: 'text-red-700 bg-red-50 border-red-200' };
}

// ─── Main form ────────────────────────────────────────────────────────────────
function TransshipmentForm() {
  const params = useSearchParams();

  const [commodity, setCommodity] = useState(params.get('commodity') ?? 'graphite');
  const [source, setSource] = useState('China');
  const [destination, setDestination] = useState('USA');
  const [year, setYear] = useState(2024);
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [maxHops, setMaxHops] = useState(4);
  const [dataPath, setDataPath] = useState('');
  const [nominalRestriction, setNominalRestriction] = useState(0.3);

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<TransshipmentResponse | null>(null);
  const [isDemo, setIsDemo] = useState(false);

  const applyPreset = (p: typeof PRESETS[0]) => {
    setCommodity(p.commodity);
    setSource(p.source);
    setDestination(p.destination);
    setYear(p.year);
    setResult(null);
    setError(null);
    setIsDemo(false);
  };

  const handleSubmit = useCallback(
    async (e: React.FormEvent) => {
      e.preventDefault();
      setLoading(true);
      setError(null);
      setResult(null);
      setIsDemo(false);
      try {
        const data = await analyzeTransshipment({
          commodity, source, destination, year,
          event_years: [year],
          max_hops: maxHops,
          data_path: dataPath,
          nominal_restriction: nominalRestriction,
        });
        setResult(data);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Analysis failed');
      } finally {
        setLoading(false);
      }
    },
    [commodity, source, destination, year, maxHops, dataPath, nominalRestriction]
  );

  const showDemo = () => {
    setResult(DEMO);
    setIsDemo(true);
    setError(null);
  };

  const maxBottleneck = result ? Math.max(...result.routes.map((r) => r.bottleneck_t), 1) : 1;
  const interp = result ? circumventionInterpretation(result.circumvention_rate, result.significant_hubs) : null;

  return (
    <div className="min-h-screen bg-zinc-50">
      <div className="max-w-5xl mx-auto px-8 py-8">
        {/* Header */}
        <div className="mb-8">
          <div className="flex items-center gap-2 mb-2">
            <span className="text-[11px] font-semibold text-indigo-600 uppercase tracking-widest">
              Trade Flow Analysis
            </span>
          </div>
          <h1 className="text-2xl font-bold text-zinc-900">Transshipment Detection</h1>
          <p className="text-sm text-zinc-500 mt-1 max-w-2xl">
            Trace multi-hop trade routes and detect circumvention of export restrictions using CEPII BACI bilateral flows (1995–2024).
          </p>
        </div>

        <HowToUse
          id="transshipment"
          steps={[
            <>Pick a <strong>Quick scenario</strong> below — or set commodity, source country, destination, and year manually.</>,
            <>Optionally add <strong>event years</strong> (when an export restriction was imposed) so the bootstrap test can compare pre-/post- volumes through suspected hubs.</>,
            <>Click <strong>Run Analysis</strong>. The backend traces top-K paths, runs a circular-block bootstrap, and returns ranked routes plus a circumvention probability.</>,
            <>Read the <strong>circumvention summary banner</strong> first (color-coded: green = clean, red = strong evasion signal). Then drill into the <strong>route cards</strong> to see which hubs absorb how much restricted volume.</>,
          ]}
          tip="Routes with non-producer intermediaries flagged in red are the prime circumvention candidates. The bootstrap CI tells you how confident the signal is."
        />

        {/* Presets */}
        <div className="mb-5">
          <p className="text-xs text-zinc-400 mb-2 font-medium">Quick scenarios</p>
          <div className="flex flex-wrap gap-2">
            {PRESETS.map((p) => (
              <button
                key={p.label}
                onClick={() => applyPreset(p)}
                className={`text-xs px-3 py-1.5 rounded-full border font-medium transition-all ${
                  commodity === p.commodity && source === p.source && destination === p.destination
                    ? 'bg-indigo-600 border-indigo-600 text-white'
                    : 'bg-white border-zinc-200 text-zinc-600 hover:border-indigo-300 hover:text-indigo-600'
                }`}
              >
                {p.label}
              </button>
            ))}
          </div>
        </div>

        {/* Config card */}
        <form onSubmit={handleSubmit} className="bg-white dark:bg-zinc-900 rounded-xl border border-zinc-200 dark:border-zinc-800 p-6 mb-6 shadow-sm">
          <div className="grid grid-cols-3 gap-4 mb-4">
            <div>
              <label className="block text-xs font-semibold text-zinc-500 mb-1.5">
                Commodity
              </label>
              <select
                value={commodity}
                onChange={(e) => setCommodity(e.target.value)}
                className="w-full text-sm border border-zinc-200 dark:border-zinc-800 rounded-lg px-3 py-2 bg-white dark:bg-zinc-900 focus:outline-none focus:ring-2 focus:ring-indigo-500"
              >
                {COMMODITIES.map((c) => (
                  <option key={c} value={c}>
                    {c.charAt(0).toUpperCase() + c.slice(1)}
                  </option>
                ))}
              </select>
            </div>
            <div>
              <label className="block text-xs font-semibold text-zinc-500 mb-1.5">
                Source country
                <span className="text-zinc-300 font-normal ml-1">(dominant exporter)</span>
              </label>
              <input
                type="text"
                value={source}
                onChange={(e) => setSource(e.target.value)}
                className="w-full text-sm border border-zinc-200 dark:border-zinc-800 rounded-lg px-3 py-2 focus:outline-none focus:ring-2 focus:ring-indigo-500"
              />
            </div>
            <div>
              <label className="block text-xs font-semibold text-zinc-500 mb-1.5">
                Destination country
              </label>
              <input
                type="text"
                value={destination}
                onChange={(e) => setDestination(e.target.value)}
                className="w-full text-sm border border-zinc-200 dark:border-zinc-800 rounded-lg px-3 py-2 focus:outline-none focus:ring-2 focus:ring-indigo-500"
              />
            </div>
          </div>

          <div className="flex items-end gap-4 mb-4">
            <div className="w-36">
              <label className="block text-xs font-semibold text-zinc-500 mb-1.5">Year</label>
              <select
                value={year}
                onChange={(e) => setYear(Number(e.target.value))}
                className="w-full text-sm border border-zinc-200 dark:border-zinc-800 rounded-lg px-3 py-2 bg-white dark:bg-zinc-900 focus:outline-none focus:ring-2 focus:ring-indigo-500"
              >
                {YEARS.map((y) => <option key={y} value={y}>{y}</option>)}
              </select>
            </div>
            <button
              type="button"
              onClick={() => setShowAdvanced(!showAdvanced)}
              className="text-xs text-zinc-400 hover:text-zinc-600 dark:text-zinc-400 transition-colors pb-2"
            >
              {showAdvanced ? '▲ Hide advanced' : '▼ Show advanced options'}
            </button>
          </div>

          {/* Advanced options */}
          {showAdvanced && (
            <div className="grid grid-cols-3 gap-4 mb-4 p-4 bg-zinc-50 dark:bg-zinc-950 rounded-lg border border-zinc-100">
              <div>
                <label className="block text-xs font-semibold text-zinc-500 mb-1.5">
                  Max hops
                  <span className="text-zinc-300 font-normal ml-1">(path depth)</span>
                </label>
                <input
                  type="number"
                  min={2} max={6}
                  value={maxHops}
                  onChange={(e) => setMaxHops(Number(e.target.value))}
                  className="w-full text-sm border border-zinc-200 dark:border-zinc-800 rounded-lg px-3 py-2 bg-white dark:bg-zinc-900 focus:outline-none focus:ring-2 focus:ring-indigo-500"
                />
              </div>
              <div>
                <label className="block text-xs font-semibold text-zinc-500 mb-1.5">
                  Restriction fraction
                  <span className="text-zinc-300 font-normal ml-1">(0–1)</span>
                </label>
                <input
                  type="number"
                  min={0} max={1} step={0.05}
                  value={nominalRestriction}
                  onChange={(e) => setNominalRestriction(Number(e.target.value))}
                  className="w-full text-sm border border-zinc-200 dark:border-zinc-800 rounded-lg px-3 py-2 bg-white dark:bg-zinc-900 focus:outline-none focus:ring-2 focus:ring-indigo-500"
                />
              </div>
              <div>
                <label className="block text-xs font-semibold text-zinc-500 mb-1.5">
                  Data path
                  <span className="text-zinc-300 font-normal ml-1">(canonical CSV dir)</span>
                </label>
                <input
                  type="text"
                  value={dataPath}
                  onChange={(e) => setDataPath(e.target.value)}
                  className="w-full text-sm border border-zinc-200 dark:border-zinc-800 rounded-lg px-3 py-2 font-mono focus:outline-none focus:ring-2 focus:ring-indigo-500"
                />
              </div>
            </div>
          )}

          <div className="flex items-center gap-3">
            <button
              type="submit"
              disabled={loading}
              className="flex items-center gap-2 px-5 py-2.5 bg-indigo-600 text-white text-sm font-semibold rounded-lg hover:bg-indigo-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors shadow-sm"
            >
              {loading ? (
                <>
                  <div className="h-4 w-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                  Analyzing…
                </>
              ) : (
                <>
                  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 20l-5.447-2.724A1 1 0 013 16.382V5.618a1 1 0 011.447-.894L9 7m0 13l6-3m-6 3V7m6 10l4.553 2.276A1 1 0 0021 18.382V7.618a1 1 0 00-.553-.894L15 4m0 13V4m0 0L9 7" />
                  </svg>
                  Run Analysis
                </>
              )}
            </button>
            {!result && !loading && (
              <button
                type="button"
                onClick={showDemo}
                className="text-sm text-zinc-400 hover:text-zinc-600 dark:text-zinc-400 transition-colors"
              >
                View sample results →
              </button>
            )}
          </div>
        </form>

        {/* Error */}
        {error && (
          <div className="rounded-xl bg-red-50 border border-red-200 px-5 py-4 mb-6">
            <p className="text-sm font-semibold text-red-700 mb-1">Analysis failed</p>
            <p className="text-sm text-red-600">{error}</p>
            <p className="text-xs text-red-400 mt-2">
              Make sure the FastAPI backend is running:{' '}
              <code className="bg-red-100 px-1 rounded">uvicorn api:app --reload</code>
              {' '}then check that{' '}
              <code className="bg-red-100 px-1 rounded">data/canonical/cepii_{commodity}.csv</code>
              {' '}exists.
            </p>
            <button
              onClick={showDemo}
              className="mt-3 text-xs font-medium text-red-600 underline hover:text-red-800"
            >
              View sample results instead
            </button>
          </div>
        )}

        {/* Empty state */}
        {!result && !loading && !error && (
          <div className="rounded-xl border border-dashed border-zinc-200 dark:border-zinc-800 bg-white dark:bg-zinc-900 px-6 py-10 text-center mb-6">
            <svg className="w-10 h-10 mx-auto text-zinc-300 mb-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M8 7h12m0 0l-4-4m4 4l-4 4m0 6H4m0 0l4 4m-4-4l4-4" />
            </svg>
            <p className="text-sm font-medium text-zinc-700 dark:text-zinc-300 mb-1">No analysis run yet</p>
            <p className="text-xs text-zinc-400 max-w-md mx-auto">
              Pick a Quick scenario above or set commodity/source/destination and click{' '}
              <span className="font-semibold text-zinc-600">Run Analysis</span> to trace
              top-K trade routes and compute a circumvention rate. Or click{' '}
              <span className="font-semibold text-zinc-600">View sample results →</span>{' '}
              to see what output looks like.
            </p>
          </div>
        )}

        {/* Demo banner */}
        {isDemo && (
          <div className="flex items-center gap-3 rounded-xl bg-amber-50 border border-amber-200 px-5 py-3 mb-6">
            <svg className="w-4 h-4 text-amber-500 shrink-0" fill="currentColor" viewBox="0 0 20 20">
              <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z" clipRule="evenodd" />
            </svg>
            <p className="text-sm text-amber-700">
              <span className="font-semibold">Sample data</span> — These are real results from a previous run. Connect the backend and click "Run Analysis" to run live.
            </p>
          </div>
        )}

        {/* Results */}
        {result && (
          <div className="space-y-6">
            {/* Interpretation headline */}
            {interp && (
              <div className={`rounded-xl border px-5 py-4 ${interp.color}`}>
                <p className="text-sm font-medium leading-relaxed">{interp.text}</p>
              </div>
            )}

            <div className="grid grid-cols-3 gap-6">
              {/* Routes */}
              <div className="col-span-2 space-y-3">
                <div className="flex items-center justify-between">
                  <h2 className="text-sm font-semibold text-zinc-700">
                    Trade routes — {result.source} → {result.destination}
                    <span className="text-zinc-400 font-normal ml-2 text-xs">
                      {result.commodity} · {result.year}
                    </span>
                  </h2>
                  <span className="text-xs text-zinc-400 bg-zinc-100 px-2 py-0.5 rounded-full">
                    {result.routes.length} routes found
                  </span>
                </div>

                {result.routes.length > 0 ? (
                  result.routes.map((r) => (
                    <RouteCard key={r.rank} route={r} maxBottleneck={maxBottleneck} />
                  ))
                ) : (
                  <div className="rounded-xl border border-dashed border-zinc-200 dark:border-zinc-800 p-8 text-center">
                    <p className="text-sm text-zinc-400">No routes found above the flow threshold.</p>
                  </div>
                )}

                {/* Summary text */}
                {result.summary && !isDemo && (
                  <div className="rounded-xl border border-zinc-200 dark:border-zinc-800 bg-white dark:bg-zinc-900 p-4 mt-4">
                    <p className="text-[11px] font-semibold text-zinc-400 uppercase tracking-wider mb-2">
                      Full report
                    </p>
                    <pre className="text-xs text-zinc-600 dark:text-zinc-400 font-mono whitespace-pre-wrap leading-relaxed">
                      {result.summary}
                    </pre>
                  </div>
                )}
              </div>

              {/* Circumvention card */}
              <div className="space-y-4">
                <CircumventionCard
                  rate={result.circumvention_rate}
                  ci={result.circumvention_rate_ci}
                  nominal_t={result.nominal_restriction_t}
                  rerouted_t={result.detected_rerouted_t}
                  hubs={result.significant_hubs}
                  notes={result.notes}
                />

                {/* Legend */}
                <div className="rounded-xl border border-zinc-200 dark:border-zinc-800 bg-white dark:bg-zinc-900 p-4">
                  <p className="text-[11px] font-semibold text-zinc-400 uppercase tracking-wider mb-3">
                    Legend
                  </p>
                  <div className="space-y-2">
                    <div className="flex items-center gap-2">
                      <div className="h-2.5 w-2.5 rounded-sm bg-indigo-600" />
                      <span className="text-xs text-zinc-600">Source country</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <div className="h-2.5 w-2.5 rounded-sm bg-blue-100 border border-blue-300" />
                      <span className="text-xs text-zinc-600">Known producer / processor</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <div className="h-2.5 w-2.5 rounded-sm bg-amber-100 border border-amber-300" />
                      <span className="text-xs text-zinc-600">Non-producer hub (⚠ suspect)</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <div className="h-2.5 w-2.5 rounded-sm bg-zinc-800" />
                      <span className="text-xs text-zinc-600">Destination country</span>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default function TransshipmentPage() {
  return (
    <Suspense>
      <TransshipmentForm />
    </Suspense>
  );
}
