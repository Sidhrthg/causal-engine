'use client';

import { useState } from 'react';
import HowToUse from '@/components/HowToUse';
import { extractShocks, predictFromText } from '@/lib/api';
import type { ExtractedShock, PredictFromTextResponse } from '@/lib/types';

const COMMODITIES = ['graphite', 'lithium', 'cobalt', 'nickel', 'copper', 'soybeans'];

const EXAMPLES = [
  {
    label: 'China graphite ban (2023)',
    text: 'China imposed export restrictions on graphite in October 2023, requiring companies to obtain licenses before shipping natural and synthetic graphite abroad. The ban came amid surging EV demand driving a spike in global graphite consumption.',
    commodity: 'graphite',
  },
  {
    label: 'Cobalt DRC strike',
    text: 'A major strike at Glencore\'s Mutanda mine in the DRC caused a shutdown of cobalt production in 2019, reducing global cobalt supply significantly and driving a demand surge from battery manufacturers.',
    commodity: 'cobalt',
  },
  {
    label: 'Lithium EV boom (2022)',
    text: 'Lithium prices surged in 2022 amid a boom in electric vehicle demand. Chilean lithium exports saw a dramatic spike as carmakers raced to secure battery supply chains.',
    commodity: 'lithium',
  },
  {
    label: 'Soybeans trade war (2018)',
    text: 'The US-China trade war in 2018 led to a collapse in US soybean exports to China, as Beijing imposed retaliatory tariffs. China shifted purchases to Brazil, causing a decline in US agricultural revenues.',
    commodity: 'soybeans',
  },
];

function ShockBadge({ type }: { type: string }) {
  const colors: Record<string, string> = {
    export_restriction: 'bg-red-50 text-red-700 border-red-200',
    demand_surge: 'bg-blue-50 text-blue-700 border-blue-200',
    capex_shock: 'bg-amber-50 text-amber-700 border-amber-200',
    stockpile_release: 'bg-emerald-50 text-emerald-700 border-emerald-200',
    macro_demand_shock: 'bg-purple-50 text-purple-700 border-purple-200',
  };
  return (
    <span className={`text-[10px] font-semibold px-2 py-0.5 rounded-full border ${colors[type] ?? 'bg-zinc-50 text-zinc-600 border-zinc-200'}`}>
      {type.replace(/_/g, ' ')}
    </span>
  );
}

function MagnitudeBar({ magnitude }: { magnitude: number }) {
  const abs = Math.abs(magnitude);
  const pct = Math.min(abs * 100, 100);
  const color = magnitude < 0 ? 'bg-blue-400' : 'bg-red-400';
  return (
    <div className="flex items-center gap-2">
      <div className="w-20 h-1.5 bg-zinc-100 rounded-full overflow-hidden">
        <div className={`h-full rounded-full ${color}`} style={{ width: `${pct}%` }} />
      </div>
      <span className="text-[11px] font-mono text-zinc-500">
        {magnitude > 0 ? '+' : ''}{(magnitude * 100).toFixed(0)}%
      </span>
    </div>
  );
}

function TrajectoryChart({ trajectory }: { trajectory: PredictFromTextResponse['trajectory'] }) {
  if (!trajectory.length) return null;
  const maxP = Math.max(...trajectory.map((r) => r.P));
  const minP = Math.min(...trajectory.map((r) => r.P));
  const range = maxP - minP || 1;

  return (
    <div className="mt-4">
      <p className="text-xs font-semibold text-zinc-500 mb-2 uppercase tracking-wider">Price Index</p>
      <div className="flex items-end gap-1 h-24">
        {trajectory.map((row) => {
          const height = ((row.P - minP) / range) * 80 + 8;
          return (
            <div key={row.year} className="flex flex-col items-center flex-1 group">
              <div
                className="w-full bg-indigo-500 rounded-t-sm transition-all group-hover:bg-indigo-600 relative"
                style={{ height: `${height}px` }}
                title={`${row.year}: P=${row.P.toFixed(3)}, shortage=${row.shortage.toFixed(1)}`}
              >
                {row.shortage > 0 && (
                  <div className="absolute top-0 left-0 right-0 h-1 bg-red-400 rounded-t-sm" />
                )}
              </div>
              <span className="text-[9px] text-zinc-400 mt-1">{row.year}</span>
            </div>
          );
        })}
      </div>
      <p className="text-[10px] text-zinc-400 mt-1">Red stripe = shortage · hover for values</p>
    </div>
  );
}

export default function ShockExtractorPage() {
  const [text, setText] = useState('');
  const [commodity, setCommodity] = useState('graphite');
  const [startYear, setStartYear] = useState(2023);
  const [endYear, setEndYear] = useState(2026);

  const [extracting, setExtracting] = useState(false);
  const [predicting, setPredicting] = useState(false);
  const [shocks, setShocks] = useState<ExtractedShock[] | null>(null);
  const [result, setResult] = useState<PredictFromTextResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  const loadExample = (ex: (typeof EXAMPLES)[0]) => {
    setText(ex.text);
    setCommodity(ex.commodity);
    setShocks(null);
    setResult(null);
    setError(null);
  };

  const handleExtract = async () => {
    if (!text.trim()) return;
    setExtracting(true);
    setShocks(null);
    setResult(null);
    setError(null);
    try {
      const resp = await extractShocks({ text });
      setShocks(resp.shocks);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Extraction failed');
    } finally {
      setExtracting(false);
    }
  };

  const handlePredict = async () => {
    if (!text.trim()) return;
    setPredicting(true);
    setResult(null);
    setError(null);
    try {
      const resp = await predictFromText({ text, commodity, start_year: startYear, end_year: endYear });
      setResult(resp);
      setShocks(resp.extracted_shocks);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Prediction failed');
    } finally {
      setPredicting(false);
    }
  };

  const commodityShocks = shocks?.filter((s) => s.commodity === commodity) ?? [];
  const otherShocks = shocks?.filter((s) => s.commodity !== commodity) ?? [];

  return (
    <div className="min-h-screen bg-zinc-50 px-6 py-8">
      <div className="max-w-4xl mx-auto">
        {/* Header */}
        <div className="mb-6">
          <p className="text-[10px] font-semibold text-indigo-600 uppercase tracking-widest mb-1">
            Option C · Event Extraction → Shock Graph
          </p>
          <h1 className="text-xl font-bold text-zinc-900">Text → Causal Prediction</h1>
          <p className="text-sm text-zinc-500 mt-1">
            Paste a news article or policy document. The causal KG extracts shocks and runs the ODE model.
          </p>
        </div>

        <HowToUse
          id="shock-extractor"
          steps={[
            <>Paste a news article, policy announcement, or any prose describing a commodity event into the <strong>Text</strong> box. (Or click an <strong>Example</strong> chip above to prefill.)</>,
            <>Pick the relevant <strong>commodity</strong> and the <strong>year range</strong> over which to simulate the trajectory.</>,
            <>Click <strong>Extract Shocks</strong> to see what the LLM identified, then <strong>Run Prediction</strong> to feed those shocks into the ODE model and project a price/quantity trajectory.</>,
            <>The chart shows the projected <strong>price index</strong> (red stripe = shortage years). Hover any bar for exact values.</>,
          ]}
          tip="Best results: text that names countries, specific actions (ban, surge, strike, tariff), and dates. Vague prose may extract zero shocks."
        />

        {/* Examples */}
        <div className="flex flex-wrap gap-2 mb-5">
          {EXAMPLES.map((ex) => (
            <button
              key={ex.label}
              onClick={() => loadExample(ex)}
              className="text-xs px-3 py-1.5 bg-white border border-zinc-200 rounded-lg hover:border-indigo-300 text-zinc-600 hover:text-indigo-600 transition-all"
            >
              {ex.label}
            </button>
          ))}
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Left: Input */}
          <div className="flex flex-col gap-4">
            <div>
              <label className="text-xs font-semibold text-zinc-500 uppercase tracking-wider block mb-1.5">
                Text
              </label>
              <textarea
                value={text}
                onChange={(e) => { setText(e.target.value); setShocks(null); setResult(null); }}
                rows={10}
                placeholder="Paste a news article, policy announcement, or any text describing a commodity market event…"
                className="w-full text-sm border border-zinc-200 rounded-xl px-4 py-3 resize-none focus:outline-none focus:ring-2 focus:ring-indigo-500 bg-white leading-relaxed"
              />
            </div>

            {/* Controls */}
            <div className="grid grid-cols-3 gap-3">
              <div>
                <label className="text-[10px] font-semibold text-zinc-400 uppercase tracking-wider block mb-1">
                  Commodity
                </label>
                <select
                  value={commodity}
                  onChange={(e) => setCommodity(e.target.value)}
                  className="w-full text-sm border border-zinc-200 rounded-lg px-2.5 py-1.5 bg-white focus:outline-none focus:ring-2 focus:ring-indigo-500"
                >
                  {COMMODITIES.map((c) => <option key={c} value={c}>{c}</option>)}
                </select>
              </div>
              <div>
                <label className="text-[10px] font-semibold text-zinc-400 uppercase tracking-wider block mb-1">
                  Start Year
                </label>
                <input
                  type="number"
                  value={startYear}
                  onChange={(e) => setStartYear(Number(e.target.value))}
                  className="w-full text-sm border border-zinc-200 rounded-lg px-2.5 py-1.5 bg-white focus:outline-none focus:ring-2 focus:ring-indigo-500"
                />
              </div>
              <div>
                <label className="text-[10px] font-semibold text-zinc-400 uppercase tracking-wider block mb-1">
                  End Year
                </label>
                <input
                  type="number"
                  value={endYear}
                  onChange={(e) => setEndYear(Number(e.target.value))}
                  className="w-full text-sm border border-zinc-200 rounded-lg px-2.5 py-1.5 bg-white focus:outline-none focus:ring-2 focus:ring-indigo-500"
                />
              </div>
            </div>

            <div className="flex gap-2">
              <button
                onClick={handleExtract}
                disabled={!text.trim() || extracting || predicting}
                className="flex-1 py-2.5 text-sm font-semibold border border-indigo-300 text-indigo-600 rounded-xl hover:bg-indigo-50 disabled:opacity-40 transition-colors"
              >
                {extracting ? 'Extracting…' : 'Extract Shocks'}
              </button>
              <button
                onClick={handlePredict}
                disabled={!text.trim() || extracting || predicting}
                className="flex-1 py-2.5 text-sm font-semibold bg-indigo-600 text-white rounded-xl hover:bg-indigo-700 disabled:opacity-40 transition-colors shadow-sm"
              >
                {predicting ? 'Running…' : 'Predict Trajectory'}
              </button>
            </div>

            {error && (
              <div className="bg-red-50 border border-red-200 rounded-xl p-3 text-sm text-red-700">
                {error}
              </div>
            )}
          </div>

          {/* Right: Results */}
          <div className="flex flex-col gap-4">
            {/* Trajectory */}
            {result && (
              <div className="bg-white border border-zinc-200 rounded-xl p-4">
                <div className="flex items-center justify-between mb-3">
                  <p className="text-xs font-semibold text-zinc-700 uppercase tracking-wider">
                    {result.commodity} · {result.trajectory[0]?.year}–{result.trajectory.at(-1)?.year}
                  </p>
                  <span className="text-[10px] text-zinc-400">
                    {result.n_shocks_applied} shocks applied
                  </span>
                </div>

                <TrajectoryChart trajectory={result.trajectory} />

                <div className="grid grid-cols-3 gap-2 mt-4">
                  {Object.entries(result.metrics).slice(0, 3).map(([k, v]) => (
                    <div key={k} className="bg-zinc-50 rounded-lg p-2 text-center">
                      <p className="text-[9px] text-zinc-400 uppercase tracking-wider mb-0.5">
                        {k.replace(/_/g, ' ')}
                      </p>
                      <p className="text-sm font-bold text-zinc-800">{v.toFixed(2)}</p>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Extracted shocks */}
            {shocks !== null && (
              <div className="bg-white border border-zinc-200 rounded-xl p-4">
                <p className="text-xs font-semibold text-zinc-500 uppercase tracking-wider mb-3">
                  Extracted Shocks ({shocks.length} total)
                </p>

                {commodityShocks.length > 0 && (
                  <div className="mb-3">
                    <p className="text-[10px] font-semibold text-indigo-500 uppercase tracking-wider mb-2">
                      {commodity} ({commodityShocks.length})
                    </p>
                    <div className="flex flex-col gap-2">
                      {commodityShocks.map((s, i) => (
                        <div key={i} className="border border-zinc-100 rounded-lg p-3 bg-zinc-50/50">
                          <div className="flex items-center gap-2 mb-1.5">
                            <ShockBadge type={s.shock.type} />
                            <span className="text-[10px] text-zinc-400">
                              {s.shock.start_year}–{s.shock.end_year}
                            </span>
                          </div>
                          <MagnitudeBar magnitude={s.shock.magnitude} />
                          <p className="text-[10px] text-zinc-500 mt-1.5 leading-relaxed">{s.reasoning}</p>
                          {s.affected_entities.length > 0 && (
                            <div className="flex flex-wrap gap-1 mt-1.5">
                              {s.affected_entities.slice(0, 5).map((e) => (
                                <span key={e} className="text-[9px] bg-zinc-100 text-zinc-500 px-1.5 py-0.5 rounded">
                                  {e}
                                </span>
                              ))}
                              {s.affected_entities.length > 5 && (
                                <span className="text-[9px] text-zinc-400">
                                  +{s.affected_entities.length - 5} more
                                </span>
                              )}
                            </div>
                          )}
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {otherShocks.length > 0 && (
                  <div>
                    <p className="text-[10px] font-semibold text-zinc-400 uppercase tracking-wider mb-2">
                      Other commodities ({otherShocks.length})
                    </p>
                    <div className="flex flex-col gap-1.5">
                      {otherShocks.map((s, i) => (
                        <div key={i} className="flex items-center gap-2 text-[11px] text-zinc-500">
                          <span className="font-medium text-zinc-700">{s.commodity}</span>
                          <ShockBadge type={s.shock.type} />
                          <MagnitudeBar magnitude={s.shock.magnitude} />
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {shocks.length === 0 && (
                  <p className="text-sm text-zinc-400 text-center py-4">
                    No shocks extracted. Try adding country + commodity + action keywords.
                  </p>
                )}
              </div>
            )}

            {!shocks && !result && (
              <div className="bg-white border border-dashed border-zinc-200 rounded-xl p-8 text-center">
                <div className="h-10 w-10 bg-indigo-50 rounded-xl flex items-center justify-center mx-auto mb-3">
                  <svg className="w-5 h-5 text-indigo-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                  </svg>
                </div>
                <p className="text-sm text-zinc-500">Paste text and click Extract or Predict</p>
                <p className="text-xs text-zinc-400 mt-1">
                  Looks for country · action · commodity patterns
                </p>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
