'use client';

import { useEffect, useMemo, useState } from 'react';
import HowToUse from '@/components/HowToUse';
import { getTemporalComparison } from '@/lib/api';
import type { TemporalSnapshot, TemporalComparisonResponse } from '@/lib/types';

const COMMODITY_LABELS: Record<string, string> = {
  graphite: 'Graphite',
  rare_earths: 'Rare Earths',
  cobalt: 'Cobalt',
  lithium: 'Lithium',
  nickel: 'Nickel',
  uranium: 'Uranium',
};

const COMMODITY_ORDER = ['graphite', 'rare_earths', 'cobalt', 'lithium', 'nickel', 'uranium'];

function formatShare(share: number | null): string {
  if (share === null) return '—';
  return `${(share * 100).toFixed(0)}%`;
}

function shareDelta(a: number | null, b: number | null): string {
  if (a === null || b === null) return '';
  const d = (b - a) * 100;
  if (Math.abs(d) < 0.5) return '';
  return `${d > 0 ? '+' : ''}${d.toFixed(0)}pp`;
}

function shareDeltaColor(a: number | null, b: number | null): string {
  if (a === null || b === null) return 'text-zinc-400';
  const d = b - a;
  if (Math.abs(d) < 0.005) return 'text-zinc-400';
  return d > 0 ? 'text-rose-600' : 'text-emerald-600';
}

export default function TemporalComparisonPage() {
  const [data, setData] = useState<TemporalComparisonResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [commodity, setCommodity] = useState('graphite');
  const [lightbox, setLightbox] = useState<TemporalSnapshot | null>(null);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    getTemporalComparison()
      .then((d) => {
        if (cancelled) return;
        setData(d);
      })
      .catch((e) => {
        if (cancelled) return;
        setError(e instanceof Error ? e.message : 'Failed to load');
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => { cancelled = true; };
  }, []);

  const series = data?.[commodity] ?? [];
  const orderedCommodities = useMemo(
    () => COMMODITY_ORDER.filter((c) => data && c in data),
    [data],
  );

  return (
    <div className="flex flex-col h-screen bg-zinc-50">
      {/* Header */}
      <div className="border-b border-zinc-200 bg-white px-6 py-3 shrink-0">
        <div className="flex items-center justify-between">
          <div>
            <p className="text-[10px] font-semibold text-indigo-600 uppercase tracking-widest mb-0.5">
              Year-by-Year KG Snapshots
            </p>
            <h1 className="text-lg font-bold text-zinc-900">
              Temporal Comparison
              <span className="ml-2 text-sm font-normal text-zinc-400">
                Show how supply chains shift over time
              </span>
            </h1>
          </div>
          <div className="flex items-center gap-2">
            <label className="text-[10px] font-semibold text-zinc-400 uppercase tracking-wider">
              Commodity
            </label>
            <select
              value={commodity}
              onChange={(e) => setCommodity(e.target.value)}
              className="text-sm border border-zinc-200 rounded-lg px-2.5 py-1.5 bg-white focus:outline-none focus:ring-2 focus:ring-indigo-500"
            >
              {orderedCommodities.map((c) => (
                <option key={c} value={c}>{COMMODITY_LABELS[c] ?? c}</option>
              ))}
            </select>
          </div>
        </div>
      </div>

      <div className="flex-1 overflow-y-auto p-6">
        <HowToUse
          id="temporal-comparison"
          steps={[
            <>Pick a <strong>commodity</strong> from the header dropdown to see its 3 year-snapshots side-by-side (pre / during / post a structural break).</>,
            <>Each card shows the <strong>year</strong>, the <strong>effective control</strong> (max of PRODUCES and PROCESSES share for the dominant supplier at that year), and the <strong>binding stage</strong> (mining vs. processing).</>,
            <>The <strong>delta badges</strong> between cards quantify the structural shift — e.g. graphite went from 65% (2008, processing-bound) to 80% (2015) to 95% (2022).</>,
            <>Click any KG image to expand. Use this view to drive thesis figures showing supply-chain concentration drift.</>,
          ]}
          tip="Effective control values come from the live enriched KG via effective_control_at(country, commodity, year), which substitutes CEPII PRODUCES shares and USGS PROCESSES shares dynamically per year."
        />

        {loading && (
          <div className="flex items-center gap-3 text-zinc-500 text-sm py-10 justify-center">
            <div className="h-4 w-4 border-2 border-indigo-200 border-t-indigo-600 rounded-full animate-spin" />
            Loading temporal series…
          </div>
        )}

        {error && (
          <div className="bg-red-50 border border-red-200 text-red-800 text-sm rounded-lg p-3 mb-4">
            {error}
          </div>
        )}

        {!loading && series.length > 0 && (
          <>
            {/* Series headline */}
            <div className="mb-5">
              <h2 className="text-2xl font-bold text-zinc-900">
                {COMMODITY_LABELS[commodity] ?? commodity}
              </h2>
              <p className="text-sm text-zinc-500 mt-1">
                {series.length} year snapshots ·{' '}
                <span className="font-mono">
                  {series[0].year}–{series[series.length - 1].year}
                </span>{' '}
                · shock origin: {Array.from(new Set(series.map((s) => s.shock_origin))).join(' / ')}
              </p>
            </div>

            {/* Year strip */}
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-4 mb-6">
              {series.map((snap, i) => {
                const prev = i > 0 ? series[i - 1] : null;
                return (
                  <div key={snap.scenario_id} className="bg-white border border-zinc-200 rounded-xl overflow-hidden">
                    {/* Year + meta header */}
                    <div className="px-4 py-3 border-b border-zinc-100 bg-zinc-50">
                      <div className="flex items-baseline justify-between mb-1">
                        <p className="text-2xl font-bold text-zinc-900 font-mono">{snap.year}</p>
                        {prev && shareDelta(prev.effective_share, snap.effective_share) && (
                          <span className={`text-xs font-semibold ${shareDeltaColor(prev.effective_share, snap.effective_share)}`}>
                            {shareDelta(prev.effective_share, snap.effective_share)} vs. {prev.year}
                          </span>
                        )}
                      </div>
                      <p className="text-[11px] text-zinc-600 leading-snug">{snap.title}</p>
                    </div>

                    {/* Effective control stats */}
                    <div className="px-4 py-3 border-b border-zinc-100 grid grid-cols-2 gap-3">
                      <div>
                        <p className="text-[9px] font-semibold text-zinc-400 uppercase tracking-wider mb-0.5">Effective control</p>
                        <p className="text-sm font-bold text-zinc-800">{formatShare(snap.effective_share)}</p>
                      </div>
                      <div>
                        <p className="text-[9px] font-semibold text-zinc-400 uppercase tracking-wider mb-0.5">Binding stage</p>
                        <p className="text-sm font-bold text-zinc-800 capitalize">{snap.binding ?? '—'}</p>
                      </div>
                    </div>

                    {/* KG image */}
                    {snap.available ? (
                      <button
                        onClick={() => setLightbox(snap)}
                        className="block w-full focus:outline-none focus:ring-2 focus:ring-indigo-400 cursor-zoom-in"
                      >
                        {/* eslint-disable-next-line @next/next/no-img-element */}
                        <img
                          src={snap.image_url}
                          alt={snap.title}
                          className="w-full h-auto block"
                          loading="lazy"
                        />
                      </button>
                    ) : (
                      <div className="px-4 py-10 bg-amber-50 border-amber-200 text-amber-800 text-xs text-center">
                        PNG not built yet for this snapshot.
                      </div>
                    )}
                  </div>
                );
              })}
            </div>

            {/* Cumulative shift */}
            {series.length >= 2 && series[0].effective_share !== null && series[series.length - 1].effective_share !== null && (
              <div className="bg-indigo-50 border border-indigo-100 rounded-xl px-5 py-4">
                <p className="text-[10px] font-semibold text-indigo-700 uppercase tracking-wider mb-2">
                  Cumulative structural shift, {series[0].year} → {series[series.length - 1].year}
                </p>
                <p className="text-sm text-zinc-800 leading-relaxed">
                  Effective control of <span className="font-mono">{COMMODITY_LABELS[commodity]?.toLowerCase() ?? commodity}</span>{' '}
                  by <span className="font-mono">{series[0].shock_origin}</span> shifted from{' '}
                  <strong className="text-zinc-900">{formatShare(series[0].effective_share)}</strong>{' '}
                  ({series[0].binding ?? 'unknown'}-bound) in {series[0].year} to{' '}
                  <strong className="text-zinc-900">{formatShare(series[series.length - 1].effective_share)}</strong>{' '}
                  ({series[series.length - 1].binding ?? 'unknown'}-bound) in {series[series.length - 1].year}{' '}
                  — a{' '}
                  <span className={`font-semibold ${shareDeltaColor(series[0].effective_share, series[series.length - 1].effective_share)}`}>
                    {shareDelta(series[0].effective_share, series[series.length - 1].effective_share)}
                  </span>{' '}
                  change over {series[series.length - 1].year - series[0].year} years.
                </p>
              </div>
            )}
          </>
        )}
      </div>

      {/* Lightbox */}
      {lightbox && (
        <div
          onClick={() => setLightbox(null)}
          className="fixed inset-0 bg-zinc-900/80 flex items-center justify-center z-50 cursor-zoom-out p-6"
        >
          <div className="max-w-6xl max-h-full overflow-auto bg-white rounded-xl">
            <div className="px-5 py-3 border-b border-zinc-100 flex items-center justify-between">
              <p className="text-sm font-semibold text-zinc-800">{lightbox.title}</p>
              <a
                href={lightbox.image_url}
                download={`${lightbox.scenario_id}.png`}
                onClick={(e) => e.stopPropagation()}
                className="text-xs px-3 py-1.5 border border-indigo-200 bg-indigo-50 text-indigo-700 rounded-lg hover:bg-indigo-100"
              >
                Download PNG
              </a>
            </div>
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img
              src={lightbox.image_url}
              alt={lightbox.title}
              className="block max-w-full h-auto"
            />
          </div>
        </div>
      )}
    </div>
  );
}
