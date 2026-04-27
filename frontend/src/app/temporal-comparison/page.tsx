'use client';

import { useEffect, useMemo, useRef, useState } from 'react';
import HowToUse from '@/components/HowToUse';
import {
  getTemporalComparison,
  getYearlyShares,
  getYearSnapshot,
} from '@/lib/api';
import type {
  TemporalSnapshot,
  TemporalComparisonResponse,
  YearlySharesResponse,
  YearSnapshotResponse,
} from '@/lib/types';

const COMMODITY_LABELS: Record<string, string> = {
  graphite: 'Graphite',
  rare_earths: 'Rare Earths',
  cobalt: 'Cobalt',
  lithium: 'Lithium',
  nickel: 'Nickel',
  uranium: 'Uranium',
};

const COMMODITY_ORDER = ['graphite', 'rare_earths', 'cobalt', 'lithium', 'nickel', 'uranium'];

const COUNTRY_COLOR: Record<string, string> = {
  china: '#dc2626',
  drc: '#a16207',
  indonesia: '#7c3aed',
  australia: '#16a34a',
  chile: '#2563eb',
  russia: '#475569',
  kazakhstan: '#0891b2',
  canada: '#dc6e1c',
  philippines: '#9333ea',
  mozambique: '#65a30d',
  madagascar: '#84cc16',
  brazil: '#10b981',
};

function colorFor(country: string): string {
  return COUNTRY_COLOR[country] ?? '#94a3b8';
}

function fmtShare(s: number | null | undefined): string {
  if (s === null || s === undefined) return '—';
  return `${(s * 100).toFixed(0)}%`;
}

function shareDelta(a: number | null | undefined, b: number | null | undefined): string {
  if (a == null || b == null) return '';
  const d = (b - a) * 100;
  if (Math.abs(d) < 0.5) return '';
  return `${d > 0 ? '+' : ''}${d.toFixed(0)}pp`;
}

function deltaColor(a: number | null | undefined, b: number | null | undefined): string {
  if (a == null || b == null) return 'text-zinc-400';
  const d = b - a;
  if (Math.abs(d) < 0.005) return 'text-zinc-400';
  return d > 0 ? 'text-rose-600' : 'text-emerald-600';
}

// ─── Share trajectory line chart ─────────────────────────────────────────────

function ShareChart({
  data,
  highlightYear,
  onYearClick,
}: {
  data: YearlySharesResponse;
  highlightYear: number | null;
  onYearClick: (year: number) => void;
}) {
  const W = 720, H = 280, PAD_L = 40, PAD_R = 12, PAD_T = 16, PAD_B = 28;
  const innerW = W - PAD_L - PAD_R;
  const innerH = H - PAD_T - PAD_B;

  const yearMin = data.years[0];
  const yearMax = data.years[data.years.length - 1];
  const yearRange = Math.max(1, yearMax - yearMin);

  const x = (year: number) => PAD_L + ((year - yearMin) / yearRange) * innerW;
  const y = (share: number) => PAD_T + (1 - share) * innerH;

  // Filter to top 6 series by peak share to avoid clutter
  const topSeries = useMemo(() => {
    const ranked = [...data.series].sort((a, b) => {
      const ap = Math.max(...a.share.map((v) => v ?? 0));
      const bp = Math.max(...b.share.map((v) => v ?? 0));
      return bp - ap;
    });
    return ranked.slice(0, 8);
  }, [data]);

  const tickYears = [yearMin, yearMin + Math.floor(yearRange / 2), yearMax];
  const tickShares = [0, 0.25, 0.5, 0.75, 1.0];

  const buildPath = (series: { share: (number | null)[] }) => {
    const pts: string[] = [];
    let started = false;
    series.share.forEach((v, i) => {
      if (v === null || v === undefined) {
        started = false;
        return;
      }
      const cx = x(data.years[i]);
      const cy = y(v);
      pts.push(`${started ? 'L' : 'M'}${cx.toFixed(1)},${cy.toFixed(1)}`);
      started = true;
    });
    return pts.join(' ');
  };

  return (
    <div className="bg-white border border-zinc-200 rounded-xl p-4">
      <div className="flex items-center justify-between mb-3">
        <p className="text-[10px] font-semibold text-zinc-400 uppercase tracking-wider">
          Share trajectory · {COMMODITY_LABELS[data.commodity] ?? data.commodity}
        </p>
        <p className="text-[10px] text-zinc-400">
          {yearMin}–{yearMax} · top {topSeries.length} suppliers
        </p>
      </div>

      <svg viewBox={`0 0 ${W} ${H}`} className="w-full h-auto">
        {/* Y-axis grid + labels */}
        {tickShares.map((s) => (
          <g key={s}>
            <line x1={PAD_L} x2={W - PAD_R} y1={y(s)} y2={y(s)} stroke="#f1f5f9" strokeWidth={1} />
            <text x={PAD_L - 4} y={y(s) + 3} fontSize="9" fill="#94a3b8" textAnchor="end" fontFamily="monospace">
              {(s * 100).toFixed(0)}%
            </text>
          </g>
        ))}

        {/* X-axis */}
        <line x1={PAD_L} y1={H - PAD_B} x2={W - PAD_R} y2={H - PAD_B} stroke="#cbd5e1" strokeWidth={1} />
        {tickYears.map((yr) => (
          <text key={yr} x={x(yr)} y={H - 8} fontSize="9" fill="#64748b" textAnchor="middle" fontFamily="monospace">
            {yr}
          </text>
        ))}

        {/* Highlight year vertical line */}
        {highlightYear !== null && (
          <line
            x1={x(highlightYear)} x2={x(highlightYear)}
            y1={PAD_T} y2={H - PAD_B}
            stroke="#6366f1" strokeWidth={1.5} strokeDasharray="3,3"
          />
        )}

        {/* Series paths */}
        {topSeries.map((s, i) => {
          const dashed = s.kind === 'produces';
          return (
            <path
              key={`${s.country}-${s.kind}-${i}`}
              d={buildPath(s)}
              fill="none"
              stroke={colorFor(s.country)}
              strokeWidth={2}
              strokeDasharray={dashed ? '4,3' : undefined}
              opacity={0.85}
            />
          );
        })}

        {/* Click overlay on each year column */}
        {data.years.map((yr) => (
          <rect
            key={yr}
            x={x(yr) - innerW / data.years.length / 2}
            y={PAD_T}
            width={innerW / data.years.length}
            height={innerH}
            fill="transparent"
            onClick={() => onYearClick(yr)}
            style={{ cursor: 'pointer' }}
          >
            <title>{yr} · click to load KG snapshot</title>
          </rect>
        ))}
      </svg>

      {/* Legend */}
      <div className="flex flex-wrap gap-x-4 gap-y-1 mt-3 text-[10px] text-zinc-600">
        {topSeries.map((s, i) => (
          <span key={`${s.country}-${s.kind}-${i}`} className="flex items-center gap-1.5">
            <svg width={20} height={6}>
              <line x1={0} y1={3} x2={20} y2={3} stroke={colorFor(s.country)} strokeWidth={2}
                    strokeDasharray={s.kind === 'produces' ? '4,3' : undefined} />
            </svg>
            <span className="font-medium">{s.country}</span>
            <span className="text-zinc-400">· {s.kind}</span>
          </span>
        ))}
      </div>
    </div>
  );
}

// ─── Page ────────────────────────────────────────────────────────────────────

export default function TemporalComparisonPage() {
  const [commodity, setCommodity] = useState('graphite');
  const [data, setData] = useState<TemporalComparisonResponse | null>(null);
  const [shares, setShares] = useState<YearlySharesResponse | null>(null);
  const [loadingMeta, setLoadingMeta] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Year slider state
  const [year, setYear] = useState<number | null>(null);
  const [snapshot, setSnapshot] = useState<YearSnapshotResponse | null>(null);
  const [snapshotLoading, setSnapshotLoading] = useState(false);
  const [snapshotError, setSnapshotError] = useState<string | null>(null);

  const [lightbox, setLightbox] = useState<TemporalSnapshot | YearSnapshotResponse | null>(null);

  const series = data?.[commodity] ?? [];
  const orderedCommodities = useMemo(
    () => COMMODITY_ORDER.filter((c) => data && c in data),
    [data],
  );

  // Load both datasets when commodity changes
  useEffect(() => {
    let cancelled = false;
    setLoadingMeta(true);
    setError(null);
    Promise.all([getTemporalComparison(), getYearlyShares(commodity)])
      .then(([d, s]) => {
        if (cancelled) return;
        setData(d);
        setShares(s);
        // Default the slider to the latest year in the share trajectory
        if (s.years.length > 0) setYear(s.years[s.years.length - 1]);
      })
      .catch((e) => {
        if (cancelled) return;
        setError(e instanceof Error ? e.message : 'Failed to load');
      })
      .finally(() => { if (!cancelled) setLoadingMeta(false); });
    return () => { cancelled = true; };
  }, [commodity]);

  // Debounced snapshot fetch when slider year changes
  const snapshotTimer = useRef<number | null>(null);
  useEffect(() => {
    if (year === null) return;
    if (snapshotTimer.current !== null) window.clearTimeout(snapshotTimer.current);
    snapshotTimer.current = window.setTimeout(() => {
      setSnapshotLoading(true);
      setSnapshotError(null);
      getYearSnapshot({ commodity, year })
        .then((r) => setSnapshot(r))
        .catch((e) => setSnapshotError(e instanceof Error ? e.message : 'Snapshot failed'))
        .finally(() => setSnapshotLoading(false));
    }, 200);
  }, [commodity, year]);

  const yearMin = shares?.years[0] ?? 2000;
  const yearMax = shares?.years[shares.years.length - 1] ?? 2024;

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
              {orderedCommodities.length > 0
                ? orderedCommodities.map((c) => (
                    <option key={c} value={c}>{COMMODITY_LABELS[c] ?? c}</option>
                  ))
                : COMMODITY_ORDER.map((c) => (
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
            <>Pick a <strong>commodity</strong> from the header dropdown.</>,
            <>The <strong>share trajectory chart</strong> shows every supplier&apos;s PRODUCES (dashed) and PROCESSES (solid) share across all years in the KG.</>,
            <>The <strong>year slider</strong> below the chart fetches an on-demand KG snapshot for any year (cached after first render — ~2-5s the first time, instant after).</>,
            <>The <strong>3-card year strip</strong> at the bottom shows pre-rendered structurally significant snapshots with inline image and stats.</>,
          ]}
          tip="Click any year on the chart to jump the slider. Shifts in the binding stage (mining→processing) show up as crossover points where the dashed PRODUCES line falls below the solid PROCESSES line."
        />

        {loadingMeta && (
          <div className="flex items-center gap-3 text-zinc-500 text-sm py-10 justify-center">
            <div className="h-4 w-4 border-2 border-indigo-200 border-t-indigo-600 rounded-full animate-spin" />
            Loading temporal data…
          </div>
        )}

        {error && (
          <div className="bg-red-50 border border-red-200 text-red-800 text-sm rounded-lg p-3 mb-4">
            {error}
          </div>
        )}

        {!loadingMeta && shares && shares.series.length > 0 && (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 mb-6">
            <ShareChart data={shares} highlightYear={year} onYearClick={setYear} />

            <YearSliderPanel
              year={year ?? yearMax}
              yearMin={yearMin}
              yearMax={yearMax}
              onYearChange={setYear}
              snapshot={snapshot}
              loading={snapshotLoading}
              error={snapshotError}
              onExpand={(s) => setLightbox(s)}
            />
          </div>
        )}

        {!loadingMeta && series.length > 0 && (
          <>
            <h2 className="text-xs font-semibold text-zinc-400 uppercase tracking-wider mb-3">
              Pre-rendered structural snapshots
            </h2>
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-4 mb-6">
              {series.map((snap, i) => {
                const prev = i > 0 ? series[i - 1] : null;
                return (
                  <div key={snap.scenario_id} className="bg-white border border-zinc-200 rounded-xl overflow-hidden">
                    <div className="px-4 py-3 border-b border-zinc-100 bg-zinc-50">
                      <div className="flex items-baseline justify-between mb-1">
                        <p className="text-2xl font-bold text-zinc-900 font-mono">{snap.year}</p>
                        {prev && shareDelta(prev.effective_share, snap.effective_share) && (
                          <span className={`text-xs font-semibold ${deltaColor(prev.effective_share, snap.effective_share)}`}>
                            {shareDelta(prev.effective_share, snap.effective_share)} vs. {prev.year}
                          </span>
                        )}
                      </div>
                      <p className="text-[11px] text-zinc-600 leading-snug">{snap.title}</p>
                    </div>
                    <div className="px-4 py-2 border-b border-zinc-100 grid grid-cols-3 gap-2 text-center">
                      <div>
                        <p className="text-[8px] font-semibold text-zinc-400 uppercase tracking-wider mb-0.5">Mining</p>
                        <p className="text-xs font-bold text-zinc-800">{fmtShare(snap.produces_share)}</p>
                      </div>
                      <div>
                        <p className="text-[8px] font-semibold text-zinc-400 uppercase tracking-wider mb-0.5">Processing</p>
                        <p className="text-xs font-bold text-zinc-800">{fmtShare(snap.processes_share)}</p>
                      </div>
                      <div>
                        <p className="text-[8px] font-semibold text-zinc-400 uppercase tracking-wider mb-0.5">Binding</p>
                        <p className="text-xs font-bold text-zinc-800 capitalize">{snap.binding ?? '—'}</p>
                      </div>
                    </div>
                    {snap.available ? (
                      <button
                        onClick={() => setLightbox(snap)}
                        className="block w-full focus:outline-none focus:ring-2 focus:ring-indigo-400 cursor-zoom-in"
                      >
                        {/* eslint-disable-next-line @next/next/no-img-element */}
                        <img src={snap.image_url} alt={snap.title} className="w-full h-auto block" loading="lazy" />
                      </button>
                    ) : (
                      <div className="px-4 py-10 bg-amber-50 border-amber-200 text-amber-800 text-xs text-center">
                        Snapshot not built. Use the slider above to render on demand.
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
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
              <p className="text-sm font-semibold text-zinc-800">
                {'title' in lightbox ? lightbox.title : `${lightbox.commodity} ${lightbox.year}`}
              </p>
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
            <img src={lightbox.image_url} alt="" className="block max-w-full h-auto" />
          </div>
        </div>
      )}
    </div>
  );
}

// ─── Year slider panel ───────────────────────────────────────────────────────

function YearSliderPanel({
  year, yearMin, yearMax, onYearChange,
  snapshot, loading, error, onExpand,
}: {
  year: number;
  yearMin: number;
  yearMax: number;
  onYearChange: (y: number) => void;
  snapshot: YearSnapshotResponse | null;
  loading: boolean;
  error: string | null;
  onExpand: (s: YearSnapshotResponse) => void;
}) {
  const ctrl = snapshot?.control;
  return (
    <div className="bg-white border border-zinc-200 rounded-xl p-4 flex flex-col">
      <div className="flex items-center justify-between mb-3">
        <p className="text-[10px] font-semibold text-zinc-400 uppercase tracking-wider">
          On-demand KG snapshot
        </p>
        <p className="text-[10px] text-zinc-400 font-mono">
          {snapshot?.cached === false ? 'just rendered' : snapshot?.cached ? 'cached' : ''}
        </p>
      </div>

      {/* Year slider */}
      <div className="mb-3">
        <div className="flex items-baseline justify-between mb-1">
          <span className="text-3xl font-bold text-zinc-900 font-mono">{year}</span>
          <span className="text-[10px] text-zinc-400 font-mono">{yearMin}–{yearMax}</span>
        </div>
        <input
          type="range"
          min={yearMin} max={yearMax} step={1}
          value={year}
          onChange={(e) => onYearChange(Number(e.target.value))}
          className="w-full accent-indigo-600"
        />
      </div>

      {/* Stats row */}
      {ctrl && (
        <div className="grid grid-cols-3 gap-2 mb-3 text-center">
          <div className="bg-zinc-50 rounded-lg p-2">
            <p className="text-[8px] font-semibold text-zinc-400 uppercase tracking-wider mb-0.5">Mining</p>
            <p className="text-sm font-bold text-zinc-800">{fmtShare(ctrl.produces_share)}</p>
          </div>
          <div className="bg-zinc-50 rounded-lg p-2">
            <p className="text-[8px] font-semibold text-zinc-400 uppercase tracking-wider mb-0.5">Processing</p>
            <p className="text-sm font-bold text-zinc-800">{fmtShare(ctrl.processes_share)}</p>
          </div>
          <div className="bg-zinc-50 rounded-lg p-2">
            <p className="text-[8px] font-semibold text-zinc-400 uppercase tracking-wider mb-0.5">Binding</p>
            <p className="text-sm font-bold text-zinc-800 capitalize">{ctrl.binding ?? '—'}</p>
          </div>
        </div>
      )}

      {/* Image */}
      <div className="flex-1 min-h-[200px] bg-zinc-50 rounded-lg flex items-center justify-center overflow-hidden">
        {loading && (
          <div className="flex flex-col items-center text-zinc-500 text-xs">
            <div className="h-6 w-6 border-2 border-indigo-200 border-t-indigo-600 rounded-full animate-spin mb-2" />
            Rendering KG for {year}…
          </div>
        )}
        {!loading && error && (
          <p className="text-xs text-red-600 px-4 text-center">{error}</p>
        )}
        {!loading && !error && snapshot && (
          <button
            onClick={() => onExpand(snapshot)}
            className="block w-full focus:outline-none cursor-zoom-in"
          >
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img src={snapshot.image_url} alt="" className="w-full h-auto block" />
          </button>
        )}
      </div>
    </div>
  );
}
