'use client';

import { useMemo, useState } from 'react';

interface Series {
  label: string;
  data: number[];
  color: string;
}

interface Props {
  years: number[];
  series: Series[];
  title: string;
  unit?: string;
  height?: number;
}

const PAD = { top: 16, right: 16, bottom: 28, left: 52 };

function niceRange(values: number[]): [number, number] {
  const min = Math.min(...values);
  const max = Math.max(...values);
  const range = max - min || 1;
  const pad = range * 0.15;
  return [min - pad, max + pad];
}

function tickValues(lo: number, hi: number, n = 5): number[] {
  const step = (hi - lo) / (n - 1);
  return Array.from({ length: n }, (_, i) => lo + step * i);
}

function hexToRgba(hex: string, alpha: number): string {
  const r = parseInt(hex.slice(1, 3), 16);
  const g = parseInt(hex.slice(3, 5), 16);
  const b = parseInt(hex.slice(5, 7), 16);
  return `rgba(${r},${g},${b},${alpha})`;
}

export default function LineChart({ years, series, title, unit = '', height = 180 }: Props) {
  const W = 480;
  const H = height;
  const innerW = W - PAD.left - PAD.right;
  const innerH = H - PAD.top - PAD.bottom;

  const [tooltip, setTooltip] = useState<{ x: number; y: number; year: number; values: { label: string; value: number; color: string }[] } | null>(null);

  const allValues = series.flatMap((s) => s.data);
  const [yLo, yHi] = useMemo(() => niceRange(allValues), [allValues]);
  const yTicks = useMemo(() => tickValues(yLo, yHi, 5), [yLo, yHi]);

  const xScale = (i: number) => (i / Math.max(years.length - 1, 1)) * innerW;
  const yScale = (v: number) => innerH - ((v - yLo) / (yHi - yLo)) * innerH;

  const fmt = (v: number) =>
    Math.abs(v) >= 1000
      ? `${(v / 1000).toFixed(1)}k`
      : v % 1 === 0
      ? v.toFixed(0)
      : v.toFixed(2);

  const handleMouseMove = (e: React.MouseEvent<SVGSVGElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const svgX = ((e.clientX - rect.left) / rect.width) * W;
    const innerX = svgX - PAD.left;
    const fraction = innerX / innerW;
    const idx = Math.round(fraction * (years.length - 1));
    if (idx < 0 || idx >= years.length) { setTooltip(null); return; }
    setTooltip({
      x: PAD.left + xScale(idx),
      y: PAD.top + Math.min(...series.map((s) => yScale(s.data[idx] ?? 0))),
      year: years[idx],
      values: series.map((s) => ({ label: s.label, value: s.data[idx] ?? 0, color: s.color })),
    });
  };

  return (
    <div className="relative">
      <p className="text-xs font-semibold text-zinc-600 dark:text-zinc-400 mb-2">{title}</p>
      <svg
        viewBox={`0 0 ${W} ${H}`}
        className="w-full"
        style={{ height }}
        aria-label={title}
        onMouseMove={handleMouseMove}
        onMouseLeave={() => setTooltip(null)}
      >
        <defs>
          {series.map((s) => (
            <linearGradient key={s.label} id={`grad-${s.label.replace(/\s/g, '')}`} x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor={s.color} stopOpacity="0.18" />
              <stop offset="100%" stopColor={s.color} stopOpacity="0.01" />
            </linearGradient>
          ))}
        </defs>

        <g transform={`translate(${PAD.left},${PAD.top})`}>
          {/* Grid lines */}
          {yTicks.map((v, i) => (
            <g key={i}>
              <line
                x1={0} y1={yScale(v)}
                x2={innerW} y2={yScale(v)}
                stroke="#f1f5f9" strokeWidth={1}
              />
              <text
                x={-6} y={yScale(v)}
                textAnchor="end"
                dominantBaseline="middle"
                fontSize={9}
                fill="#94a3b8"
              >
                {fmt(v)}{unit}
              </text>
            </g>
          ))}

          {/* X axis labels */}
          {years.map((yr, i) => {
            if (years.length > 10 && i % 2 !== 0) return null;
            return (
              <text
                key={yr}
                x={xScale(i)} y={innerH + 16}
                textAnchor="middle"
                fontSize={9}
                fill="#94a3b8"
              >
                {yr}
              </text>
            );
          })}

          {/* Zero line if in range */}
          {yLo < 0 && yHi > 0 && (
            <line
              x1={0} y1={yScale(0)}
              x2={innerW} y2={yScale(0)}
              stroke="#cbd5e1" strokeWidth={1} strokeDasharray="3,3"
            />
          )}

          {/* Area fills */}
          {series.map((s) => {
            const pts = s.data.map((v, i) => `${xScale(i)},${yScale(v)}`).join(' ');
            const area = `${xScale(0)},${innerH} ${pts} ${xScale(s.data.length - 1)},${innerH}`;
            return (
              <polygon
                key={`area-${s.label}`}
                points={area}
                fill={`url(#grad-${s.label.replace(/\s/g, '')})`}
              />
            );
          })}

          {/* Series lines */}
          {series.map((s) => {
            const pts = s.data
              .map((v, i) => `${xScale(i)},${yScale(v)}`)
              .join(' ');
            return (
              <g key={s.label}>
                <polyline
                  points={pts}
                  fill="none"
                  stroke={s.color}
                  strokeWidth={2}
                  strokeLinejoin="round"
                  strokeLinecap="round"
                />
                {/* Dots */}
                {s.data.map((v, i) => (
                  <circle
                    key={i}
                    cx={xScale(i)} cy={yScale(v)}
                    r={3}
                    fill="white"
                    stroke={s.color}
                    strokeWidth={1.5}
                  />
                ))}
              </g>
            );
          })}

          {/* Tooltip vertical line */}
          {tooltip && (
            <line
              x1={tooltip.x - PAD.left} y1={0}
              x2={tooltip.x - PAD.left} y2={innerH}
              stroke="#94a3b8" strokeWidth={1} strokeDasharray="3,3"
            />
          )}
        </g>

        {/* Tooltip box */}
        {tooltip && (() => {
          const bx = tooltip.x + 8;
          const tooltipW = 110;
          const flipped = bx + tooltipW > W - 8;
          const tx = flipped ? tooltip.x - tooltipW - 8 : bx;
          return (
            <g>
              <rect
                x={tx} y={tooltip.y - 4}
                width={tooltipW}
                height={18 + tooltip.values.length * 16}
                rx={4} ry={4}
                fill="white"
                stroke="#e4e4e7"
                strokeWidth={1}
                style={{ filter: 'drop-shadow(0 1px 3px rgba(0,0,0,0.08))' }}
              />
              <text x={tx + 8} y={tooltip.y + 9} fontSize={9} fill="#71717a" fontWeight="600">
                {tooltip.year}
              </text>
              {tooltip.values.map((v, i) => (
                <g key={v.label}>
                  <rect x={tx + 8} y={tooltip.y + 18 + i * 16 - 5} width={6} height={6} rx={1} fill={v.color} />
                  <text x={tx + 18} y={tooltip.y + 18 + i * 16 + 0.5} fontSize={9} fill="#3f3f46">
                    {v.label}: {fmt(v.value)}{unit}
                  </text>
                </g>
              ))}
            </g>
          );
        })()}
      </svg>

      {/* Legend */}
      <div className="flex gap-4 mt-1 pl-1">
        {series.map((s) => (
          <div key={s.label} className="flex items-center gap-1.5">
            <div className="h-0.5 w-5 rounded-full" style={{ backgroundColor: s.color }} />
            <span className="text-[10px] text-zinc-500">{s.label}</span>
          </div>
        ))}
      </div>
    </div>
  );
}
