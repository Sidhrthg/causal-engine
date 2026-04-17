'use client';

import { useMemo } from 'react';

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
  const pad = range * 0.12;
  return [min - pad, max + pad];
}

function tickValues(lo: number, hi: number, n = 5): number[] {
  const step = (hi - lo) / (n - 1);
  return Array.from({ length: n }, (_, i) => lo + step * i);
}

export default function LineChart({ years, series, title, unit = '', height = 180 }: Props) {
  const W = 480;
  const H = height;
  const innerW = W - PAD.left - PAD.right;
  const innerH = H - PAD.top - PAD.bottom;

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

  return (
    <div>
      <p className="text-xs font-semibold text-zinc-600 mb-1">{title}</p>
      <svg
        viewBox={`0 0 ${W} ${H}`}
        className="w-full"
        style={{ height }}
        aria-label={title}
      >
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
                    r={2.5}
                    fill={s.color}
                  />
                ))}
              </g>
            );
          })}
        </g>
      </svg>

      {/* Legend */}
      <div className="flex gap-4 mt-1 pl-1">
        {series.map((s) => (
          <div key={s.label} className="flex items-center gap-1.5">
            <div className="h-2 w-6 rounded-full" style={{ backgroundColor: s.color }} />
            <span className="text-[10px] text-zinc-500">{s.label}</span>
          </div>
        ))}
      </div>
    </div>
  );
}
