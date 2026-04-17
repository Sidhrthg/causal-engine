import type { RouteResult } from '@/lib/types';

interface Props {
  routes: RouteResult[];
}

export default function RouteTable({ routes }: Props) {
  const maxBottleneck = Math.max(...routes.map((r) => r.bottleneck_t), 1);

  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b border-zinc-200 text-[11px] text-zinc-400 font-medium uppercase tracking-wide">
            <th className="py-2 px-3 text-left w-8">#</th>
            <th className="py-2 px-3 text-left">Route</th>
            <th className="py-2 px-3 text-right">Volume (t)</th>
            <th className="py-2 px-3 text-right">% Source</th>
            <th className="py-2 px-3 text-left w-28">Flow</th>
            <th className="py-2 px-3 text-left w-24">Type</th>
          </tr>
        </thead>
        <tbody>
          {routes.map((r) => (
            <tr
              key={r.rank}
              className="border-b border-zinc-100 hover:bg-zinc-50 transition-colors"
            >
              <td className="py-2.5 px-3 text-zinc-400 font-mono text-xs">{r.rank}</td>
              <td className="py-2.5 px-3 max-w-xs">
                <span className="font-mono text-xs text-zinc-700 break-words">
                  {r.path.join(' → ')}
                </span>
                {r.non_producer_intermediaries.length > 0 && (
                  <div className="mt-0.5 flex flex-wrap gap-1">
                    {r.non_producer_intermediaries.map((h) => (
                      <span
                        key={h}
                        className="text-[10px] text-amber-600 bg-amber-50 px-1 rounded"
                      >
                        {h}
                      </span>
                    ))}
                  </div>
                )}
              </td>
              <td className="py-2.5 px-3 text-right font-mono tabular-nums text-zinc-800 text-xs">
                {r.bottleneck_t.toLocaleString('en-US', { maximumFractionDigits: 0 })}
              </td>
              <td className="py-2.5 px-3 text-right font-mono tabular-nums text-zinc-500 text-xs">
                {(r.pct_of_source * 100).toFixed(1)}%
              </td>
              <td className="py-2.5 px-3">
                <div className="h-1.5 bg-zinc-100 rounded-full overflow-hidden">
                  <div
                    className={`h-full rounded-full transition-all ${
                      r.is_circumvention ? 'bg-amber-400' : 'bg-indigo-400'
                    }`}
                    style={{
                      width: `${Math.max((r.bottleneck_t / maxBottleneck) * 100, 2)}%`,
                    }}
                  />
                </div>
              </td>
              <td className="py-2.5 px-3">
                {r.is_circumvention ? (
                  <span className="text-[10px] font-semibold text-amber-700 bg-amber-50 border border-amber-200 px-1.5 py-0.5 rounded">
                    TRANSSHIP
                  </span>
                ) : (
                  <span className="text-[10px] font-semibold text-emerald-700 bg-emerald-50 border border-emerald-200 px-1.5 py-0.5 rounded">
                    DIRECT
                  </span>
                )}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
