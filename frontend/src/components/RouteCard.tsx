import type { RouteResult } from '@/lib/types';

interface Props {
  route: RouteResult;
  maxBottleneck: number;
}

export default function RouteCard({ route, maxBottleneck }: Props) {
  const widthPct = Math.max((route.bottleneck_t / maxBottleneck) * 100, 3);
  const isCirc = route.is_circumvention;

  return (
    <div
      className={`rounded-xl border p-4 transition-all ${
        isCirc
          ? 'border-amber-200 bg-amber-50/40 hover:border-amber-300'
          : 'border-zinc-200 bg-white hover:border-zinc-300'
      }`}
    >
      {/* Top row */}
      <div className="flex items-start justify-between mb-3">
        <div className="flex items-center gap-2">
          <span className="text-[11px] font-mono text-zinc-300">#{route.rank}</span>
          {isCirc ? (
            <span className="inline-flex items-center gap-1 text-[11px] font-semibold text-amber-700 bg-amber-100 border border-amber-200 px-2 py-0.5 rounded-full">
              <svg className="w-3 h-3" fill="currentColor" viewBox="0 0 20 20">
                <path fillRule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
              </svg>
              Transshipment candidate
            </span>
          ) : (
            <span className="inline-flex items-center gap-1 text-[11px] font-semibold text-emerald-700 bg-emerald-50 border border-emerald-200 px-2 py-0.5 rounded-full">
              <svg className="w-3 h-3" fill="currentColor" viewBox="0 0 20 20">
                <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
              </svg>
              Direct route
            </span>
          )}
        </div>
        <div className="text-right shrink-0 ml-4">
          <p className="text-sm font-bold tabular-nums text-zinc-900">
            {route.bottleneck_t >= 1000
              ? `${(route.bottleneck_t / 1000).toFixed(1)}k t`
              : `${Math.round(route.bottleneck_t)} t`}
          </p>
          <p className="text-[11px] text-zinc-400">
            {(route.pct_of_source * 100).toFixed(1)}% of source exports
          </p>
        </div>
      </div>

      {/* Path pill chain */}
      <div className="flex items-center flex-wrap gap-y-2 gap-x-1 mb-3">
        {route.path.map((country, i) => {
          const isNonProducer = route.non_producer_intermediaries.includes(country);
          const isSource = i === 0;
          const isDest = i === route.path.length - 1;

          return (
            <div key={`${country}-${i}`} className="flex items-center gap-1">
              {i > 0 && (
                <svg
                  className={`w-4 h-4 ${isNonProducer ? 'text-amber-400' : 'text-zinc-300'}`}
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                </svg>
              )}
              <div className="flex flex-col items-center">
                <span
                  className={`text-xs font-medium px-2.5 py-1 rounded-lg border ${
                    isSource
                      ? 'bg-indigo-600 border-indigo-600 text-white'
                      : isDest
                      ? 'bg-zinc-800 border-zinc-800 text-white'
                      : isNonProducer
                      ? 'bg-amber-100 border-amber-300 text-amber-800'
                      : 'bg-blue-50 border-blue-200 text-blue-800'
                  }`}
                >
                  {country}
                </span>
                {isNonProducer && (
                  <span className="text-[9px] text-amber-600 mt-0.5 font-medium">non-producer</span>
                )}
              </div>
            </div>
          );
        })}
      </div>

      {/* Volume bar */}
      <div className="h-1 bg-zinc-100 rounded-full overflow-hidden">
        <div
          className={`h-full rounded-full transition-all duration-500 ${
            isCirc ? 'bg-amber-400' : 'bg-indigo-400'
          }`}
          style={{ width: `${widthPct}%` }}
        />
      </div>
    </div>
  );
}
