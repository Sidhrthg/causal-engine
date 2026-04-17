interface Props {
  rate: number;
  ci: [number, number];
  nominal_t: number;
  rerouted_t: number;
  hubs: string[];
  notes: string[];
}

export default function CircumventionCard({
  rate,
  ci,
  nominal_t,
  rerouted_t,
  hubs,
  notes,
}: Props) {
  const pct = (rate * 100).toFixed(1);
  const ciLo = (ci[0] * 100).toFixed(1);
  const ciHi = (ci[1] * 100).toFixed(1);

  const severity =
    rate >= 0.2 ? 'text-red-600' : rate >= 0.05 ? 'text-amber-600' : 'text-emerald-600';

  return (
    <div className="rounded-lg border border-zinc-200 p-5 bg-white">
      <h3 className="text-[11px] font-semibold text-zinc-400 uppercase tracking-wider mb-4">
        Circumvention Estimate
      </h3>

      {/* Point estimate */}
      <div className="mb-4">
        <div className="flex items-baseline gap-2 mb-1">
          <span className={`text-3xl font-bold tabular-nums ${severity}`}>{pct}%</span>
          <span className="text-xs text-zinc-400">detected</span>
        </div>
        <p className="text-[11px] text-zinc-400 mb-2">
          95% CI [{ciLo}%, {ciHi}%]
        </p>

        {/* CI bar */}
        <div className="relative h-2 bg-zinc-100 rounded-full overflow-visible">
          <div
            className="absolute top-0 h-full bg-indigo-100 rounded-full"
            style={{
              left: `${ci[0] * 100}%`,
              width: `${Math.max((ci[1] - ci[0]) * 100, 0.5)}%`,
            }}
          />
          <div
            className="absolute w-0.5 h-4 bg-indigo-600 -top-1 rounded-full"
            style={{ left: `${Math.min(rate * 100, 99)}%` }}
          />
        </div>
        <div className="flex justify-between text-[10px] text-zinc-300 mt-1">
          <span>0%</span>
          <span>50%</span>
          <span>100%</span>
        </div>
      </div>

      {/* Volume breakdown */}
      <dl className="grid grid-cols-2 gap-3 text-sm pt-3 border-t border-zinc-100">
        <div>
          <dt className="text-[10px] text-zinc-400 uppercase tracking-wide">Nominal restricted</dt>
          <dd className="font-mono font-semibold text-zinc-800 mt-0.5">
            {nominal_t.toLocaleString('en-US', { maximumFractionDigits: 0 })}t
          </dd>
        </div>
        <div>
          <dt className="text-[10px] text-zinc-400 uppercase tracking-wide">Detected rerouted</dt>
          <dd className="font-mono font-semibold text-zinc-800 mt-0.5">
            {rerouted_t.toLocaleString('en-US', { maximumFractionDigits: 0 })}t
          </dd>
        </div>
      </dl>

      {/* Significant hubs */}
      {hubs.length > 0 && (
        <div className="mt-4 pt-3 border-t border-zinc-100">
          <p className="text-[10px] text-zinc-400 uppercase tracking-wide mb-1.5">
            Rerouting hubs (p&lt;0.10)
          </p>
          <div className="flex flex-wrap gap-1">
            {hubs.map((h) => (
              <span
                key={h}
                className="text-[11px] font-medium bg-amber-50 text-amber-700 border border-amber-200 px-1.5 py-0.5 rounded"
              >
                {h}
              </span>
            ))}
          </div>
        </div>
      )}

      {/* Notes */}
      {notes.length > 0 && (
        <div className="mt-3 pt-3 border-t border-zinc-100">
          {notes.map((n, i) => (
            <p key={i} className="text-[11px] text-zinc-500 leading-relaxed">
              {n}
            </p>
          ))}
        </div>
      )}
    </div>
  );
}
