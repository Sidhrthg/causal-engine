'use client';

/**
 * "How to read this output" panel. Sits next to MathPanel on the L1/L2/L3
 * pages and spells out what each field in the result actually means, plus a
 * sample reading so the user knows how to interpret typical values.
 */

interface OutputField {
  name: string;       // monospace, e.g. "ate_mean.P" or "Peak vs baseline"
  meaning: string;    // what it represents
  read?: string;      // optional example reading
}

interface OutputGuideProps {
  rung: 'L1' | 'L2' | 'L3';
  title?: string;
  intro?: string;
  fields: OutputField[];
  takeaway?: string;
}

const TINT: Record<OutputGuideProps['rung'], string> = {
  L1: 'bg-zinc-50 dark:bg-zinc-900/80 border-sky-200 dark:border-sky-700/40',
  L2: 'bg-zinc-50 dark:bg-zinc-900/80 border-indigo-200 dark:border-indigo-700/40',
  L3: 'bg-zinc-50 dark:bg-zinc-900/80 border-violet-200 dark:border-violet-700/40',
};

const BADGE: Record<OutputGuideProps['rung'], string> = {
  L1: 'bg-sky-100 text-sky-800 dark:bg-sky-500/20 dark:text-sky-200',
  L2: 'bg-indigo-100 text-indigo-800 dark:bg-indigo-500/20 dark:text-indigo-200',
  L3: 'bg-violet-100 text-violet-800 dark:bg-violet-500/20 dark:text-violet-200',
};

export default function OutputGuide({
  rung, title = 'How to read this output', intro, fields, takeaway,
}: OutputGuideProps) {
  return (
    <div className={`border rounded-lg p-4 ${TINT[rung]}`}>
      <div className="flex items-center gap-2 mb-2">
        <span className={`text-[10px] font-bold rounded px-1.5 py-0.5 ${BADGE[rung]}`}>{rung}</span>
        <p className="text-xs font-semibold uppercase tracking-wider text-zinc-700 dark:text-zinc-200">
          {title}
        </p>
      </div>

      {intro && (
        <p className="text-xs text-zinc-600 dark:text-zinc-400 leading-relaxed mb-3">{intro}</p>
      )}

      <div className="flex flex-col gap-2.5">
        {fields.map((f, i) => (
          <div key={i} className="text-xs leading-relaxed">
            <span className="font-mono font-semibold text-zinc-800 dark:text-zinc-100">{f.name}</span>
            <span className="text-zinc-600 dark:text-zinc-400"> — {f.meaning}</span>
            {f.read && (
              <p className="text-[11px] text-zinc-500 dark:text-zinc-500 mt-0.5 ml-3 italic">
                e.g. {f.read}
              </p>
            )}
          </div>
        ))}
      </div>

      {takeaway && (
        <p className="text-[11px] leading-relaxed mt-3 pt-3 border-t border-zinc-200 dark:border-zinc-700/50 text-zinc-600 dark:text-zinc-400">
          <span className="font-semibold text-zinc-700 dark:text-zinc-200">Key takeaway:</span>{' '}
          {takeaway}
        </p>
      )}
    </div>
  );
}
