'use client';

/**
 * Compact mathematical-explanation panel used on the L1/L2/L3 pages.
 * Shows: ladder rung name, formal Pearl definition, the specific
 * structural equations this codebase actually runs, and the source file
 * the equations live in.
 */

interface Equation {
  label?: string;
  code: string;       // pre-formatted math, displayed in monospace
}

interface MathPanelProps {
  rung: 'L1' | 'L2' | 'L3';
  title: string;
  formal: string;
  equations: Equation[];
  caveat?: string;
  source?: string;    // e.g. "src/minerals/pearl_layers.py — do_compare()"
}

const RUNG_COLOR: Record<MathPanelProps['rung'], string> = {
  L1: 'bg-sky-50 dark:bg-sky-500/10 border-sky-200 dark:border-sky-700/50 text-sky-900 dark:text-sky-200',
  L2: 'bg-indigo-50 dark:bg-indigo-500/10 border-indigo-200 dark:border-indigo-700/50 text-indigo-900 dark:text-indigo-200',
  L3: 'bg-violet-50 dark:bg-violet-500/10 border-violet-200 dark:border-violet-700/50 text-violet-900 dark:text-violet-200',
};

const BADGE_COLOR: Record<MathPanelProps['rung'], string> = {
  L1: 'bg-sky-200 text-sky-900 dark:bg-sky-500/30 dark:text-sky-200',
  L2: 'bg-indigo-200 text-indigo-900 dark:bg-indigo-500/30 dark:text-indigo-200',
  L3: 'bg-violet-200 text-violet-900 dark:bg-violet-500/30 dark:text-violet-200',
};

export default function MathPanel({ rung, title, formal, equations, caveat, source }: MathPanelProps) {
  return (
    <div className={`border rounded-lg p-4 ${RUNG_COLOR[rung]}`}>
      <div className="flex items-center gap-2 mb-2">
        <span className={`text-[10px] font-bold rounded px-1.5 py-0.5 ${BADGE_COLOR[rung]}`}>{rung}</span>
        <p className="text-xs font-semibold uppercase tracking-wider">{title}</p>
      </div>

      <p className="text-xs leading-relaxed mb-3">
        <span className="font-semibold">Formal:</span>{' '}
        <span className="font-mono">{formal}</span>
      </p>

      <div className="flex flex-col gap-2">
        {equations.map((eq, i) => (
          <div key={i} className="bg-white/60 dark:bg-zinc-900/40 rounded p-2.5 border border-current/10">
            {eq.label && (
              <p className="text-[10px] font-semibold uppercase tracking-wider mb-1 opacity-70">
                {eq.label}
              </p>
            )}
            <pre className="text-[11px] font-mono whitespace-pre-wrap leading-snug overflow-x-auto">
              {eq.code}
            </pre>
          </div>
        ))}
      </div>

      {caveat && (
        <p className="text-[11px] leading-relaxed mt-3 italic opacity-90">
          <span className="font-semibold not-italic">Caveat:</span> {caveat}
        </p>
      )}

      {source && (
        <p className="text-[10px] mt-2 opacity-75">
          Source: <span className="font-mono">{source}</span>
        </p>
      )}
    </div>
  );
}
