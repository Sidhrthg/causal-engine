'use client';

import { useEffect, useState } from 'react';

/**
 * Collapsible "How to use" panel. Dismiss state persists in localStorage
 * (per-page, keyed by `id`) so users see it once and can hide it after.
 */
export default function HowToUse({
  id,
  title = 'How to use this page',
  steps,
  tip,
  defaultOpen = true,
}: {
  id: string;
  title?: string;
  steps: React.ReactNode[];
  tip?: React.ReactNode;
  defaultOpen?: boolean;
}) {
  const storageKey = `howto:${id}`;
  const [open, setOpen] = useState<boolean | null>(null);

  // Hydrate on mount to avoid SSR mismatch
  useEffect(() => {
    try {
      const v = window.localStorage.getItem(storageKey);
      setOpen(v === null ? defaultOpen : v === '1');
    } catch {
      setOpen(defaultOpen);
    }
  }, [storageKey, defaultOpen]);

  const toggle = (next: boolean) => {
    setOpen(next);
    try { window.localStorage.setItem(storageKey, next ? '1' : '0'); } catch {}
  };

  if (open === null) return null; // pre-hydration; avoid flicker

  return (
    <div className="bg-indigo-50/60 dark:bg-indigo-500/10 border border-indigo-100 dark:border-indigo-500/20 rounded-lg mb-4">
      <button
        onClick={() => toggle(!open)}
        className="w-full flex items-center justify-between px-4 py-2.5 text-left"
      >
        <div className="flex items-center gap-2">
          <span className="h-5 w-5 rounded-full bg-indigo-100 dark:bg-indigo-500/20 flex items-center justify-center text-indigo-700 dark:text-indigo-300 text-[11px] font-bold">i</span>
          <span className="text-xs font-semibold text-indigo-900 dark:text-indigo-200">{title}</span>
        </div>
        <svg
          className={`w-3.5 h-3.5 text-indigo-600 dark:text-indigo-400 transition-transform ${open ? 'rotate-90' : ''}`}
          fill="none" stroke="currentColor" viewBox="0 0 24 24"
        >
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
        </svg>
      </button>
      {open && (
        <div className="px-4 pb-3 pt-0">
          <ol className="text-xs text-zinc-700 dark:text-zinc-300 leading-relaxed space-y-1.5 list-decimal pl-5">
            {steps.map((s, i) => <li key={i}>{s}</li>)}
          </ol>
          {tip && (
            <p className="mt-2.5 text-[11px] text-indigo-700 dark:text-indigo-300 bg-white/60 dark:bg-zinc-900/40 rounded px-2.5 py-1.5 border border-indigo-100 dark:border-indigo-500/20">
              <span className="font-semibold">Tip:</span> {tip}
            </p>
          )}
        </div>
      )}
    </div>
  );
}
