'use client';

import { useState } from 'react';
import { batchEnrichKG, enrichKG } from '@/lib/api';

interface RunRecord {
  ts: string;
  mode: 'single' | 'batch';
  query: string;
  result: string;
  error?: string;
}

const SUGGESTED_QUERIES = [
  'graphite anode processing China export controls',
  'lithium brine triangle Atacama Salar',
  'cobalt artisanal mining Democratic Republic of Congo',
  'nickel HPAL processing Indonesia ore ban',
  'rare earths separation refining Lynas MP Materials',
  'copper smelting concentrate Chile Peru',
];

export default function KGEnrichPage() {
  const [query, setQuery] = useState('');
  const [topK, setTopK] = useState(5);
  const [loading, setLoading] = useState<'single' | 'batch' | null>(null);
  const [history, setHistory] = useState<RunRecord[]>([]);

  const runSingle = async () => {
    const q = query.trim();
    if (!q || loading) return;
    setLoading('single');
    const ts = new Date().toLocaleTimeString();
    try {
      const { result } = await enrichKG({ query: q, top_k: topK });
      setHistory((h) => [{ ts, mode: 'single', query: q, result }, ...h]);
    } catch (e) {
      setHistory((h) => [{
        ts, mode: 'single', query: q, result: '',
        error: e instanceof Error ? e.message : 'Enrich failed',
      }, ...h]);
    } finally {
      setLoading(null);
    }
  };

  const runBatch = async () => {
    if (loading) return;
    setLoading('batch');
    const ts = new Date().toLocaleTimeString();
    try {
      const { result } = await batchEnrichKG({ top_k: 3 });
      setHistory((h) => [{ ts, mode: 'batch', query: 'all 6 minerals', result }, ...h]);
    } catch (e) {
      setHistory((h) => [{
        ts, mode: 'batch', query: 'all 6 minerals', result: '',
        error: e instanceof Error ? e.message : 'Batch enrich failed',
      }, ...h]);
    } finally {
      setLoading(null);
    }
  };

  return (
    <div className="flex flex-col h-screen bg-zinc-50">
      {/* Header */}
      <div className="border-b border-zinc-200 bg-white px-6 py-3 shrink-0">
        <div>
          <p className="text-[10px] font-semibold text-indigo-600 uppercase tracking-widest mb-0.5">
            HippoRAG → Claude → Knowledge Graph
          </p>
          <h1 className="text-lg font-bold text-zinc-900">
            KG Enrichment
            <span className="ml-2 text-sm font-normal text-zinc-400">
              Grow the causal KG from the document corpus
            </span>
          </h1>
        </div>
      </div>

      <div className="flex flex-1 overflow-hidden">
        {/* Form */}
        <div className="w-96 border-r border-zinc-200 bg-white p-5 overflow-y-auto shrink-0">
          <div className="flex flex-col gap-5">
            {/* Pipeline diagram */}
            <div className="bg-indigo-50 border border-indigo-100 rounded-lg p-3">
              <p className="text-[10px] font-semibold text-indigo-700 uppercase tracking-wider mb-2">
                Pipeline
              </p>
              <ol className="text-[11px] text-zinc-700 space-y-1 leading-relaxed">
                <li>1. HippoRAG retrieves top-K chunks for your query</li>
                <li>2. Claude extracts (subject, relation, object) triples</li>
                <li>3. Triples merged into the live enriched KG</li>
                <li>4. Saved to <code className="font-mono text-[10px] bg-white px-1 rounded">data/canonical/enriched_kg.json</code> (persisted on Fly volume)</li>
              </ol>
            </div>

            {/* Single query */}
            <div>
              <label className="text-[10px] font-semibold text-zinc-400 uppercase tracking-wider mb-1.5 block">
                Enrichment Query
              </label>
              <textarea
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="e.g. graphite anode processing China export controls"
                rows={3}
                className="w-full text-sm border border-zinc-200 rounded-lg px-3 py-2 bg-white focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500 resize-none"
              />
            </div>

            <div>
              <label className="text-[10px] font-semibold text-zinc-400 uppercase tracking-wider mb-1.5 flex items-center justify-between">
                <span>Top-K Chunks</span>
                <span className="text-zinc-700 font-mono">{topK}</span>
              </label>
              <input
                type="range" min={1} max={20} step={1}
                value={topK}
                onChange={(e) => setTopK(Number(e.target.value))}
                className="w-full accent-indigo-600"
              />
              <p className="text-[10px] text-zinc-400 mt-1">
                More chunks = more candidate triples but slower (~5s per chunk).
              </p>
            </div>

            <button
              onClick={runSingle}
              disabled={!query.trim() || loading !== null}
              className="w-full bg-gradient-to-r from-indigo-600 to-violet-600 text-white text-sm font-semibold py-2.5 rounded-lg hover:opacity-95 disabled:opacity-40 disabled:cursor-not-allowed transition-opacity"
            >
              {loading === 'single' ? 'Enriching…' : 'Enrich KG'}
            </button>

            {/* Suggestions */}
            <div>
              <p className="text-[10px] font-semibold text-zinc-400 uppercase tracking-wider mb-2">
                Suggested Queries
              </p>
              <div className="flex flex-col gap-1">
                {SUGGESTED_QUERIES.map((q) => (
                  <button
                    key={q}
                    onClick={() => setQuery(q)}
                    disabled={loading !== null}
                    className="text-left text-[11px] text-zinc-600 hover:text-indigo-700 hover:bg-indigo-50 px-2 py-1.5 rounded transition-colors disabled:opacity-50"
                  >
                    {q}
                  </button>
                ))}
              </div>
            </div>

            <div className="border-t border-zinc-100 pt-4">
              <p className="text-[10px] font-semibold text-zinc-400 uppercase tracking-wider mb-2">
                Batch Mode
              </p>
              <p className="text-[11px] text-zinc-500 mb-2 leading-relaxed">
                Run a canonical enrichment query for each of the 6 critical
                minerals (graphite, lithium, cobalt, nickel, copper, soybeans).
                Takes ~1–3 min depending on retrieval depth.
              </p>
              <button
                onClick={runBatch}
                disabled={loading !== null}
                className="w-full text-xs px-3 py-2 border border-zinc-200 bg-white text-zinc-700 rounded-lg hover:bg-zinc-50 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
              >
                {loading === 'batch' ? 'Running batch enrichment…' : 'Enrich for all 6 minerals'}
              </button>
            </div>
          </div>
        </div>

        {/* Results panel */}
        <div className="flex-1 overflow-y-auto p-6">
          {loading && (
            <div className="bg-white border border-indigo-200 rounded-lg p-4 mb-4 flex items-center gap-3">
              <div className="h-4 w-4 border-2 border-indigo-200 border-t-indigo-600 rounded-full animate-spin" />
              <p className="text-sm text-zinc-700">
                {loading === 'batch'
                  ? 'Running batch enrichment for all 6 minerals — this typically takes 1–3 minutes…'
                  : 'Retrieving chunks and extracting triples — this takes 10–30 seconds…'}
              </p>
            </div>
          )}

          {history.length === 0 && !loading && (
            <div className="flex flex-col items-center justify-center h-full text-center text-zinc-400">
              <svg className="w-16 h-16 mb-4 text-zinc-300" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10" />
              </svg>
              <p className="text-sm font-medium text-zinc-500">No enrichments yet</p>
              <p className="text-xs mt-1 max-w-sm">
                Pick a suggested query or write your own, then click{' '}
                <span className="font-semibold">Enrich KG</span>.
              </p>
            </div>
          )}

          <div className="flex flex-col gap-4">
            {history.map((r, idx) => (
              <RunCard key={idx} record={r} />
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}

function RunCard({ record }: { record: RunRecord }) {
  const isError = Boolean(record.error);
  return (
    <div className={`bg-white border rounded-lg p-4 ${isError ? 'border-red-200' : 'border-zinc-200'}`}>
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-2">
          <span className={`text-[10px] font-semibold uppercase tracking-wider px-2 py-0.5 rounded-full ${
            record.mode === 'batch'
              ? 'bg-violet-50 text-violet-700'
              : 'bg-indigo-50 text-indigo-700'
          }`}>
            {record.mode}
          </span>
          <span className="text-[11px] text-zinc-400">{record.ts}</span>
        </div>
      </div>

      <p className="text-xs text-zinc-500 mb-2">
        Query: <span className="font-mono text-zinc-700">{record.query}</span>
      </p>

      {isError ? (
        <p className="text-sm text-red-700">{record.error}</p>
      ) : (
        <Markdown text={record.result} />
      )}
    </div>
  );
}

// Minimal markdown renderer for the bold/code/list output app.py returns.
function Markdown({ text }: { text: string }) {
  const lines = text.split('\n');
  return (
    <div className="text-sm text-zinc-700 space-y-1 leading-relaxed">
      {lines.map((line, i) => {
        if (!line.trim()) return <div key={i} className="h-2" />;
        const isBullet = line.startsWith('- ');
        const content = isBullet ? line.slice(2) : line;
        return (
          <div key={i} className={isBullet ? 'pl-4 relative' : ''}>
            {isBullet && (
              <span className="absolute left-0 top-1.5 h-1 w-1 rounded-full bg-zinc-400" />
            )}
            <InlineMarkdown text={content} />
          </div>
        );
      })}
    </div>
  );
}

function InlineMarkdown({ text }: { text: string }) {
  // Handle **bold** and `code` inline.
  const parts: React.ReactNode[] = [];
  const re = /(\*\*[^*]+\*\*|`[^`]+`)/g;
  let last = 0;
  let key = 0;
  let m: RegExpExecArray | null;
  while ((m = re.exec(text)) !== null) {
    if (m.index > last) parts.push(<span key={key++}>{text.slice(last, m.index)}</span>);
    const tok = m[1];
    if (tok.startsWith('**')) {
      parts.push(
        <strong key={key++} className="font-semibold text-zinc-900">{tok.slice(2, -2)}</strong>,
      );
    } else {
      parts.push(
        <code key={key++} className="font-mono text-[12px] bg-zinc-100 text-zinc-800 px-1.5 py-0.5 rounded">
          {tok.slice(1, -1)}
        </code>,
      );
    }
    last = m.index + tok.length;
  }
  if (last < text.length) parts.push(<span key={key++}>{text.slice(last)}</span>);
  return <>{parts}</>;
}
