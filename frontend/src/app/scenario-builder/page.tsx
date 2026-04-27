'use client';

import { useEffect, useMemo, useRef, useState } from 'react';
import HowToUse from '@/components/HowToUse';
import {
  getKnowledgeGraph,
  getScenarioPresets,
  renderScenario,
} from '@/lib/api';
import type { ScenarioPreset, ScenarioResult } from '@/lib/types';

interface EntitySuggestion {
  id: string;
  entity_type: string;
}

const PROGRESS_HINTS = [
  'Generating retrieval query…',
  'Retrieving documents from HippoRAG…',
  'Extracting causal triples (Claude)…',
  'Resolving focal nodes against KG…',
  'Propagating shock through subgraph…',
  'Rendering PNG…',
];

type Mode = 'preset' | 'custom';

function Autocomplete({
  label,
  value,
  onChange,
  suggestions,
  placeholder,
}: {
  label: string;
  value: string;
  onChange: (v: string) => void;
  suggestions: string[];
  placeholder: string;
}) {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const handler = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false);
    };
    document.addEventListener('mousedown', handler);
    return () => document.removeEventListener('mousedown', handler);
  }, []);

  const filtered = useMemo(() => {
    const q = value.trim().toLowerCase();
    const pool = suggestions.slice().sort();
    if (!q) return pool.slice(0, 12);
    return pool.filter((s) => s.toLowerCase().includes(q)).slice(0, 12);
  }, [value, suggestions]);

  return (
    <div ref={ref} className="relative">
      <label className="text-[10px] font-semibold text-zinc-400 uppercase tracking-wider mb-1.5 block">
        {label}
      </label>
      <input
        type="text"
        value={value}
        onChange={(e) => { onChange(e.target.value); setOpen(true); }}
        onFocus={() => setOpen(true)}
        placeholder={placeholder}
        className="w-full text-sm border border-zinc-200 dark:border-zinc-800 rounded-lg px-3 py-2 bg-white dark:bg-zinc-900 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500"
      />
      {open && filtered.length > 0 && (
        <div className="absolute z-50 mt-1 w-full bg-white dark:bg-zinc-900 border border-zinc-200 dark:border-zinc-800 rounded-lg shadow-lg max-h-48 overflow-y-auto">
          {filtered.map((s) => (
            <button
              key={s}
              type="button"
              onClick={() => { onChange(s); setOpen(false); }}
              className="w-full text-left px-3 py-1.5 text-sm text-zinc-700 dark:text-zinc-300 hover:bg-indigo-50 hover:text-indigo-700"
            >
              {s.replace(/_/g, ' ')}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}

export default function ScenarioBuilderPage() {
  const [mode, setMode] = useState<Mode>('preset');

  // Preset state
  const [presets, setPresets] = useState<ScenarioPreset[]>([]);
  const [presetsLoading, setPresetsLoading] = useState(true);
  const [selectedPreset, setSelectedPreset] = useState<ScenarioPreset | null>(null);

  // Custom form state
  const [year, setYear] = useState(2024);
  const [shockOrigin, setShockOrigin] = useState('');
  const [commodity, setCommodity] = useState('');
  const [title, setTitle] = useState('');

  const [entities, setEntities] = useState<EntitySuggestion[]>([]);

  const [loading, setLoading] = useState(false);
  const [hintIndex, setHintIndex] = useState(0);
  const [result, setResult] = useState<ScenarioResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Load presets on mount
  useEffect(() => {
    let cancelled = false;
    getScenarioPresets()
      .then((data) => {
        if (cancelled) return;
        const all = [...data.validation, ...data.predictive];
        setPresets(all);
        const firstAvailable = all.find((p) => p.available);
        if (firstAvailable) setSelectedPreset(firstAvailable);
      })
      .catch(() => {
        // Non-fatal — just leaves presets empty
      })
      .finally(() => {
        if (!cancelled) setPresetsLoading(false);
      });
    return () => { cancelled = true; };
  }, []);

  // Load KG entities for autocomplete (custom mode)
  useEffect(() => {
    let cancelled = false;
    getKnowledgeGraph()
      .then((data) => {
        if (cancelled) return;
        setEntities(data.entities.map((e) => ({ id: e.id, entity_type: e.entity_type })));
      })
      .catch(() => {});
    return () => { cancelled = true; };
  }, []);

  // Cycle progress hints while rendering custom scenario
  useEffect(() => {
    if (!loading) { setHintIndex(0); return; }
    const id = setInterval(() => {
      setHintIndex((i) => (i + 1) % PROGRESS_HINTS.length);
    }, 8000);
    return () => clearInterval(id);
  }, [loading]);

  const countrySuggestions = useMemo(
    () => entities.filter((e) => e.entity_type === 'country' || e.entity_type === 'region').map((e) => e.id),
    [entities],
  );
  const commoditySuggestions = useMemo(
    () => entities.filter((e) => e.entity_type === 'commodity').map((e) => e.id),
    [entities],
  );

  const validationPresets = presets.filter((p) => p.kind === 'validation');
  const predictivePresets = presets.filter((p) => p.kind === 'predictive');

  const canSubmit = !loading && shockOrigin.trim() && commodity.trim() && title.trim();

  const usePresetAsTemplate = (p: ScenarioPreset) => {
    setMode('custom');
    setYear(p.year);
    setShockOrigin(p.shock_origin);
    setCommodity(p.commodity);
    setTitle(p.title);
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    setResult(null);
    setLoading(true);
    try {
      const r = await renderScenario({
        year,
        shock_origin: shockOrigin.trim(),
        commodity: commodity.trim(),
        title: title.trim(),
      });
      setResult(r);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Render failed');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex flex-col h-screen bg-zinc-50">
      {/* Header */}
      <div className="border-b border-zinc-200 dark:border-zinc-800 bg-white dark:bg-zinc-900 px-6 py-3 shrink-0">
        <div className="flex items-center justify-between">
          <div>
            <p className="text-[10px] font-semibold text-indigo-600 uppercase tracking-widest mb-0.5">
              KG Scenario Renderer
            </p>
            <h1 className="text-lg font-bold text-zinc-900">
              Scenario Builder
              <span className="ml-2 text-sm font-normal text-zinc-400">
                Browse pre-rendered or generate custom causal subgraphs
              </span>
            </h1>
          </div>

          {/* Mode switch */}
          <div className="flex items-center gap-1 bg-zinc-100 rounded-lg p-1">
            <button
              onClick={() => setMode('preset')}
              className={`text-xs px-3 py-1.5 rounded-md font-medium transition-colors ${
                mode === 'preset' ? 'bg-white text-indigo-700 shadow-sm' : 'text-zinc-500 hover:text-zinc-700'
              }`}
            >
              Pre-rendered
            </button>
            <button
              onClick={() => setMode('custom')}
              className={`text-xs px-3 py-1.5 rounded-md font-medium transition-colors ${
                mode === 'custom' ? 'bg-white text-indigo-700 shadow-sm' : 'text-zinc-500 hover:text-zinc-700'
              }`}
            >
              Custom
            </button>
          </div>
        </div>
      </div>

      <div className="flex flex-1 overflow-hidden">
        {/* Sidebar */}
        <div className="w-80 border-r border-zinc-200 dark:border-zinc-800 bg-white dark:bg-zinc-900 p-5 overflow-y-auto shrink-0">
          <HowToUse
            id="scenario-builder"
            steps={[
              <><strong>Pre-rendered</strong>: instant view of any of the 10 historical episodes (graphite 2008, lithium 2016, cobalt 2016, etc.). Pick from the list — the PNG loads immediately.</>,
              <><strong>Custom</strong>: build a new scenario. Set year, shock origin (country), commodity, and a free-text title. Or click <strong>Quick Fill</strong> to seed the form from a known scenario.</>,
              <>Click <strong>Generate Knowledge Graph</strong>. The pipeline runs HippoRAG retrieval → Claude triple extraction → focal-node subgraph render. Takes 30–90s.</>,
              <>Use <strong>Use as Custom template →</strong> on a preset to copy its fields into the Custom form so you can tweak them.</>,
            ]}
            tip="First custom render of the day takes ~30s extra (HippoRAG warmup); subsequent calls reuse the warmed pipeline."
          />
          {mode === 'preset' ? (
            <PresetSidebar
              loading={presetsLoading}
              validation={validationPresets}
              predictive={predictivePresets}
              selected={selectedPreset}
              onSelect={setSelectedPreset}
              onUseAsTemplate={usePresetAsTemplate}
            />
          ) : (
            <CustomForm
              year={year} setYear={setYear}
              shockOrigin={shockOrigin} setShockOrigin={setShockOrigin}
              commodity={commodity} setCommodity={setCommodity}
              title={title} setTitle={setTitle}
              countrySuggestions={countrySuggestions}
              commoditySuggestions={commoditySuggestions}
              presets={presets}
              onApplyPreset={(p) => {
                setYear(p.year);
                setShockOrigin(p.shock_origin);
                setCommodity(p.commodity);
                setTitle(p.title);
              }}
              loading={loading}
              canSubmit={Boolean(canSubmit)}
              onSubmit={handleSubmit}
            />
          )}
        </div>

        {/* Result panel */}
        <div className="flex-1 relative overflow-auto p-6">
          {mode === 'preset' && selectedPreset && (
            <PresetView preset={selectedPreset} />
          )}
          {mode === 'preset' && !selectedPreset && !presetsLoading && (
            <Empty
              title="No scenarios available"
              hint="Generate one with Custom mode."
            />
          )}

          {mode === 'custom' && error && (
            <div className="bg-red-50 border border-red-200 text-red-800 text-sm rounded-lg p-3 mb-4">
              <p className="font-semibold mb-1">Render failed</p>
              <p className="text-red-700">{error}</p>
            </div>
          )}

          {mode === 'custom' && loading && (
            <div className="flex flex-col items-center justify-center h-full text-center">
              <div className="h-10 w-10 border-2 border-indigo-200 border-t-indigo-600 rounded-full animate-spin mb-4" />
              <p className="text-sm font-medium text-zinc-700 dark:text-zinc-300 mb-1">{PROGRESS_HINTS[hintIndex]}</p>
              <p className="text-[11px] text-zinc-400">This typically takes 30–90 seconds.</p>
            </div>
          )}

          {mode === 'custom' && !loading && !result && !error && (
            <Empty
              title="No scenario yet"
              hint="Fill out the form on the left and click Generate to render a custom causal subgraph."
            />
          )}

          {mode === 'custom' && !loading && result && (
            <CustomResultView result={result} year={year} />
          )}
        </div>
      </div>
    </div>
  );
}

// ─── Sub-components ──────────────────────────────────────────────────────────

function PresetSidebar({
  loading, validation, predictive, selected, onSelect, onUseAsTemplate,
}: {
  loading: boolean;
  validation: ScenarioPreset[];
  predictive: ScenarioPreset[];
  selected: ScenarioPreset | null;
  onSelect: (p: ScenarioPreset) => void;
  onUseAsTemplate: (p: ScenarioPreset) => void;
}) {
  return (
    <div className="flex flex-col gap-5">
      <div>
        <p className="text-[10px] font-semibold text-zinc-400 uppercase tracking-wider mb-2">
          Validation Episodes ({validation.length})
        </p>
        <p className="text-[10px] text-zinc-400 mb-3 leading-relaxed">
          Real-world historical shocks. Click to view the pre-rendered KG.
        </p>
        <PresetList items={validation} selected={selected} onSelect={onSelect} />
      </div>

      {predictive.length > 0 && (
        <div>
          <p className="text-[10px] font-semibold text-zinc-400 uppercase tracking-wider mb-2">
            Predictive Scenarios ({predictive.length})
          </p>
          <PresetList items={predictive} selected={selected} onSelect={onSelect} />
        </div>
      )}

      {selected && (
        <button
          onClick={() => onUseAsTemplate(selected)}
          className="w-full text-xs px-3 py-2 border border-indigo-200 bg-indigo-50 text-indigo-700 rounded-lg hover:bg-indigo-100"
        >
          Use as Custom template →
        </button>
      )}

      {loading && <p className="text-xs text-zinc-400">Loading scenarios…</p>}
    </div>
  );
}

function PresetList({
  items, selected, onSelect,
}: {
  items: ScenarioPreset[];
  selected: ScenarioPreset | null;
  onSelect: (p: ScenarioPreset) => void;
}) {
  return (
    <div className="flex flex-col gap-1.5">
      {items.map((p) => {
        const active = selected?.scenario_id === p.scenario_id;
        return (
          <button
            key={p.scenario_id}
            onClick={() => onSelect(p)}
            className={`text-left px-3 py-2 rounded-lg border transition-colors ${
              active
                ? 'border-indigo-300 bg-indigo-50'
                : 'border-zinc-200 bg-white hover:bg-zinc-50'
            } ${!p.available ? 'opacity-60' : ''}`}
            title={!p.available ? 'Not yet rendered — image may 404' : undefined}
          >
            <p className={`text-xs font-semibold ${active ? 'text-indigo-700' : 'text-zinc-800'}`}>
              {p.scenario_id.replace(/^pred_/, '').replace(/_/g, ' ')}
            </p>
            <p className="text-[10px] text-zinc-500 mt-0.5">
              {p.year} · {p.shock_origin} · {p.commodity}
              {!p.available && <span className="text-amber-600 ml-1">· not built</span>}
            </p>
          </button>
        );
      })}
    </div>
  );
}

function PresetView({ preset }: { preset: ScenarioPreset }) {
  const [imgError, setImgError] = useState(false);
  return (
    <div>
      <div className="flex items-start justify-between mb-3">
        <div>
          <h2 className="text-base font-bold text-zinc-900">{preset.title}</h2>
          <p className="text-xs text-zinc-500 mt-1">
            {preset.year} · shock origin: <span className="font-mono">{preset.shock_origin}</span>
            {' · commodity: '}
            <span className="font-mono">{preset.commodity}</span>
          </p>
        </div>
        {preset.available && (
          <a
            href={preset.image_url}
            download={`${preset.scenario_id}.png`}
            className="text-xs px-3 py-1.5 border border-indigo-200 bg-indigo-50 text-indigo-700 rounded-lg hover:bg-indigo-100"
          >
            Download PNG
          </a>
        )}
      </div>

      {imgError || !preset.available ? (
        <div className="bg-amber-50 border border-amber-200 text-amber-800 text-sm rounded-lg p-4">
          This scenario hasn&apos;t been pre-rendered yet. Switch to{' '}
          <span className="font-semibold">Custom</span> mode and use this preset
          as a template to generate it (takes 30–90s).
        </div>
      ) : (
        <div className="border border-zinc-200 dark:border-zinc-800 rounded-lg overflow-hidden bg-white">
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            src={preset.image_url}
            alt={preset.title}
            className="w-full h-auto block"
            onError={() => setImgError(true)}
          />
        </div>
      )}
    </div>
  );
}

function CustomForm({
  year, setYear, shockOrigin, setShockOrigin, commodity, setCommodity,
  title, setTitle, countrySuggestions, commoditySuggestions,
  presets, onApplyPreset, loading, canSubmit, onSubmit,
}: {
  year: number; setYear: (v: number) => void;
  shockOrigin: string; setShockOrigin: (v: string) => void;
  commodity: string; setCommodity: (v: string) => void;
  title: string; setTitle: (v: string) => void;
  countrySuggestions: string[];
  commoditySuggestions: string[];
  presets: ScenarioPreset[];
  onApplyPreset: (p: ScenarioPreset) => void;
  loading: boolean;
  canSubmit: boolean;
  onSubmit: (e: React.FormEvent) => void;
}) {
  return (
    <form onSubmit={onSubmit} className="flex flex-col gap-4">
      {/* Quick-fill preset dropdown */}
      {presets.length > 0 && (
        <div>
          <label className="text-[10px] font-semibold text-zinc-400 uppercase tracking-wider mb-1.5 block">
            Quick Fill (optional)
          </label>
          <select
            defaultValue=""
            onChange={(e) => {
              const p = presets.find((x) => x.scenario_id === e.target.value);
              if (p) onApplyPreset(p);
              e.target.value = '';
            }}
            className="w-full text-sm border border-zinc-200 dark:border-zinc-800 rounded-lg px-3 py-2 bg-white dark:bg-zinc-900 focus:outline-none focus:ring-2 focus:ring-indigo-500"
          >
            <option value="">Choose a known scenario…</option>
            <optgroup label="Validation">
              {presets.filter((p) => p.kind === 'validation').map((p) => (
                <option key={p.scenario_id} value={p.scenario_id}>{p.title}</option>
              ))}
            </optgroup>
            <optgroup label="Predictive">
              {presets.filter((p) => p.kind === 'predictive').map((p) => (
                <option key={p.scenario_id} value={p.scenario_id}>{p.title}</option>
              ))}
            </optgroup>
          </select>
        </div>
      )}

      <div>
        <label className="text-[10px] font-semibold text-zinc-400 uppercase tracking-wider mb-1.5 block">Year</label>
        <input
          type="number" min={1990} max={2030}
          value={year}
          onChange={(e) => setYear(Number(e.target.value))}
          className="w-full text-sm border border-zinc-200 dark:border-zinc-800 rounded-lg px-3 py-2 bg-white dark:bg-zinc-900 focus:outline-none focus:ring-2 focus:ring-indigo-500"
        />
      </div>

      <Autocomplete
        label="Shock Origin"
        value={shockOrigin}
        onChange={setShockOrigin}
        suggestions={countrySuggestions}
        placeholder="e.g. china, drc, indonesia"
      />

      <Autocomplete
        label="Commodity"
        value={commodity}
        onChange={setCommodity}
        suggestions={commoditySuggestions}
        placeholder="e.g. graphite, lithium, cobalt"
      />

      <div>
        <label className="text-[10px] font-semibold text-zinc-400 uppercase tracking-wider mb-1.5 block">
          Scenario Title
        </label>
        <input
          type="text"
          value={title}
          onChange={(e) => setTitle(e.target.value)}
          placeholder="e.g. Copper 2024 — Chile Production Strike"
          className="w-full text-sm border border-zinc-200 dark:border-zinc-800 rounded-lg px-3 py-2 bg-white dark:bg-zinc-900 focus:outline-none focus:ring-2 focus:ring-indigo-500"
        />
      </div>

      <button
        type="submit"
        disabled={!canSubmit}
        className="mt-2 w-full bg-gradient-to-r from-indigo-600 to-violet-600 text-white text-sm font-semibold py-2.5 rounded-lg hover:opacity-95 disabled:opacity-40 disabled:cursor-not-allowed transition-opacity"
      >
        {loading ? 'Generating…' : 'Generate Knowledge Graph'}
      </button>

      <p className="text-[10px] text-zinc-400 leading-relaxed mt-1">
        First call warms up HippoRAG (~30s extra). Each render takes 30–90s
        because of Claude API calls for triple extraction.
      </p>
    </form>
  );
}

function CustomResultView({ result, year }: { result: ScenarioResult; year: number }) {
  return (
    <div>
      <div className="flex items-start justify-between mb-4">
        <div className="flex flex-wrap gap-2">
          <Stat label="Nodes" value={result.node_count} />
          <Stat label="Edges" value={result.edge_count} />
          <Stat label="Focal" value={result.focal_count} />
          <Stat label="Impact" value={result.impact_count} />
          {result.effective_share !== null && (
            <Stat
              label={`Eff. Control (${result.binding ?? 'unknown'})`}
              value={`${(result.effective_share * 100).toFixed(0)}%`}
            />
          )}
        </div>
        <a
          href={result.image_url}
          download={`${result.scenario_id}.png`}
          className="text-xs px-3 py-1.5 border border-indigo-200 bg-indigo-50 text-indigo-700 rounded-lg hover:bg-indigo-100 ml-2 shrink-0"
        >
          Download PNG
        </a>
      </div>

      {result.query && (
        <p className="text-[11px] text-zinc-500 mb-3 font-mono bg-zinc-100 px-3 py-1.5 rounded">
          HippoRAG query: &quot;{result.query}&quot;
        </p>
      )}

      {result.skipped ? (
        <div className="bg-amber-50 border border-amber-200 text-amber-800 text-sm rounded-lg p-4">
          No nodes found in KG snapshot at year {year}. Try a different year or
          commodity, or check that the enriched KG has data for this combination.
        </div>
      ) : (
        <div className="border border-zinc-200 dark:border-zinc-800 rounded-lg overflow-hidden bg-white">
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            src={result.image_url}
            alt={result.scenario_id}
            className="w-full h-auto block"
          />
        </div>
      )}
    </div>
  );
}

function Stat({ label, value }: { label: string; value: string | number }) {
  return (
    <div className="bg-white dark:bg-zinc-900 border border-zinc-200 dark:border-zinc-800 rounded-lg px-3 py-2">
      <p className="text-[9px] font-semibold text-zinc-400 uppercase tracking-wider">{label}</p>
      <p className="text-sm font-bold text-zinc-800">{value}</p>
    </div>
  );
}

function Empty({ title, hint }: { title: string; hint: string }) {
  return (
    <div className="flex flex-col items-center justify-center h-full text-center text-zinc-400">
      <svg className="w-16 h-16 mb-4 text-zinc-300" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M4 6a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2H6a2 2 0 01-2-2V6zM14 6a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2h-2a2 2 0 01-2-2V6zM4 16a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2H6a2 2 0 01-2-2v-2zM14 16a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2h-2a2 2 0 01-2-2v-2z" />
      </svg>
      <p className="text-sm font-medium text-zinc-500">{title}</p>
      <p className="text-xs mt-1 max-w-sm">{hint}</p>
    </div>
  );
}
