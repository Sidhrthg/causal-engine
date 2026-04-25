'use client';

import { useState, useEffect, useCallback } from 'react';
import LineChart from '@/components/LineChart';
import { getScenarios, runCounterfactual } from '@/lib/api';
import type { ScenarioMeta, CounterfactualResponse, TrajectoryRow } from '@/lib/types';

// ─── Helpers ─────────────────────────────────────────────────────────────────

type CfType = 'substitution' | 'fringe' | 'trajectory';

function ateColor(val: number, outcome: string): string {
  const positive_good = ['Q_total', 'tight']; // higher = worse for tight, hmm
  const negative_good = ['shortage', 'P'];     // lower = better
  if (Math.abs(val) < 0.001) return 'text-zinc-500';
  if (negative_good.includes(outcome)) return val < 0 ? 'text-emerald-600' : 'text-red-600';
  if (outcome === 'Q_total' || outcome === 'Q_sub') return val > 0 ? 'text-emerald-600' : 'text-red-600';
  return val !== 0 ? 'text-indigo-600' : 'text-zinc-500';
}

function ateArrow(val: number): string {
  if (Math.abs(val) < 0.001) return '→';
  return val > 0 ? '↑' : '↓';
}

function fmtAte(val: number): string {
  if (Math.abs(val) >= 1000) return `${val > 0 ? '+' : ''}${(val / 1000).toFixed(1)}k`;
  return `${val > 0 ? '+' : ''}${val.toFixed(2)}`;
}

function extractSeries(rows: TrajectoryRow[], key: keyof TrajectoryRow): number[] {
  return rows.map((r) => r[key] as number);
}

const OUTCOME_LABELS: Record<string, string> = {
  P: 'Price (normalised)',
  Q_total: 'Total supply',
  shortage: 'Shortage',
  tight: 'Tightness',
  Q_sub: 'Substitution supply',
  Q_fringe: 'Fringe supply',
  K: 'Capacity',
  D: 'Demand',
};

const OUTCOMES_TO_SHOW: (keyof TrajectoryRow)[] = ['P', 'Q_total', 'shortage', 'Q_sub'];

const CF_TYPE_INFO: Record<CfType, { label: string; question: string; color: string }> = {
  substitution: {
    label: 'Substitution',
    question: 'What if non-dominant suppliers had different capacity to absorb restricted volume?',
    color: 'text-indigo-600 bg-indigo-50 border-indigo-200',
  },
  fringe: {
    label: 'Fringe Supply',
    question: 'What if high-cost fringe producers had entered the market at a different threshold?',
    color: 'text-violet-600 bg-violet-50 border-violet-200',
  },
  trajectory: {
    label: 'Shock Override',
    question: 'What if a specific shock (e.g. export restriction) had not happened, or been different?',
    color: 'text-amber-600 bg-amber-50 border-amber-200',
  },
};

// ─── Sub-forms ────────────────────────────────────────────────────────────────

function SubstitutionForm({
  elasticity, setElasticity, cap, setCap,
}: {
  elasticity: number; setElasticity: (v: number) => void;
  cap: number; setCap: (v: number) => void;
}) {
  return (
    <div className="grid grid-cols-2 gap-4">
      <div>
        <label className="block text-xs font-semibold text-zinc-500 mb-1">
          Counterfactual elasticity
          <span className="font-normal text-zinc-400 ml-1">(factual in scenario)</span>
        </label>
        <input
          type="number" step="0.05" min={0} max={5}
          value={elasticity}
          onChange={(e) => setElasticity(Number(e.target.value))}
          className="w-full text-sm border border-zinc-200 rounded-lg px-3 py-2 focus:outline-none focus:ring-2 focus:ring-indigo-500"
        />
        <p className="text-[10px] text-zinc-400 mt-1">
          0.0 = no substitution · 0.8 = strong substitution · controls how fast non-dominant
          suppliers fill the gap when export_restriction &gt; 0
        </p>
      </div>
      <div>
        <label className="block text-xs font-semibold text-zinc-500 mb-1">
          Substitution cap
          <span className="font-normal text-zinc-400 ml-1">(0–1)</span>
        </label>
        <input
          type="number" step="0.05" min={0} max={1}
          value={cap}
          onChange={(e) => setCap(Number(e.target.value))}
          className="w-full text-sm border border-zinc-200 rounded-lg px-3 py-2 focus:outline-none focus:ring-2 focus:ring-indigo-500"
        />
        <p className="text-[10px] text-zinc-400 mt-1">
          Maximum fraction of restricted volume that can ever be substituted
        </p>
      </div>
    </div>
  );
}

function FringeForm({
  share, setShare, entry, setEntry,
}: {
  share: number; setShare: (v: number) => void;
  entry: number; setEntry: (v: number) => void;
}) {
  return (
    <div className="grid grid-cols-2 gap-4">
      <div>
        <label className="block text-xs font-semibold text-zinc-500 mb-1">
          Fringe capacity share
          <span className="font-normal text-zinc-400 ml-1">(fraction of K0)</span>
        </label>
        <input
          type="number" step="0.05" min={0} max={2}
          value={share}
          onChange={(e) => setShare(Number(e.target.value))}
          className="w-full text-sm border border-zinc-200 rounded-lg px-3 py-2 focus:outline-none focus:ring-2 focus:ring-indigo-500"
        />
        <p className="text-[10px] text-zinc-400 mt-1">
          0.0 = no fringe · 0.3 = fringe can supply up to 30% of dominant capacity
        </p>
      </div>
      <div>
        <label className="block text-xs font-semibold text-zinc-500 mb-1">
          Entry price threshold
          <span className="font-normal text-zinc-400 ml-1">(× P_ref)</span>
        </label>
        <input
          type="number" step="0.1" min={0.5} max={5}
          value={entry}
          onChange={(e) => setEntry(Number(e.target.value))}
          className="w-full text-sm border border-zinc-200 rounded-lg px-3 py-2 focus:outline-none focus:ring-2 focus:ring-indigo-500"
        />
        <p className="text-[10px] text-zinc-400 mt-1">
          Price at which fringe producers start entering (e.g. 1.5 = 50% above P_ref)
        </p>
      </div>
    </div>
  );
}

function TrajectoryForm({
  overrides, setOverrides,
}: {
  overrides: string; setOverrides: (v: string) => void;
}) {
  return (
    <div>
      <label className="block text-xs font-semibold text-zinc-500 mb-1">
        Shock overrides by year (JSON)
      </label>
      <textarea
        rows={5}
        value={overrides}
        onChange={(e) => setOverrides(e.target.value)}
        className="w-full text-xs font-mono border border-zinc-200 rounded-lg px-3 py-2 focus:outline-none focus:ring-2 focus:ring-indigo-500"
        spellCheck={false}
      />
      <p className="text-[10px] text-zinc-400 mt-1">
        Keys: <code className="bg-zinc-100 px-0.5 rounded">export_restriction</code>{' '}
        <code className="bg-zinc-100 px-0.5 rounded">demand_surge</code>{' '}
        <code className="bg-zinc-100 px-0.5 rounded">capex_shock</code>{' '}
        <code className="bg-zinc-100 px-0.5 rounded">policy_supply_mult</code>{' '}
        <code className="bg-zinc-100 px-0.5 rounded">stockpile_release</code>
      </p>
    </div>
  );
}

// ─── Main page ────────────────────────────────────────────────────────────────

export default function CounterfactualPage() {
  const [scenarios, setScenarios] = useState<ScenarioMeta[]>([]);
  const [scenariosLoading, setScenariosLoading] = useState(true);

  const [scenario, setScenario] = useState('graphite_2023_china_export_controls_substitution');
  const [cfType, setCfType] = useState<CfType>('substitution');
  const [useCalibrated, setUseCalibrated] = useState(true);

  // substitution params
  const [elasticity, setElasticity] = useState(0.0);
  const [cap, setCap] = useState(0.6);

  // fringe params
  const [share, setShare] = useState(0.2);
  const [entry, setEntry] = useState(1.5);

  // trajectory params
  const [overrides, setOverrides] = useState(
    JSON.stringify({ '2023': { export_restriction: 0.0 }, '2024': { export_restriction: 0.0 } }, null, 2)
  );

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<CounterfactualResponse | null>(null);

  // Load scenarios
  useEffect(() => {
    getScenarios()
      .then((r) => {
        setScenarios(r.scenarios);
        if (r.scenarios.length > 0 && !r.scenarios.find((s) => s.name === scenario)) {
          setScenario(r.scenarios[0].name);
        }
      })
      .catch(() => {})
      .finally(() => setScenariosLoading(false));
  }, []);

  const selectedMeta = scenarios.find((s) => s.name === scenario);

  const handleSubmit = useCallback(async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      let params: Parameters<typeof runCounterfactual>[0] = {
        scenario,
        cf_type: cfType,
        use_calibrated: useCalibrated,
      };

      if (cfType === 'substitution') {
        params = { ...params, cf_elasticity: elasticity, cf_cap: cap };
      } else if (cfType === 'fringe') {
        params = { ...params, cf_capacity_share: share, cf_entry_price: entry };
      } else {
        let parsed: Record<string, Record<string, number>>;
        try {
          parsed = JSON.parse(overrides);
        } catch {
          throw new Error('Shock overrides JSON is invalid — check the format');
        }
        params = { ...params, shock_overrides: parsed };
      }

      const data = await runCounterfactual(params);
      setResult(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Counterfactual failed');
    } finally {
      setLoading(false);
    }
  }, [scenario, cfType, useCalibrated, elasticity, cap, share, entry, overrides]);

  const years = result?.factual.map((r) => r.year) ?? [];

  return (
    <div className="min-h-screen bg-zinc-50">
      <div className="max-w-5xl mx-auto px-8 py-8">

        {/* Header */}
        <div className="mb-7">
          <span className="text-[11px] font-semibold text-purple-600 uppercase tracking-widest">
            Pearl Layer 3 — Imagining
          </span>
          <h1 className="text-2xl font-bold text-zinc-900 mt-1">Counterfactual Analysis</h1>
          <p className="text-sm text-zinc-500 mt-1 max-w-2xl">
            Abduction-Action-Prediction: fix the exogenous noise sequence from the factual run,
            change one structural mechanism, replay to see what <em>would have been</em>.
          </p>
        </div>

        {/* Config card */}
        <form onSubmit={handleSubmit} className="bg-white rounded-xl border border-zinc-200 p-6 mb-6 shadow-sm space-y-5">

          {/* Scenario selector */}
          <div>
            <label className="block text-xs font-semibold text-zinc-500 mb-1.5">Scenario</label>
            {scenariosLoading ? (
              <div className="h-9 bg-zinc-100 rounded-lg animate-pulse" />
            ) : (
              <select
                value={scenario}
                onChange={(e) => { setScenario(e.target.value); setResult(null); }}
                className="w-full text-sm border border-zinc-200 rounded-lg px-3 py-2 bg-white focus:outline-none focus:ring-2 focus:ring-purple-500"
              >
                {scenarios.map((s) => (
                  <option key={s.name} value={s.name}>
                    {s.name}
                    {s.calibrated ? ' ✓ calibrated' : ' (hardcoded params)'}
                    {s.has_shocks ? ` · ${s.start_year}–${s.end_year}` : ''}
                  </option>
                ))}
              </select>
            )}
            {selectedMeta && (
              <div className="mt-1.5 flex items-center gap-3 text-[11px] text-zinc-400">
                <span>{selectedMeta.commodity} · {selectedMeta.start_year}–{selectedMeta.end_year}</span>
                {selectedMeta.calibrated ? (
                  <span className="text-emerald-600 font-medium">✓ empirically fitted params</span>
                ) : (
                  <span className="text-amber-600">
                    ⚠ hardcoded params — run{' '}
                    <code className="bg-amber-50 px-1 rounded">python scripts/fit_and_write_scenarios.py</code>
                    {' '}to calibrate
                  </span>
                )}
                <label className="flex items-center gap-1 ml-auto cursor-pointer">
                  <input
                    type="checkbox"
                    checked={useCalibrated}
                    onChange={(e) => setUseCalibrated(e.target.checked)}
                    className="accent-purple-600"
                  />
                  <span>Prefer calibrated</span>
                </label>
              </div>
            )}
          </div>

          {/* CF type */}
          <div>
            <label className="block text-xs font-semibold text-zinc-500 mb-2">Counterfactual type</label>
            <div className="grid grid-cols-3 gap-2">
              {(Object.entries(CF_TYPE_INFO) as [CfType, typeof CF_TYPE_INFO[CfType]][]).map(([type, info]) => (
                <button
                  key={type}
                  type="button"
                  onClick={() => { setCfType(type); setResult(null); }}
                  className={`text-left p-3 rounded-lg border text-xs transition-all ${
                    cfType === type
                      ? info.color
                      : 'bg-white border-zinc-200 text-zinc-600 hover:border-zinc-300'
                  }`}
                >
                  <p className="font-semibold mb-0.5">{info.label}</p>
                  <p className="opacity-75 leading-relaxed">{info.question}</p>
                </button>
              ))}
            </div>
          </div>

          {/* CF parameters (conditional) */}
          <div className="p-4 bg-zinc-50 rounded-lg border border-zinc-100">
            <p className="text-[11px] font-semibold text-zinc-400 uppercase tracking-wider mb-3">
              Counterfactual parameters
            </p>
            {cfType === 'substitution' && (
              <SubstitutionForm elasticity={elasticity} setElasticity={setElasticity} cap={cap} setCap={setCap} />
            )}
            {cfType === 'fringe' && (
              <FringeForm share={share} setShare={setShare} entry={entry} setEntry={setEntry} />
            )}
            {cfType === 'trajectory' && (
              <TrajectoryForm overrides={overrides} setOverrides={setOverrides} />
            )}
          </div>

          <button
            type="submit"
            disabled={loading || scenariosLoading}
            className="flex items-center gap-2 px-5 py-2.5 bg-purple-600 text-white text-sm font-semibold rounded-lg hover:bg-purple-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors shadow-sm"
          >
            {loading ? (
              <>
                <div className="h-4 w-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                Running abduction-action-prediction…
              </>
            ) : (
              <>
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                </svg>
                Run Counterfactual (L3)
              </>
            )}
          </button>
        </form>

        {/* Error */}
        {error && (
          <div className="rounded-xl bg-red-50 border border-red-200 px-5 py-4 mb-6">
            <p className="text-sm font-semibold text-red-700 mb-1">Counterfactual failed</p>
            <p className="text-sm text-red-600">{error}</p>
          </div>
        )}

        {/* Results */}
        {result && (
          <div className="space-y-6">
            {/* Description */}
            <div className="rounded-xl border border-purple-100 bg-purple-50 px-5 py-4">
              <p className="text-[11px] font-semibold text-purple-500 uppercase tracking-wider mb-1">
                L3 query
              </p>
              <p className="text-sm text-purple-800 font-mono leading-relaxed">{result.description}</p>
            </div>

            {/* ATE summary */}
            <div>
              <h2 className="text-xs font-semibold text-zinc-400 uppercase tracking-wider mb-3">
                Average Treatment Effects — E[Y_cf] − E[Y_factual]
              </h2>
              <div className="grid grid-cols-4 gap-3">
                {(['P', 'Q_total', 'shortage', 'Q_sub'] as const).map((key) => {
                  const val = result.ate[key] ?? 0;
                  const col = ateColor(val, key);
                  return (
                    <div key={key} className="bg-white rounded-xl border border-zinc-200 p-4">
                      <p className="text-[10px] text-zinc-400 uppercase tracking-wide mb-1">
                        {OUTCOME_LABELS[key]}
                      </p>
                      <p className={`text-xl font-bold tabular-nums ${col}`}>
                        {ateArrow(val)} {fmtAte(val)}
                      </p>
                      <p className="text-[10px] text-zinc-400 mt-0.5">cf − factual</p>
                    </div>
                  );
                })}
              </div>
            </div>

            {/* Parameters comparison */}
            <div className="bg-white rounded-xl border border-zinc-200 p-4">
              <h3 className="text-[11px] font-semibold text-zinc-400 uppercase tracking-wider mb-3">
                Structural change
              </h3>
              <div className="flex gap-8 text-sm">
                <div>
                  <p className="text-[10px] text-zinc-400 mb-2">Factual (scenario)</p>
                  {Object.entries(result.factual_params).map(([k, v]) => (
                    <div key={k} className="flex gap-3 font-mono text-xs text-zinc-600">
                      <span className="text-zinc-400 w-32">{k}</span>
                      <span>{typeof v === 'number' ? v.toFixed(4) : String(v)}</span>
                    </div>
                  ))}
                </div>
                <div className="border-l border-zinc-100 pl-8">
                  <p className="text-[10px] text-zinc-400 mb-2">Counterfactual (do-operator)</p>
                  {Object.entries(result.cf_params).map(([k, v]) => (
                    <div key={k} className="flex gap-3 font-mono text-xs text-purple-700 font-semibold">
                      <span className="text-purple-400 w-32">{k}</span>
                      <span>{typeof v === 'number' ? v.toFixed(4) : JSON.stringify(v)}</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>

            {/* Charts */}
            <div>
              <h2 className="text-xs font-semibold text-zinc-400 uppercase tracking-wider mb-3">
                Trajectories — factual vs counterfactual
              </h2>
              <div className="grid grid-cols-2 gap-6">
                {OUTCOMES_TO_SHOW.map((key) => (
                  <div key={key} className="bg-white rounded-xl border border-zinc-200 p-4">
                    <LineChart
                      years={years}
                      title={OUTCOME_LABELS[key as string] ?? String(key)}
                      series={[
                        {
                          label: 'Factual',
                          data: extractSeries(result.factual, key),
                          color: '#6366f1',
                        },
                        {
                          label: 'Counterfactual',
                          data: extractSeries(result.counterfactual, key),
                          color: '#f59e0b',
                        },
                      ]}
                    />
                  </div>
                ))}
              </div>
            </div>

            {/* Methodology note */}
            <div className="rounded-xl border border-zinc-200 bg-zinc-50 px-5 py-4">
              <p className="text-[11px] font-semibold text-zinc-400 uppercase tracking-wider mb-2">
                Methodology
              </p>
              <p className="text-xs text-zinc-500 leading-relaxed">
                <strong>Abduction:</strong> factual scenario <code className="bg-zinc-100 px-1 rounded font-mono">{result.scenario}</code> run — noise sequence ε_t captured.{' '}
                <strong>Action:</strong> structural equation modified (do-operator) per parameters above.{' '}
                <strong>Prediction:</strong> simulation replayed with same ε_t — identical exogenous shocks, different mechanism.{' '}
                ATEs are per-year mean differences across the simulation window. The shared noise sequence
                is the twin-network coupling that distinguishes L3 from L2 (which uses a fresh simulation).
              </p>
            </div>
          </div>
        )}

      </div>
    </div>
  );
}
