'use client';

import { useEffect, useState } from 'react';
import HowToUse from '@/components/HowToUse';
import LineChart from '@/components/LineChart';
import MathPanel from '@/components/MathPanel';
import OutputGuide from '@/components/OutputGuide';
import {
  getScenarios,
  runL2Intervention,
  type L2InterventionResponse,
  type L2Param,
} from '@/lib/api';
import type { ScenarioMeta } from '@/lib/types';

const PARAM_META: Record<L2Param, { label: string; desc: string; min: number; max: number; step: number; default: number; severs: string }> = {
  substitution_elasticity: { label: 'substitution_elasticity', desc: 'How fast non-dominant suppliers fill the gap when restricted', min: 0,    max: 2,   step: 0.05, default: 0.8,  severs: 'Price → SubstitutionSupply' },
  substitution_cap:        { label: 'substitution_cap',        desc: 'Max fraction of restricted supply that can be substituted',     min: 0,    max: 1,   step: 0.05, default: 0.6,  severs: 'Price → SubstitutionSupply (cap)' },
  fringe_capacity_share:   { label: 'fringe_capacity_share',   desc: 'High-cost entrant capacity as fraction of K0',                  min: 0,    max: 1,   step: 0.05, default: 0.4,  severs: 'Investment → FringeSupply' },
  fringe_entry_price:      { label: 'fringe_entry_price',      desc: 'Normalised price at which fringe producers first compete',      min: 1,    max: 5,   step: 0.1,  default: 1.5,  severs: 'Cost → FringeSupply' },
  eta_D:                   { label: 'η_D (demand elasticity)', desc: 'Price elasticity of demand (negative; more negative = more elastic)', min: -1.5, max: -0.01, step: 0.05, default: -0.5, severs: 'Preferences → Demand' },
  alpha_P:                 { label: 'α_P (price speed)',       desc: 'How fast price reacts to shortage',                            min: 0.1,  max: 3,   step: 0.1,  default: 1.5,  severs: 'MarketCond → Price' },
  tau_K:                   { label: 'τ_K (capacity time)',     desc: 'Years for capacity to adjust toward target (lower = faster)',  min: 0.5,  max: 20,  step: 0.5,  default: 5,    severs: 'Investment → Capacity' },
  eta_K:                   { label: 'η_K (capacity invest)',   desc: 'Investment response to price gap',                             min: 0,    max: 1,   step: 0.05, default: 0.4,  severs: 'Price → CapacityInvest' },
};

const OUTCOMES = ['P', 'Q_total', 'Q_sub', 'Q_fringe', 'shortage'];
const OUTCOME_LABEL: Record<string, string> = {
  P: 'Price (P)',
  Q_total: 'Total supply (Q_total)',
  Q_sub: 'Substitution supply (Q_sub)',
  Q_fringe: 'Fringe supply (Q_fringe)',
  shortage: 'Shortage',
};

export default function L2InterventionPage() {
  const [scenarios, setScenarios] = useState<ScenarioMeta[]>([]);
  const [scenario, setScenario] = useState('lithium_2022_ev_boom_with_fringe');
  const [enabled, setEnabled] = useState<Record<L2Param, boolean>>({
    substitution_elasticity: true,
    substitution_cap: false,
    fringe_capacity_share: false,
    fringe_entry_price: false,
    eta_D: false,
    alpha_P: false,
    tau_K: false,
    eta_K: false,
  });
  const [values, setValues] = useState<Record<L2Param, number>>(() =>
    Object.fromEntries(
      (Object.keys(PARAM_META) as L2Param[]).map((p) => [p, PARAM_META[p].default]),
    ) as Record<L2Param, number>,
  );
  const [outcomes, setOutcomes] = useState<string[]>(['P', 'Q_total', 'Q_sub', 'Q_fringe', 'shortage']);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<L2InterventionResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    getScenarios()
      .then((d) => {
        setScenarios(d.scenarios);
        if (d.scenarios.length && !d.scenarios.find((s) => s.name === scenario)) {
          setScenario(d.scenarios[0].name);
        }
      })
      .catch(() => {});
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const enabledParams = (Object.keys(enabled) as L2Param[]).filter((p) => enabled[p]);

  const handleRun = async (e: React.FormEvent) => {
    e.preventDefault();
    if (enabledParams.length === 0) {
      setError('Enable at least one parameter to intervene on.');
      return;
    }
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const overrides = Object.fromEntries(enabledParams.map((p) => [p, values[p]])) as Partial<Record<L2Param, number>>;
      const r = await runL2Intervention({
        scenario_name: scenario,
        parameter_overrides: overrides,
        outcomes,
      });
      setResult(r);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'L2 intervention failed');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex flex-col h-screen bg-zinc-50 dark:bg-zinc-950">
      {/* Header */}
      <div className="border-b border-zinc-200 dark:border-zinc-800 bg-white dark:bg-zinc-900 px-6 py-3 shrink-0">
        <div className="flex items-center gap-2 mb-0.5">
          <span className="text-[10px] font-bold text-indigo-700 bg-indigo-100 dark:bg-indigo-500/15 dark:text-indigo-300 rounded px-1.5 py-0.5">L2</span>
          <p className="text-[10px] font-semibold text-indigo-600 uppercase tracking-widest">Pearl Layer 2 — Intervention</p>
        </div>
        <h1 className="text-lg font-bold text-zinc-900 dark:text-zinc-100">
          Canonical do-calculus
          <span className="ml-2 text-sm font-normal text-zinc-400">
            P(Y | do(X)) — sever incoming edges to X, re-simulate, report ATE
          </span>
        </h1>
      </div>

      <div className="flex flex-1 overflow-hidden">
        {/* Sidebar */}
        <div className="w-96 border-r border-zinc-200 dark:border-zinc-800 bg-white dark:bg-zinc-900 p-5 overflow-y-auto shrink-0">
          <HowToUse
            id="l2-intervention"
            steps={[
              <>This is the <strong>canonical L2 interface</strong>. Unlike the Forecast page (which intervenes only on the supply-shock signal), this page lets you do() any structural parameter.</>,
              <>Pick a <strong>scenario</strong>. The factual trajectory is the scenario run as-written.</>,
              <>Toggle any subset of <strong>structural parameters</strong> below. Each enabled parameter gets pinned to your chosen value via graph surgery — its incoming causal edges are severed.</>,
              <>Click <strong>Run L2 Intervention</strong>. The engine runs factual and intervened simulations and returns per-year ATE per outcome.</>,
              <>For each outcome, ATE_t = Y_t(intervention) − Y_t(factual). Positive ATE on Price means the intervention raised price; negative means it lowered.</>,
            ]}
            tip="This calls POST /api/pearl/l2/do — the same endpoint your thesis methods section cites. The Forecast page is a thin shock-only wrapper; this is the full do-calculus surface."
          />

          <form onSubmit={handleRun} className="flex flex-col gap-4 mt-5">
            <div>
              <label className="text-[10px] font-semibold text-zinc-400 uppercase tracking-wider mb-1.5 block">Scenario (factual)</label>
              <select
                value={scenario}
                onChange={(e) => setScenario(e.target.value)}
                className="w-full text-sm border border-zinc-200 dark:border-zinc-800 rounded-lg px-3 py-2 bg-white dark:bg-zinc-900 focus:outline-none focus:ring-2 focus:ring-indigo-500"
              >
                {scenarios.map((s) => (
                  <option key={s.name} value={s.name}>{s.name}</option>
                ))}
              </select>
            </div>

            <div>
              <p className="text-[10px] font-semibold text-zinc-400 uppercase tracking-wider mb-2">do(·) parameter overrides</p>
              <div className="flex flex-col gap-2">
                {(Object.keys(PARAM_META) as L2Param[]).map((p) => (
                  <ParamRow
                    key={p}
                    param={p}
                    enabled={enabled[p]}
                    value={values[p]}
                    onToggle={(v) => setEnabled({ ...enabled, [p]: v })}
                    onValue={(v) => setValues({ ...values, [p]: v })}
                  />
                ))}
              </div>
            </div>

            <div>
              <p className="text-[10px] font-semibold text-zinc-400 uppercase tracking-wider mb-2">Outcomes to report</p>
              <div className="flex flex-wrap gap-2">
                {OUTCOMES.map((o) => {
                  const on = outcomes.includes(o);
                  return (
                    <button
                      key={o}
                      type="button"
                      onClick={() => setOutcomes(on ? outcomes.filter((x) => x !== o) : [...outcomes, o])}
                      className={`text-[11px] px-2 py-1 rounded border ${on ? 'bg-indigo-50 dark:bg-indigo-500/10 border-indigo-300 text-indigo-700 dark:text-indigo-300' : 'border-zinc-200 dark:border-zinc-700 text-zinc-500'}`}
                    >
                      {o}
                    </button>
                  );
                })}
              </div>
            </div>

            <button
              type="submit"
              disabled={loading || !scenario || enabledParams.length === 0 || outcomes.length === 0}
              className="mt-2 w-full bg-gradient-to-r from-indigo-600 to-violet-600 text-white text-sm font-semibold py-2.5 rounded-lg hover:opacity-95 disabled:opacity-40 transition-opacity"
            >
              {loading ? 'Running L2…' : 'Run L2 Intervention'}
            </button>

            <p className="text-[10px] text-zinc-400 leading-relaxed">
              Backend call: <span className="font-mono">POST /api/pearl/l2/do</span>.
              Source: <span className="font-mono">src/minerals/pearl_layers.py</span> →{' '}
              <span className="font-mono">do_compare</span> +{' '}
              <span className="font-mono">mutilated_graph_for_do</span>.
            </p>
          </form>
        </div>

        {/* Result panel */}
        <div className="flex-1 overflow-auto p-6">
          <div className="mb-5 grid grid-cols-1 xl:grid-cols-2 gap-5">
            <OutputGuide
              rung="L2"
              intro="The result has three sections: (1) intervention summary chips, (2) Mean ATE tiles per outcome, (3) one factual-vs-intervention line chart per outcome. Read them as 'what happens to Y if I force X = x going forward.'"
              fields={[
                { name: 'do(param)',
                  meaning: 'the structural-parameter value pinned by graph surgery for this run',
                  read: 'do(substitution_elasticity=0.8) → that parameter is held at 0.8 for the entire run, regardless of upstream causes' },
                { name: 'Mean ATE per outcome',
                  meaning: 'time-averaged Average Treatment Effect: (1/T)·Σ_t [Y_t(intervention) − Y_t(factual)]. Sign tells direction; magnitude tells size.',
                  read: 'ATE(P) = −0.015 → on average, the intervention lowered price by 0.015 normalised units across the horizon' },
                { name: 'red ↑ / green ↓ tile colors',
                  meaning: 'red = intervention raised the outcome (bad for price/shortage, good for supply); green = lowered it. Color is opinionated against price-rising effects.' },
                { name: 'factual line (gray) vs intervention line (indigo)',
                  meaning: 'per-year trajectories under the un-intervened SCM (factual) vs. the mutilated SCM with do(·) applied (intervention)',
                  read: 'If the indigo line sits below the gray line for most of the horizon, the intervention reduced that outcome on average' },
                { name: 'ATE_t (per-year)',
                  meaning: 'the gap between intervention and factual at year t. Look at the chart: ATE_t at any year is the vertical distance between the two lines.' },
              ]}
              takeaway="L2 ATE is a forward-looking expectation: 'what would Y look like on average if we forced X = x from now on?' It is NOT 'what would have happened in some specific past' — that's L3."
            />
            <MathPanel
              rung="L2"
              title="What L2 (Intervention) computes"
              formal="P(Y | do(X = x))   —   sever the structural equation for X, pin X = x, re-simulate"
              equations={[
                {
                  label: "Graph surgery (mutilated SCM)",
                  code:
`Original SCM:    X_t  =  f_X( parents(X)_t , noise_X_t )
After do(X=x):   X_t  :=  x                  (incoming edges to X removed)

All other structural equations of the model remain unchanged.`,
                },
                {
                  label: "Example structural equation (substitution supply)",
                  code:
`Q_sub_t  =  shock.export_restriction_t · Q_t
             · clamp( 0, sub_cap,
                      sub_elasticity · max(0, P_t/P_ref − 1) )

do(sub_elasticity = 0.8) replaces the 'sub_elasticity' literal in
the equation above — the rest of the ODE continues to run.`,
                },
                {
                  label: "Average Treatment Effect per outcome",
                  code:
`ATE_t (Y)   =  Y_t^(intervention)  −  Y_t^(factual)
mean ATE(Y) =  (1/T) · Σ_t ATE_t (Y)             [reported per outcome]`,
                },
              ]}
              caveat="L2 answers 'what would Y look like if we forced X = x going forward?'. It runs a fresh forward simulation under the intervened SCM — it does NOT condition on a specific past trajectory. For that, use L3."
              source="src/minerals/pearl_layers.py — do_compare(), mutilated_graph_for_do(); model in src/minerals/model.py"
            />
          </div>

          {error && (
            <div className="bg-red-50 border border-red-200 text-red-800 text-sm rounded-lg p-3 mb-4">
              <p className="font-semibold mb-1">L2 intervention failed</p>
              <p className="text-red-700">{error}</p>
            </div>
          )}

          {!loading && !result && !error && (
            <div className="flex flex-col items-center justify-center h-full text-center text-zinc-400">
              <p className="text-sm font-medium text-zinc-500">No L2 intervention run yet</p>
              <p className="text-xs mt-1 max-w-md">
                Pick a scenario, enable parameters to intervene on, set their values, then
                click Run L2 Intervention. The engine performs graph surgery on the SCM
                and returns factual vs. intervened trajectories with per-outcome ATE.
              </p>
            </div>
          )}

          {loading && (
            <div className="flex flex-col items-center justify-center h-full text-center">
              <div className="h-10 w-10 border-2 border-indigo-200 border-t-indigo-600 rounded-full animate-spin mb-4" />
              <p className="text-sm font-medium text-zinc-700 dark:text-zinc-300">
                Graph surgery + running factual and intervened simulations…
              </p>
            </div>
          )}

          {result && !loading && <L2View r={result} />}
        </div>
      </div>
    </div>
  );
}

function ParamRow({
  param, enabled, value, onToggle, onValue,
}: {
  param: L2Param;
  enabled: boolean;
  value: number;
  onToggle: (v: boolean) => void;
  onValue: (v: number) => void;
}) {
  const meta = PARAM_META[param];
  return (
    <div className={`border rounded-lg p-2.5 transition-colors ${enabled ? 'border-indigo-300 bg-indigo-50/40 dark:bg-indigo-500/5' : 'border-zinc-200 dark:border-zinc-800'}`}>
      <label className="flex items-start gap-2 cursor-pointer">
        <input
          type="checkbox"
          checked={enabled}
          onChange={(e) => onToggle(e.target.checked)}
          className="mt-0.5"
        />
        <div className="flex-1">
          <p className="text-xs font-mono font-semibold text-zinc-800 dark:text-zinc-100">{meta.label}</p>
          <p className="text-[10px] text-zinc-500 leading-snug">{meta.desc}</p>
          <p className="text-[9px] text-zinc-400 mt-0.5 leading-snug">
            do(·) severs: <span className="font-mono text-zinc-500">{meta.severs}</span>
          </p>
        </div>
      </label>
      {enabled && (
        <div className="mt-2 flex items-center gap-2">
          <input
            type="range"
            min={meta.min}
            max={meta.max}
            step={meta.step}
            value={value}
            onChange={(e) => onValue(Number(e.target.value))}
            className="flex-1"
          />
          <input
            type="number"
            min={meta.min}
            max={meta.max}
            step={meta.step}
            value={value}
            onChange={(e) => onValue(Number(e.target.value))}
            className="w-20 text-xs font-mono border border-zinc-200 dark:border-zinc-700 rounded px-2 py-0.5 bg-white dark:bg-zinc-900"
          />
        </div>
      )}
    </div>
  );
}

function L2View({ r }: { r: L2InterventionResponse }) {
  const years = r.trajectory.map((t) => t.year);

  return (
    <div className="flex flex-col gap-5">
      <div className="bg-indigo-50 dark:bg-indigo-500/10 border border-indigo-200 dark:border-indigo-600/40 text-indigo-900 dark:text-indigo-200 text-xs rounded-lg p-3">
        <p className="font-semibold mb-1">{r.layer}</p>
        <p>{r.description}</p>
      </div>

      <div className="flex flex-wrap gap-2">
        <Stat label="Scenario" value={r.scenario} mono />
        {Object.entries(r.intervention).map(([k, v]) => (
          <Stat key={k} label={`do(${k})`} value={String(v)} mono />
        ))}
      </div>

      {/* ATE panel */}
      <div className="bg-white dark:bg-zinc-900 border border-zinc-200 dark:border-zinc-800 rounded-lg p-4">
        <p className="text-[10px] font-semibold text-zinc-400 uppercase tracking-wider mb-3">
          Mean ATE per outcome (intervention − factual, averaged over horizon)
        </p>
        <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
          {Object.entries(r.ate_mean).map(([k, v]) => (
            <ATETile key={k} label={OUTCOME_LABEL[k] || k} value={v} />
          ))}
        </div>
      </div>

      {/* Trajectory charts: one chart per outcome, factual vs intervention overlaid */}
      {r.outcomes.map((o) => {
        const factual = r.trajectory.map((t) => Number(t[`${o}_factual`] ?? 0));
        const intervention = r.trajectory.map((t) => Number(t[`${o}_intervention`] ?? 0));
        return (
          <div key={o} className="bg-white dark:bg-zinc-900 border border-zinc-200 dark:border-zinc-800 rounded-lg p-4">
            <LineChart
              title={OUTCOME_LABEL[o] || o}
              years={years}
              height={200}
              series={[
                { label: 'factual',      data: factual,      color: '#94a3b8' },
                { label: 'intervention', data: intervention, color: '#6366f1' },
              ]}
            />
            <p className="text-[10px] text-zinc-400 mt-2">
              Mean ATE on <span className="font-mono">{o}</span>:{' '}
              <span className="font-mono font-semibold text-zinc-700 dark:text-zinc-300">
                {r.ate_mean[o] !== undefined ? r.ate_mean[o].toFixed(4) : '—'}
              </span>
            </p>
          </div>
        );
      })}

      <div className="bg-white dark:bg-zinc-900 border border-zinc-200 dark:border-zinc-800 rounded-lg p-4">
        <p className="text-[10px] font-semibold text-zinc-400 uppercase tracking-wider mb-2">
          Where this sits on Pearl&apos;s ladder
        </p>
        <ol className="text-xs text-zinc-600 dark:text-zinc-400 space-y-1 list-decimal list-inside">
          <li><strong>L1</strong>: P(Y|X) — observational. → /l1-association</li>
          <li><strong>L2 (you are here)</strong>: P(Y|do(X)) — explicit graph surgery on a structural parameter, re-simulate, report ATE.</li>
          <li><strong>L3</strong>: P(Y_x | X′, Y′) — abduction recovers latent noise from the factual, then re-runs under intervention. Twin-network coupling. → /counterfactual</li>
        </ol>
      </div>
    </div>
  );
}

function Stat({ label, value, mono }: { label: string; value: string; mono?: boolean }) {
  return (
    <div className="bg-white dark:bg-zinc-900 border border-zinc-200 dark:border-zinc-800 rounded-lg px-3 py-2">
      <p className="text-[9px] font-semibold text-zinc-400 uppercase tracking-wider">{label}</p>
      <p className={`text-sm font-bold text-zinc-800 dark:text-zinc-100 ${mono ? 'font-mono' : ''}`}>{value}</p>
    </div>
  );
}

function ATETile({ label, value }: { label: string; value: number }) {
  const color = value > 1e-6 ? 'text-red-600' : value < -1e-6 ? 'text-emerald-600' : 'text-zinc-500';
  const arrow = value > 1e-6 ? '↑' : value < -1e-6 ? '↓' : '·';
  return (
    <div>
      <p className="text-[10px] text-zinc-500">{label}</p>
      <p className={`text-xl font-bold font-mono ${color}`}>
        {arrow} {value.toFixed(4)}
      </p>
    </div>
  );
}
