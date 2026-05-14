'use client';

import { useEffect, useState } from 'react';
import HowToUse from '@/components/HowToUse';
import MathPanel from '@/components/MathPanel';
import OutputGuide from '@/components/OutputGuide';
import {
  getScenarios,
  runL1Association,
  type L1AssociationResponse,
} from '@/lib/api';
import type { ScenarioMeta } from '@/lib/types';

export default function L1AssociationPage() {
  const [scenarios, setScenarios] = useState<ScenarioMeta[]>([]);
  const [scenario, setScenario] = useState('graphite_baseline');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<L1AssociationResponse | null>(null);
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

  const handleRun = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const r = await runL1Association(scenario);
      setResult(r);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'L1 query failed');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex flex-col h-screen bg-zinc-50 dark:bg-zinc-950">
      {/* Header */}
      <div className="border-b border-zinc-200 dark:border-zinc-800 bg-white dark:bg-zinc-900 px-6 py-3 shrink-0">
        <div className="flex items-center gap-2 mb-0.5">
          <span className="text-[10px] font-bold text-indigo-700 bg-indigo-100 dark:bg-indigo-500/15 dark:text-indigo-300 rounded px-1.5 py-0.5">
            L1
          </span>
          <p className="text-[10px] font-semibold text-indigo-600 uppercase tracking-widest">
            Pearl Layer 1 — Association
          </p>
        </div>
        <h1 className="text-lg font-bold text-zinc-900 dark:text-zinc-100">
          Observational Association
          <span className="ml-2 text-sm font-normal text-zinc-400">
            P(Y | X) — what correlates with what, without intervening
          </span>
        </h1>
      </div>

      <div className="flex flex-1 overflow-hidden">
        {/* Sidebar */}
        <div className="w-80 border-r border-zinc-200 dark:border-zinc-800 bg-white dark:bg-zinc-900 p-5 overflow-y-auto shrink-0">
          <HowToUse
            id="l1-association"
            steps={[
              <>Pick a <strong>scenario</strong>. The engine runs the ODE forward under the scenario&apos;s shock sequence and records every state {'{K, I, P, Q_sub, Q_fringe}'} per year.</>,
              <>Click <strong>Run L1 Association</strong>. The endpoint computes purely observational statistics: Spearman ρ between substitution supply and export restriction, plus fringe supply vs. price quartile.</>,
              <>These numbers are <strong>correlations only</strong>. They tell you "when restriction was on, did substitution rise?" — but not whether the restriction <em>caused</em> the substitution rise (a confounder could explain it).</>,
              <>For causal estimates: <strong>L2</strong> (Forecast page) breaks the confounders via graph surgery, <strong>L3</strong> (Counterfactual page) computes "what would have been" under a fixed factual history.</>,
            ]}
            tip="L1 is the ladder rung you stand on without manipulating the system. It can't distinguish causation from confounding — that's what L2 and L3 exist to do."
          />

          <form onSubmit={handleRun} className="flex flex-col gap-4 mt-5">
            <div>
              <label className="text-[10px] font-semibold text-zinc-400 uppercase tracking-wider mb-1.5 block">
                Scenario
              </label>
              <select
                value={scenario}
                onChange={(e) => setScenario(e.target.value)}
                className="w-full text-sm border border-zinc-200 dark:border-zinc-800 rounded-lg px-3 py-2 bg-white dark:bg-zinc-900 focus:outline-none focus:ring-2 focus:ring-indigo-500"
              >
                {scenarios.map((s) => (
                  <option key={s.name} value={s.name}>
                    {s.name}
                  </option>
                ))}
              </select>
              <p className="text-[10px] text-zinc-400 mt-1">
                Scenario YAML loaded from <span className="font-mono">scenarios/</span>.
              </p>
            </div>

            <button
              type="submit"
              disabled={loading || !scenario}
              className="mt-2 w-full bg-gradient-to-r from-indigo-600 to-violet-600 text-white text-sm font-semibold py-2.5 rounded-lg hover:opacity-95 disabled:opacity-40 transition-opacity"
            >
              {loading ? 'Running L1…' : 'Run L1 Association'}
            </button>

            <p className="text-[10px] text-zinc-400 leading-relaxed">
              Backend call: <span className="font-mono">POST /api/pearl/l1/association</span>.
              Source: <span className="font-mono">src/minerals/pearl_layers.py</span> →{' '}
              <span className="font-mono">observe_substitution_association</span>,{' '}
              <span className="font-mono">observe_fringe_association</span>.
            </p>
          </form>
        </div>

        {/* Result panel */}
        <div className="flex-1 overflow-auto p-6">
          <div className="mb-5 grid grid-cols-1 xl:grid-cols-2 gap-5">
            <OutputGuide
              rung="L1"
              intro="The result has two tables (substitution + fringe), each with one row per bin. Read each row as: 'in this bin, the observed quantity behaved like…'"
              fields={[
                { name: 'count',
                  meaning: 'sample size — number of simulation years in this bin',
                  read: 'count=11 means 11 of the simulated years fell into this bin' },
                { name: 'mean_Q_sub / mean_Q_fringe',
                  meaning: 'average substitution-supply or fringe-supply level inside the bin (units: same as Q)',
                  read: 'mean_Q_fringe=4.97 in the Q4 (highest-price) bin → fringe producers entered when price was high' },
                { name: 'std_Q_sub / std_Q_fringe',
                  meaning: 'within-bin variability of the quantity. If std=0 the quantity was constant in this bin.' },
                { name: 'spearman_rho_…',
                  meaning: 'rank correlation between the quantity and the binning variable (range −1 to +1)',
                  read: 'ρ=+0.76 means strong positive monotone association (the quantity rises with the binning variable in rank-order)' },
                { name: 'spearman_pval',
                  meaning: 'p-value under H₀ of zero correlation. Small p → unlikely to be noise.',
                  read: 'p=0.027 → significant at the 5% level (but with this small N treat any p > 0.01 with caution)' },
                { name: 'null (None)',
                  meaning: 'a None means the statistic is undefined for that bin — usually because there was no variance (e.g. Q_sub = 0 throughout)' },
              ]}
              takeaway="A large |ρ| with small p says 'X and Y co-move'. It does NOT say 'X caused Y'. To distinguish, climb the ladder to L2 (intervention) or L3 (counterfactual)."
            />
            <MathPanel
              rung="L1"
              title="What L1 (Association) computes"
              formal="P(Y | X)   —   conditional distribution of Y given observed X, no manipulation"
              equations={[
                {
                  label: "Substitution association — Spearman rank correlation",
                  code:
`bin = { restricted: years where shock.export_restriction > 0,
        normal:     years where shock.export_restriction = 0 }

ρ_sub  = corr_Spearman( Q_sub_t , shock.export_restriction_t )
         within each bin   (return ρ, p-value, mean & std of Q_sub)`,
                },
                {
                  label: "Fringe association — Spearman by price quartile",
                  code:
`Q1..Q4 = price quartiles over the simulation horizon

ρ_fringe = corr_Spearman( Q_fringe_t , P_t )
           within each quartile (return ρ, p-value, mean & std of Q_fringe)`,
                },
              ]}
              caveat="ρ measures association only. A non-zero ρ does NOT imply 'restriction caused substitution' — a confounder (e.g. policy regime, year fixed effects) could explain both. Use L2 to sever those confounders via graph surgery."
              source="src/minerals/pearl_layers.py — observe_substitution_association(), observe_fringe_association()"
            />
          </div>

          {error && (
            <div className="bg-red-50 border border-red-200 text-red-800 text-sm rounded-lg p-3 mb-4">
              <p className="font-semibold mb-1">L1 query failed</p>
              <p className="text-red-700">{error}</p>
            </div>
          )}

          {!loading && !result && !error && (
            <div className="flex flex-col items-center justify-center h-full text-center text-zinc-400">
              <p className="text-sm font-medium text-zinc-500">No L1 query run yet</p>
              <p className="text-xs mt-1 max-w-md">
                Pick a scenario on the left and click Run L1 Association. The result
                shows observational correlations — the first rung of Pearl&apos;s
                ladder, before any intervention or counterfactual reasoning.
              </p>
            </div>
          )}

          {loading && (
            <div className="flex flex-col items-center justify-center h-full text-center">
              <div className="h-10 w-10 border-2 border-indigo-200 border-t-indigo-600 rounded-full animate-spin mb-4" />
              <p className="text-sm font-medium text-zinc-700 dark:text-zinc-300">
                Running ODE + computing associations…
              </p>
            </div>
          )}

          {result && !loading && <L1View r={result} />}
        </div>
      </div>
    </div>
  );
}

function L1View({ r }: { r: L1AssociationResponse }) {
  return (
    <div className="flex flex-col gap-5">
      <div className="bg-amber-50 border border-amber-300 dark:bg-amber-500/10 dark:border-amber-600/40 text-amber-900 dark:text-amber-200 text-xs rounded-lg p-3">
        <p className="font-semibold mb-1">⚠ Observational only — not causal</p>
        <p>{r.warning}</p>
      </div>

      <div>
        <div className="flex items-center gap-2 mb-2">
          <span className="text-[10px] font-bold text-indigo-700 bg-indigo-100 dark:bg-indigo-500/15 dark:text-indigo-300 rounded px-1.5 py-0.5">
            {r.layer}
          </span>
          <p className="text-xs text-zinc-500">
            Scenario: <span className="font-mono text-zinc-700 dark:text-zinc-300">{r.scenario}</span>
          </p>
        </div>
      </div>

      <AssociationTable
        title="Substitution association — Q_sub vs export_restriction"
        rows={Array.isArray(r.substitution_association) ? r.substitution_association : []}
        error={!Array.isArray(r.substitution_association) ? (r.substitution_association as { error: string }).error : undefined}
        note="Years are binned by whether the dominant-supplier restriction was active. Positive ρ means: in the observational data, substitution supply tends to co-move with restriction status."
      />

      <AssociationTable
        title="Fringe association — Q_fringe vs price quartile"
        rows={Array.isArray(r.fringe_association) ? r.fringe_association : []}
        error={!Array.isArray(r.fringe_association) ? (r.fringe_association as { error: string }).error : undefined}
        note="Years binned by price quartile (Q1 = lowest 25% of prices, Q4 = highest). High fringe supply in upper quartiles is consistent with the cost-curve story — but L1 cannot rule out a confounder."
      />

      <div className="bg-white dark:bg-zinc-900 border border-zinc-200 dark:border-zinc-800 rounded-lg p-4">
        <p className="text-[10px] font-semibold text-zinc-400 uppercase tracking-wider mb-2">
          Where this sits on Pearl&apos;s ladder
        </p>
        <ol className="text-xs text-zinc-600 dark:text-zinc-400 space-y-1 list-decimal list-inside">
          <li><strong>L1 (you are here)</strong>: P(Y|X) — observational. Sees the world but cannot manipulate it.</li>
          <li><strong>L2</strong>: P(Y|do(X)) — interventional. Severs incoming edges to X via graph surgery. → /forecast or /api/pearl/l2/do</li>
          <li><strong>L3</strong>: P(Y_x | X′, Y′) — counterfactual. Conditions on the factual history then re-runs under the intervention. → /counterfactual</li>
        </ol>
      </div>
    </div>
  );
}

function AssociationTable({
  title, rows, error, note,
}: { title: string; rows: Record<string, unknown>[]; error?: string; note: string }) {
  return (
    <div className="bg-white dark:bg-zinc-900 border border-zinc-200 dark:border-zinc-800 rounded-lg p-4">
      <p className="text-xs font-semibold text-zinc-700 dark:text-zinc-200 mb-2">{title}</p>
      {error ? (
        <p className="text-xs text-red-600">Error: {error}</p>
      ) : rows.length === 0 ? (
        <p className="text-xs text-zinc-400">No rows returned (scenario may not exercise this channel).</p>
      ) : (
        <div className="overflow-x-auto">
          <table className="text-xs w-full">
            <thead>
              <tr className="border-b border-zinc-200 dark:border-zinc-800">
                {Object.keys(rows[0]).map((k) => (
                  <th key={k} className="text-left py-1.5 px-2 font-semibold text-zinc-500 uppercase text-[10px] tracking-wider">
                    {k}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {rows.map((row, i) => (
                <tr key={i} className="border-b border-zinc-100 dark:border-zinc-800/50">
                  {Object.entries(row).map(([k, v]) => (
                    <td key={k} className="py-1.5 px-2 font-mono text-zinc-700 dark:text-zinc-300">
                      {typeof v === 'number' ? v.toFixed(4) : String(v)}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      <p className="text-[10px] text-zinc-400 mt-2 leading-relaxed">{note}</p>
    </div>
  );
}
