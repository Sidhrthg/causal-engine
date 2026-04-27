import Link from 'next/link';

const VALIDATION = [
  { label: 'In-sample DA', value: '91.7%', sub: '10 episodes', color: 'text-emerald-600' },
  { label: 'Clean OOS DA', value: '65.7%', sub: '5 transfer pairs', color: 'text-indigo-600' },
  { label: 'vs Momentum', value: '+33.4pp', sub: 'directional accuracy', color: 'text-violet-600' },
  { label: 'vs Conc. Heuristic', value: '+23.6pp', sub: 'directional accuracy', color: 'text-blue-600' },
];

const FEATURES = [
  {
    href: '/transshipment',
    label: 'Transshipment Detection',
    desc: 'Trace multi-hop CEPII trade routes. Estimate circumvention rates with bootstrap CIs.',
    icon: (
      <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.75} d="M8 7h12m0 0l-4-4m4 4l-4 4m0 6H4m0 0l4 4m-4-4l4-4" />
      </svg>
    ),
    accent: 'bg-indigo-50 text-indigo-600 border-indigo-100',
    cta: 'Analyze routes →',
  },
  {
    href: '/counterfactual',
    label: 'Counterfactual Analysis',
    desc: 'Pearl L3 abduction-action-prediction. Fix noise, change mechanism, see what would have been.',
    icon: (
      <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.75} d="M13 10V3L4 14h7v7l9-11h-7z" />
      </svg>
    ),
    accent: 'bg-purple-50 text-purple-600 border-purple-100',
    cta: 'Run counterfactual →',
  },
  {
    href: '/query',
    label: 'Knowledge Query',
    desc: 'Ask questions against 1,661 USGS/CEPII/IEA chunks via HippoRAG personalized PageRank.',
    icon: (
      <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.75} d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" />
      </svg>
    ),
    accent: 'bg-blue-50 text-blue-600 border-blue-100',
    cta: 'Ask a question →',
  },
  {
    href: '/knowledge-graph',
    label: 'Knowledge Graph',
    desc: 'Explore 3,132 entities and 31,916 causal edges. Filter by commodity or relationship type.',
    icon: (
      <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.75} d="M4 6a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2H6a2 2 0 01-2-2V6zM14 6a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2h-2a2 2 0 01-2-2V6zM4 16a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2H6a2 2 0 01-2-2v-2zM14 16a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2h-2a2 2 0 01-2-2v-2z" />
      </svg>
    ),
    accent: 'bg-emerald-50 text-emerald-600 border-emerald-100',
    cta: 'Explore graph →',
  },
  {
    href: '/shock-extractor',
    label: 'Shock Extractor',
    desc: 'Paste any news article. KG extracts shocks, runs the ODE model, returns a price trajectory.',
    icon: (
      <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.75} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
      </svg>
    ),
    accent: 'bg-amber-50 text-amber-600 border-amber-100',
    cta: 'Extract shocks →',
  },
  {
    href: '/scenario-builder',
    label: 'Scenario Builder',
    desc: 'Pick a pre-rendered episode or generate a custom KG render via HippoRAG + Claude triple extraction.',
    icon: (
      <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.75} d="M12 6v6m0 0v6m0-6h6m-6 0H6" />
      </svg>
    ),
    accent: 'bg-violet-50 text-violet-600 border-violet-100',
    cta: 'Build a scenario →',
  },
  {
    href: '/kg-enrich',
    label: 'KG Enrich',
    desc: 'Grow the causal KG: HippoRAG retrieves chunks, Claude extracts triples, merged into the live graph.',
    icon: (
      <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.75} d="M12 4v16m8-8H4" />
      </svg>
    ),
    accent: 'bg-rose-50 text-rose-600 border-rose-100',
    cta: 'Enrich the KG →',
  },
];

const METHODS = [
  {
    level: 'L1',
    title: 'Association',
    subtitle: 'Seeing',
    desc: 'P(Y | X=x). Correlations from BACI bilateral flows — no causal claim.',
    border: 'border-zinc-200',
    bg: 'bg-zinc-50',
    badge: 'bg-zinc-100 text-zinc-500',
    text: 'text-zinc-700',
  },
  {
    level: 'L2',
    title: 'Intervention',
    subtitle: 'Doing',
    desc: 'P(Y | do(X=x)). Graph surgery on the SCM + calibrated parameters (η_D, α_P, τ_K).',
    border: 'border-indigo-200',
    bg: 'bg-indigo-50',
    badge: 'bg-indigo-100 text-indigo-600',
    text: 'text-indigo-800',
  },
  {
    level: 'L3',
    title: 'Counterfactual',
    subtitle: 'Imagining',
    desc: 'P(Y_x | X=x′). Abduction-action-prediction with twin-network SDE noise replay.',
    border: 'border-purple-200',
    bg: 'bg-purple-50',
    badge: 'bg-purple-100 text-purple-600',
    text: 'text-purple-800',
  },
];

const COMMODITIES = [
  { name: 'Graphite', icon: '◆', hs: 'HS 250490', producers: 'China · Madagascar · Mozambique', slug: 'graphite', accent: 'text-slate-600 bg-slate-50 border-slate-200' },
  { name: 'Lithium', icon: '⬡', hs: 'HS 283691', producers: 'Australia · Chile · China', slug: 'lithium', accent: 'text-blue-600 bg-blue-50 border-blue-200' },
  { name: 'Cobalt', icon: '◉', hs: 'HS 810520', producers: 'DRC · Russia · Australia', slug: 'cobalt', accent: 'text-violet-600 bg-violet-50 border-violet-200' },
  { name: 'Nickel', icon: '○', hs: 'HS 750110', producers: 'Indonesia · Philippines · Russia', slug: 'nickel', accent: 'text-emerald-600 bg-emerald-50 border-emerald-200' },
];

export default function DashboardPage() {
  return (
    <div className="min-h-screen bg-zinc-50 dark:bg-zinc-950">
      <div className="max-w-4xl mx-auto px-8 py-10">

        {/* Hero — animated entrance */}
        <div className="mb-10">
          <div className="inline-flex items-center gap-2 text-[11px] font-semibold text-indigo-600 dark:text-indigo-400 uppercase tracking-widest mb-4 animate-fade-in-up">
            <span className="h-1.5 w-1.5 rounded-full bg-indigo-500 inline-block animate-pulse-soft" />
            PhD Research · Critical Minerals Supply Chain Intelligence
          </div>
          <h1 className="text-4xl font-bold text-zinc-900 dark:text-zinc-100 leading-tight mb-3 animate-fade-in-up delay-1">
            Critical Minerals<br />
            <span
              className="bg-gradient-to-r from-indigo-600 via-violet-600 to-fuchsia-600 bg-clip-text text-transparent animate-shimmer"
              style={{ backgroundImage: 'linear-gradient(90deg,#4f46e5 0%,#7c3aed 33%,#c026d3 66%,#7c3aed 100%)' }}
            >
              Causal Engine
            </span>
          </h1>
          <p className="text-base text-zinc-500 dark:text-zinc-400 max-w-2xl leading-relaxed mb-6 animate-fade-in-up delay-2">
            Pearl&apos;s Ladder of Causation (L1/L2/L3) over an ODE commodity model, validated against CEPII BACI
            bilateral trade data. Identify supply shocks, trace trade circumvention, and answer
            counterfactual policy questions for graphite, lithium, cobalt, nickel, and uranium.
          </p>
          <div className="flex gap-3 animate-fade-in-up delay-3">
            <Link
              href="/scenario-builder"
              className="inline-flex items-center gap-2 px-5 py-2.5 bg-gradient-to-r from-indigo-600 to-violet-600 text-white text-sm font-semibold rounded-xl hover:opacity-95 transition-opacity shadow-sm shadow-indigo-500/30"
            >
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 6v6m0 0v6m0-6h6m-6 0H6" />
              </svg>
              Build a scenario
            </Link>
            <Link
              href="/temporal-comparison"
              className="inline-flex items-center gap-2 px-5 py-2.5 bg-white dark:bg-zinc-900 text-zinc-700 dark:text-zinc-300 text-sm font-semibold rounded-xl border border-zinc-200 dark:border-zinc-800 hover:border-zinc-300 dark:hover:border-zinc-700 hover-lift transition-colors"
            >
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
              See supply chains shift
            </Link>
          </div>
        </div>

        {/* Pipeline visualization — what the engine actually does */}
        <section className="mb-12 animate-fade-in-up delay-4">
          <h2 className="text-[11px] font-semibold text-zinc-400 uppercase tracking-wider mb-4">
            The pipeline · Documents → KG → ODE → Counterfactual
          </h2>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
            {/* Step 1 — Documents to KG */}
            <div className="bg-white dark:bg-zinc-900 rounded-xl border border-zinc-200 dark:border-zinc-800 p-4 hover-lift">
              <svg viewBox="0 0 200 80" className="w-full h-20 mb-3">
                <defs>
                  <linearGradient id="g1" x1="0" x2="1" y1="0" y2="0">
                    <stop offset="0%" stopColor="#6366f1" /><stop offset="100%" stopColor="#a855f7" />
                  </linearGradient>
                </defs>
                {/* Document stack */}
                <g className="animate-float">
                  <rect x="10" y="20" width="36" height="48" rx="3" fill="white" stroke="#cbd5e1" strokeWidth="1.5" />
                  <line x1="16" y1="32" x2="40" y2="32" stroke="#cbd5e1" strokeWidth="1" />
                  <line x1="16" y1="40" x2="36" y2="40" stroke="#cbd5e1" strokeWidth="1" />
                  <line x1="16" y1="48" x2="40" y2="48" stroke="#cbd5e1" strokeWidth="1" />
                </g>
                {/* Arrow */}
                <path d="M55 44 L90 44" stroke="url(#g1)" strokeWidth="2" strokeLinecap="round" markerEnd="url(#arr1)" />
                <defs><marker id="arr1" markerWidth="8" markerHeight="8" refX="6" refY="4" orient="auto">
                  <path d="M0,0 L6,4 L0,8" fill="#a855f7" /></marker></defs>
                {/* KG nodes */}
                <circle cx="110" cy="30" r="6" fill="#6366f1" className="animate-pulse-soft" />
                <circle cx="150" cy="30" r="6" fill="#10b981" className="animate-pulse-soft" style={{ animationDelay: '0.4s' }} />
                <circle cx="180" cy="50" r="6" fill="#f59e0b" className="animate-pulse-soft" style={{ animationDelay: '0.8s' }} />
                <circle cx="125" cy="60" r="6" fill="#ec4899" className="animate-pulse-soft" style={{ animationDelay: '1.2s' }} />
                <line x1="110" y1="30" x2="150" y2="30" stroke="#94a3b8" strokeWidth="1" opacity="0.6" />
                <line x1="150" y1="30" x2="180" y2="50" stroke="#94a3b8" strokeWidth="1" opacity="0.6" />
                <line x1="110" y1="30" x2="125" y2="60" stroke="#94a3b8" strokeWidth="1" opacity="0.6" />
                <line x1="125" y1="60" x2="150" y2="30" stroke="#94a3b8" strokeWidth="1" opacity="0.6" />
              </svg>
              <p className="text-[10px] font-semibold text-indigo-600 dark:text-indigo-400 uppercase tracking-wider mb-1">Step 1</p>
              <p className="text-sm font-bold text-zinc-900 dark:text-zinc-100 mb-1">Documents → Knowledge Graph</p>
              <p className="text-xs text-zinc-500 dark:text-zinc-400 leading-relaxed">
                HippoRAG retrieves USGS / IEA / CEPII passages. Claude extracts (subject, relation, object) triples. 472 entities, 588 edges.
              </p>
            </div>

            {/* Step 2 — KG to shocks/trajectory */}
            <div className="bg-white dark:bg-zinc-900 rounded-xl border border-zinc-200 dark:border-zinc-800 p-4 hover-lift">
              <svg viewBox="0 0 200 80" className="w-full h-20 mb-3">
                <defs>
                  <linearGradient id="g2" x1="0" x2="0" y1="0" y2="1">
                    <stop offset="0%" stopColor="#fb923c" stopOpacity="0.4" /><stop offset="100%" stopColor="#fb923c" stopOpacity="0" />
                  </linearGradient>
                </defs>
                {/* Y-axis tick lines */}
                <line x1="10" y1="20" x2="190" y2="20" stroke="#e2e8f0" strokeWidth="0.5" strokeDasharray="2,2" />
                <line x1="10" y1="40" x2="190" y2="40" stroke="#e2e8f0" strokeWidth="0.5" strokeDasharray="2,2" />
                <line x1="10" y1="60" x2="190" y2="60" stroke="#e2e8f0" strokeWidth="0.5" strokeDasharray="2,2" />
                {/* Shock spike */}
                <path
                  d="M 10,55 L 40,55 L 60,30 L 90,38 L 120,15 L 150,28 L 180,42 L 190,40"
                  fill="none"
                  stroke="#ea580c"
                  strokeWidth="2.2"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  className="animate-draw-line"
                />
                {/* Shock highlight area */}
                <path
                  d="M 60,30 L 90,38 L 120,15 L 150,28 L 150,68 L 60,68 Z"
                  fill="url(#g2)"
                  opacity="0.5"
                />
                <circle cx="120" cy="15" r="3" fill="#dc2626" className="animate-pulse-soft" />
              </svg>
              <p className="text-[10px] font-semibold text-amber-600 dark:text-amber-400 uppercase tracking-wider mb-1">Step 2</p>
              <p className="text-sm font-bold text-zinc-900 dark:text-zinc-100 mb-1">KG → ODE shocks → Trajectory</p>
              <p className="text-xs text-zinc-500 dark:text-zinc-400 leading-relaxed">
                Year-specific shares feed the {'{K, I, P}'} ODE. Pearl L2 do-calculus on substitution and fringe supply nodes. Annual price index forward.
              </p>
            </div>

            {/* Step 3 — Counterfactual */}
            <div className="bg-white dark:bg-zinc-900 rounded-xl border border-zinc-200 dark:border-zinc-800 p-4 hover-lift">
              <svg viewBox="0 0 200 80" className="w-full h-20 mb-3">
                {/* Factual */}
                <path
                  d="M 10,55 L 50,52 L 90,42 L 130,18 L 170,15 L 190,16"
                  fill="none"
                  stroke="#dc2626"
                  strokeWidth="2.2"
                  strokeLinecap="round"
                  className="animate-draw-line"
                />
                {/* Counterfactual */}
                <path
                  d="M 10,55 L 50,53 L 90,50 L 130,46 L 170,42 L 190,40"
                  fill="none"
                  stroke="#10b981"
                  strokeWidth="2.2"
                  strokeLinecap="round"
                  strokeDasharray="4,3"
                  className="animate-draw-line"
                  style={{ animationDelay: '0.5s' }}
                />
                {/* Legend dots */}
                <circle cx="190" cy="16" r="3" fill="#dc2626" />
                <circle cx="190" cy="40" r="3" fill="#10b981" />
                <text x="178" y="14" fontSize="7" fill="#dc2626" textAnchor="end">factual</text>
                <text x="178" y="38" fontSize="7" fill="#10b981" textAnchor="end">no-shock</text>
              </svg>
              <p className="text-[10px] font-semibold text-emerald-600 dark:text-emerald-400 uppercase tracking-wider mb-1">Step 3</p>
              <p className="text-sm font-bold text-zinc-900 dark:text-zinc-100 mb-1">Pearl L3 — what would have been</p>
              <p className="text-xs text-zinc-500 dark:text-zinc-400 leading-relaxed">
                Abduction-Action-Prediction. Recover noise from the realised trajectory, intervene, replay. Quantifies the causal price premium of a policy.
              </p>
            </div>
          </div>
        </section>

        {/* Validation metrics strip */}
        <section className="mb-10">
          <h2 className="text-[11px] font-semibold text-zinc-400 uppercase tracking-wider mb-3">
            Model validation — CEPII BACI price series
          </h2>
          <div className="grid grid-cols-4 gap-3">
            {VALIDATION.map(({ label, value, sub, color }) => (
              <div key={label} className="bg-white dark:bg-zinc-900 rounded-xl border border-zinc-200 dark:border-zinc-800 p-4 shadow-sm">
                <p className={`text-2xl font-bold tabular-nums ${color}`}>{value}</p>
                <p className="text-xs font-semibold text-zinc-700 dark:text-zinc-300 mt-1">{label}</p>
                <p className="text-[10px] text-zinc-400 mt-0.5">{sub}</p>
              </div>
            ))}
          </div>
          <p className="text-[10px] text-zinc-400 mt-2 leading-relaxed">
            Directional accuracy (DA) on price index year-on-year moves · OOS = parameters transferred
            across episodes without re-fitting · Baselines receive no shock information
          </p>
        </section>

        {/* Feature cards */}
        <section className="mb-10">
          <h2 className="text-[11px] font-semibold text-zinc-400 uppercase tracking-wider mb-3">
            Tools
          </h2>
          <div className="grid grid-cols-1 gap-3">
            {FEATURES.map(({ href, label, desc, icon, accent, cta }) => (
              <Link
                key={href}
                href={href}
                className="group flex items-start gap-4 bg-white dark:bg-zinc-900 rounded-xl border border-zinc-200 dark:border-zinc-800 p-4 hover:border-indigo-300 dark:hover:border-indigo-500/50 hover:shadow-md hover-lift transition-all"
              >
                <div className={`h-9 w-9 rounded-lg border flex items-center justify-center shrink-0 mt-0.5 ${accent}`}>
                  {icon}
                </div>
                <div className="flex-1 min-w-0">
                  <p className="text-sm font-semibold text-zinc-800 dark:text-zinc-200 mb-0.5">{label}</p>
                  <p className="text-xs text-zinc-500 leading-relaxed">{desc}</p>
                </div>
                <span className="text-xs font-medium text-indigo-500 group-hover:text-indigo-700 transition-colors shrink-0 mt-1 pr-1 whitespace-nowrap">
                  {cta}
                </span>
              </Link>
            ))}
          </div>
        </section>

        {/* Pearl ladder */}
        <section className="mb-10">
          <h2 className="text-[11px] font-semibold text-zinc-400 uppercase tracking-wider mb-3">
            Causal framework — Pearl&apos;s Ladder
          </h2>
          <div className="grid grid-cols-3 gap-3">
            {METHODS.map(({ level, title, subtitle, desc, border, bg, badge, text }) => (
              <div key={level} className={`rounded-xl border p-4 ${bg} ${border}`}>
                <div className="flex items-center gap-2 mb-2.5">
                  <span className={`text-xs font-bold px-1.5 py-0.5 rounded ${badge}`}>
                    {level}
                  </span>
                  <div>
                    <p className={`text-xs font-semibold leading-none ${text}`}>{title}</p>
                    <p className="text-[10px] text-zinc-400 mt-0.5">{subtitle}</p>
                  </div>
                </div>
                <p className={`text-xs leading-relaxed ${text} opacity-75`}>{desc}</p>
              </div>
            ))}
          </div>
        </section>

        {/* Two columns: Commodities + KG stats */}
        <div className="grid grid-cols-2 gap-6 mb-10">
          {/* Commodities */}
          <section>
            <h2 className="text-[11px] font-semibold text-zinc-400 uppercase tracking-wider mb-3">
              Commodities
            </h2>
            <div className="flex flex-col gap-2">
              {COMMODITIES.map(({ name, icon, hs, producers, slug, accent }) => (
                <Link
                  key={name}
                  href={`/transshipment?commodity=${slug}`}
                  className="flex items-center gap-3 bg-white dark:bg-zinc-900 rounded-xl border border-zinc-200 dark:border-zinc-800 p-3 hover:border-indigo-300 hover:shadow-sm transition-all group"
                >
                  <span className={`text-[10px] font-medium px-1.5 py-0.5 rounded border ${accent} shrink-0`}>
                    {hs}
                  </span>
                  <div className="flex-1 min-w-0">
                    <p className="text-sm font-semibold text-zinc-800">
                      {icon} {name}
                    </p>
                    <p className="text-[10px] text-zinc-400 truncate">{producers}</p>
                  </div>
                  <svg className="w-4 h-4 text-zinc-300 group-hover:text-indigo-400 transition-colors shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                  </svg>
                </Link>
              ))}
            </div>
          </section>

          {/* KG stats */}
          <section>
            <h2 className="text-[11px] font-semibold text-zinc-400 uppercase tracking-wider mb-3">
              Knowledge graph
            </h2>
            <div className="grid grid-cols-2 gap-2 mb-3">
              {[
                { label: 'Docs indexed', value: '1,661', sub: 'chunks (500 tok)' },
                { label: 'OpenIE triples', value: '4,368', sub: 'gpt-4o-mini extracted' },
                { label: 'KG entities', value: '3,132', sub: 'unique after norm.' },
                { label: 'KG edges', value: '31,916', sub: 'igraph directed' },
              ].map(({ label, value, sub }) => (
                <div key={label} className="bg-white dark:bg-zinc-900 rounded-xl border border-zinc-200 dark:border-zinc-800 p-3">
                  <p className="text-xl font-bold tabular-nums text-zinc-900">{value}</p>
                  <p className="text-xs font-medium text-zinc-600 dark:text-zinc-400 mt-0.5">{label}</p>
                  <p className="text-[10px] text-zinc-400">{sub}</p>
                </div>
              ))}
            </div>
            <p className="text-[10px] text-zinc-400 leading-relaxed">
              HippoRAG personalized PageRank · text-embedding-3-large 3072-dim · gpt-4o-mini OpenIE cache
            </p>
          </section>
        </div>

        {/* Data sources */}
        <section>
          <h2 className="text-[11px] font-semibold text-zinc-400 uppercase tracking-wider mb-3">
            Data sources
          </h2>
          <div className="bg-white dark:bg-zinc-900 rounded-xl border border-zinc-200 dark:border-zinc-800 p-4">
            <div className="grid grid-cols-4 gap-4 text-xs">
              {[
                { name: 'CEPII BACI', detail: '1995–2024 · pre-reconciled bilateral flows · ISO 3166-1 normalized' },
                { name: 'USGS MCS', detail: '2020–2024 · 5 annual volumes · 17 commodities · reserves + production' },
                { name: 'IEA 2021', detail: 'Critical minerals supply chain risk · 20 chunks' },
                { name: 'DRC Reports', detail: '2 artisanal cobalt reports · 79 chunks' },
              ].map(({ name, detail }) => (
                <div key={name}>
                  <p className="font-semibold text-zinc-700 dark:text-zinc-300 mb-0.5">{name}</p>
                  <p className="text-zinc-400 leading-relaxed">{detail}</p>
                </div>
              ))}
            </div>
          </div>
        </section>

      </div>
    </div>
  );
}
