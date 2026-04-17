import Link from 'next/link';

const STEPS = [
  {
    n: '1',
    title: 'Start the backend',
    code: 'uvicorn src.api:app --reload',
    desc: 'Run from the project root. The backend serves CEPII + RAG endpoints.',
  },
  {
    n: '2',
    title: 'Pick a scenario',
    desc: 'Go to Transshipment → click a quick scenario (e.g. "China → USA Graphite") → hit Run Analysis.',
    code: null,
  },
  {
    n: '3',
    title: 'Explore the routes',
    desc: 'See which countries intermediate the flow, identify circumvention candidates, and check the bootstrap-CI circumvention rate.',
    code: null,
  },
  {
    n: '4',
    title: 'Ask the knowledge base',
    desc: 'Go to Knowledge Query and ask "How does China control graphite anode processing?" to get sourced answers from USGS, CEPII, and IEA.',
    code: null,
  },
];

const COMMODITIES = [
  { name: 'Graphite', icon: '◆', hs: 'HS 250490', producers: 'China · Madagascar · Mozambique', slug: 'graphite', accent: 'text-slate-600 bg-slate-50 border-slate-200' },
  { name: 'Lithium', icon: '⬡', hs: 'HS 283691', producers: 'Australia · Chile · China', slug: 'lithium', accent: 'text-blue-600 bg-blue-50 border-blue-200' },
  { name: 'Cobalt', icon: '◉', hs: 'HS 810520', producers: 'DRC · Russia · Australia', slug: 'cobalt', accent: 'text-violet-600 bg-violet-50 border-violet-200' },
  { name: 'Nickel', icon: '○', hs: 'HS 750110', producers: 'Indonesia · Philippines · Russia', slug: 'nickel', accent: 'text-emerald-600 bg-emerald-50 border-emerald-200' },
];

const KG = [
  { label: 'Docs indexed', value: '1,661', sub: 'chunks (500 tok)' },
  { label: 'OpenIE triples', value: '4,368', sub: 'gpt-4o-mini extracted' },
  { label: 'KG entities', value: '3,132', sub: 'unique after norm.' },
  { label: 'KG edges', value: '31,916', sub: 'igraph directed' },
];

const METHODS = [
  {
    level: 'L1',
    title: 'Association',
    subtitle: 'Seeing',
    desc: 'Observe P(Y | X=x). Correlations from BACI bilateral flows — no causal claim.',
    color: 'bg-zinc-50 border-zinc-200 text-zinc-600',
    badge: 'text-zinc-500 bg-zinc-100',
  },
  {
    level: 'L2',
    title: 'Intervention',
    subtitle: 'Doing',
    desc: 'P(Y | do(X=x)). Graph surgery on the SCM + 2SLS identified parameters (η_D, α_P, τ_K).',
    color: 'bg-indigo-50 border-indigo-200 text-indigo-700',
    badge: 'text-indigo-600 bg-indigo-100',
  },
  {
    level: 'L3',
    title: 'Counterfactual',
    subtitle: 'Imagining',
    desc: 'P(Y_x | X=x′). Abduction-action-prediction with fixed noise seed (twin-network SDE replay).',
    color: 'bg-purple-50 border-purple-200 text-purple-700',
    badge: 'text-purple-600 bg-purple-100',
  },
];

export default function DashboardPage() {
  return (
    <div className="min-h-screen bg-zinc-50">
      <div className="max-w-4xl mx-auto px-8 py-8">

        {/* Hero */}
        <div className="mb-10">
          <div className="inline-flex items-center gap-2 text-[11px] font-semibold text-indigo-600 uppercase tracking-widest mb-3">
            <div className="h-1.5 w-1.5 rounded-full bg-indigo-500" />
            Supply Chain Intelligence
          </div>
          <h1 className="text-3xl font-bold text-zinc-900 leading-tight mb-3">
            Critical Minerals<br />
            <span className="text-indigo-600">Causal Engine</span>
          </h1>
          <p className="text-base text-zinc-500 max-w-xl leading-relaxed">
            Identify trade route circumvention, estimate policy effects, and answer research
            questions — using Pearl's Ladder of Causation over CEPII BACI, USGS MCS, and a
            HippoRAG knowledge graph.
          </p>
          <div className="flex gap-3 mt-5">
            <Link
              href="/transshipment"
              className="inline-flex items-center gap-2 px-5 py-2.5 bg-indigo-600 text-white text-sm font-semibold rounded-xl hover:bg-indigo-700 transition-colors shadow-sm"
            >
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 7h12m0 0l-4-4m4 4l-4 4m0 6H4m0 0l4 4m-4-4l4-4" />
              </svg>
              Analyze routes
            </Link>
            <Link
              href="/query"
              className="inline-flex items-center gap-2 px-5 py-2.5 bg-white text-zinc-700 text-sm font-semibold rounded-xl border border-zinc-200 hover:border-zinc-300 hover:bg-zinc-50 transition-colors"
            >
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" />
              </svg>
              Query knowledge base
            </Link>
          </div>
        </div>

        {/* Getting started */}
        <section className="mb-10">
          <h2 className="text-[11px] font-semibold text-zinc-400 uppercase tracking-wider mb-4">
            Getting started
          </h2>
          <div className="grid grid-cols-2 gap-3">
            {STEPS.map(({ n, title, code, desc }) => (
              <div
                key={n}
                className="bg-white rounded-xl border border-zinc-200 p-4 flex gap-3"
              >
                <div className="h-6 w-6 rounded-full bg-indigo-100 text-indigo-700 text-xs font-bold flex items-center justify-center shrink-0 mt-0.5">
                  {n}
                </div>
                <div>
                  <p className="text-sm font-semibold text-zinc-800 mb-1">{title}</p>
                  <p className="text-xs text-zinc-500 leading-relaxed mb-1.5">{desc}</p>
                  {code && (
                    <code className="text-[11px] font-mono bg-zinc-100 text-zinc-700 px-2 py-0.5 rounded">
                      {code}
                    </code>
                  )}
                </div>
              </div>
            ))}
          </div>
        </section>

        {/* Pearl ladder */}
        <section className="mb-10">
          <h2 className="text-[11px] font-semibold text-zinc-400 uppercase tracking-wider mb-4">
            Causal framework — Pearl's Ladder
          </h2>
          <div className="grid grid-cols-3 gap-3">
            {METHODS.map(({ level, title, subtitle, desc, color, badge }) => (
              <div key={level} className={`rounded-xl border p-4 ${color}`}>
                <div className="flex items-center gap-2 mb-2">
                  <span className={`text-xs font-bold px-1.5 py-0.5 rounded ${badge}`}>
                    {level}
                  </span>
                  <div>
                    <p className="text-xs font-semibold leading-none">{title}</p>
                    <p className="text-[10px] opacity-60">{subtitle}</p>
                  </div>
                </div>
                <p className="text-xs leading-relaxed opacity-75">{desc}</p>
              </div>
            ))}
          </div>
        </section>

        {/* Two columns: KG + Commodities */}
        <div className="grid grid-cols-2 gap-6 mb-10">
          {/* KG stats */}
          <section>
            <h2 className="text-[11px] font-semibold text-zinc-400 uppercase tracking-wider mb-3">
              Knowledge graph
            </h2>
            <div className="grid grid-cols-2 gap-2">
              {KG.map(({ label, value, sub }) => (
                <div key={label} className="bg-white rounded-xl border border-zinc-200 p-3">
                  <p className="text-xl font-bold tabular-nums text-zinc-900">{value}</p>
                  <p className="text-xs font-medium text-zinc-600 mt-0.5">{label}</p>
                  <p className="text-[10px] text-zinc-400">{sub}</p>
                </div>
              ))}
            </div>
            <p className="text-[10px] text-zinc-400 mt-2 leading-relaxed">
              HippoRAG personalized PageRank · text-embedding-3-large 3072-dim · gpt-4o-mini OpenIE cache
            </p>
          </section>

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
                  className="flex items-center gap-3 bg-white rounded-xl border border-zinc-200 p-3 hover:border-indigo-300 hover:shadow-sm transition-all group"
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
                  <svg
                    className="w-4 h-4 text-zinc-300 group-hover:text-indigo-400 transition-colors shrink-0"
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                  </svg>
                </Link>
              ))}
            </div>
          </section>
        </div>

        {/* Data sources */}
        <section>
          <h2 className="text-[11px] font-semibold text-zinc-400 uppercase tracking-wider mb-3">
            Data sources
          </h2>
          <div className="bg-white rounded-xl border border-zinc-200 p-4">
            <div className="grid grid-cols-4 gap-4 text-xs">
              {[
                { name: 'CEPII BACI', detail: '1995–2024 · pre-reconciled bilateral flows · ISO 3166-1 normalized' },
                { name: 'USGS MCS', detail: '2020–2024 · 5 annual volumes · 17 commodities · reserves + production' },
                { name: 'IEA 2021', detail: 'Critical minerals supply chain risk · 20 chunks' },
                { name: 'DRC Reports', detail: '2 artisanal cobalt reports · 79 chunks' },
              ].map(({ name, detail }) => (
                <div key={name}>
                  <p className="font-semibold text-zinc-700 mb-0.5">{name}</p>
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
