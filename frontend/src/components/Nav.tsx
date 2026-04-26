'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { useEffect, useState } from 'react';

const links = [
  {
    href: '/',
    label: 'Dashboard',
    icon: (
      <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 12l2-2m0 0l7-7 7 7M5 10v10a1 1 0 001 1h3m10-11l2 2m-2-2v10a1 1 0 01-1 1h-3m-6 0a1 1 0 001-1v-4a1 1 0 011-1h2a1 1 0 011 1v4a1 1 0 001 1m-6 0h6" />
      </svg>
    ),
    desc: 'Overview & metrics',
  },
  {
    href: '/transshipment',
    label: 'Transshipment',
    icon: (
      <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 7h12m0 0l-4-4m4 4l-4 4m0 6H4m0 0l4 4m-4-4l4-4" />
      </svg>
    ),
    desc: 'Trace & detect routes',
  },
  {
    href: '/query',
    label: 'Knowledge Query',
    icon: (
      <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" />
      </svg>
    ),
    desc: 'Ask the knowledge base',
  },
  {
    href: '/counterfactual',
    label: 'Counterfactual',
    icon: (
      <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
      </svg>
    ),
    desc: 'Pearl L3 — what would have been',
  },
  {
    href: '/knowledge-graph',
    label: 'Knowledge Graph',
    icon: (
      <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2H6a2 2 0 01-2-2V6zM14 6a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2h-2a2 2 0 01-2-2V6zM4 16a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2H6a2 2 0 01-2-2v-2zM14 16a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2h-2a2 2 0 01-2-2v-2z" />
      </svg>
    ),
    desc: 'Entities & causal edges',
  },
  {
    href: '/scenario-builder',
    label: 'Scenario Builder',
    icon: (
      <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 6v6m0 0v6m0-6h6m-6 0H6" />
      </svg>
    ),
    desc: 'Custom scenario → KG render',
  },
  {
    href: '/shock-extractor',
    label: 'Shock Extractor',
    icon: (
      <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
      </svg>
    ),
    desc: 'Text → causal prediction',
  },
];

type BackendStatus = 'checking' | 'online' | 'offline';

export default function Nav() {
  const path = usePathname();
  const [status, setStatus] = useState<BackendStatus>('checking');

  useEffect(() => {
    const check = () => {
      fetch('/api/health')
        .then((r) => r.json())
        .then((d) => setStatus(d.status === 'healthy' ? 'online' : 'offline'))
        .catch(() => setStatus('offline'));
    };
    check();
    const id = setInterval(check, 30_000);
    return () => clearInterval(id);
  }, []);

  const statusColor =
    status === 'online'
      ? 'bg-emerald-400'
      : status === 'offline'
      ? 'bg-red-400'
      : 'bg-amber-300 animate-pulse';
  const statusLabel =
    status === 'online' ? 'Backend online' : status === 'offline' ? 'Backend offline' : 'Checking…';

  return (
    <nav className="w-58 min-h-screen bg-white border-r border-zinc-100 flex flex-col py-5 shrink-0 shadow-[1px_0_0_0_#f4f4f5]">
      {/* Logo */}
      <div className="px-4 mb-7">
        <div className="flex items-center gap-2.5 mb-1">
          <div className="h-7 w-7 rounded-lg bg-gradient-to-br from-indigo-500 to-violet-600 flex items-center justify-center shadow-sm">
            <span className="text-white text-[11px] font-bold tracking-tight">CE</span>
          </div>
          <div>
            <span className="text-sm font-bold text-zinc-900 leading-none block">Causal Engine</span>
            <span className="text-[10px] text-zinc-400 leading-none">Critical Minerals</span>
          </div>
        </div>
      </div>

      {/* Nav links */}
      <div className="flex flex-col gap-0.5 px-2 flex-1">
        {links.map(({ href, label, icon, desc }) => {
          const active = href === '/' ? path === '/' : path.startsWith(href);
          return (
            <Link
              key={href}
              href={href}
              className={`group relative flex items-center gap-3 px-3 py-2.5 rounded-lg transition-all ${
                active
                  ? 'bg-indigo-50 text-indigo-700'
                  : 'text-zinc-500 hover:text-zinc-900 hover:bg-zinc-50'
              }`}
            >
              {/* Active left-border accent */}
              {active && (
                <span className="absolute left-0 top-1.5 bottom-1.5 w-0.5 rounded-full bg-indigo-500" />
              )}
              <span className={`${active ? 'text-indigo-600' : 'text-zinc-400 group-hover:text-zinc-600'} transition-colors`}>
                {icon}
              </span>
              <div>
                <p className={`text-sm font-medium leading-none ${active ? 'text-indigo-700' : ''}`}>
                  {label}
                </p>
                <p className="text-[10px] text-zinc-400 mt-0.5">{desc}</p>
              </div>
            </Link>
          );
        })}
      </div>

      {/* Backend status */}
      <div className="px-4 pt-4 mt-2 border-t border-zinc-100">
        <div className="flex items-center gap-2">
          <div className={`h-2 w-2 rounded-full ${statusColor}`} />
          <span className="text-[11px] text-zinc-400">{statusLabel}</span>
        </div>
        <p className="text-[10px] text-zinc-300 mt-2 leading-relaxed">
          Pearl L1·L2·L3<br />
          CEPII BACI · USGS · IEA
        </p>
      </div>
    </nav>
  );
}
