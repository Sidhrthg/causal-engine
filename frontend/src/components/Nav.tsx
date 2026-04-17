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
    desc: 'Overview',
  },
  {
    href: '/transshipment',
    label: 'Transshipment',
    icon: (
      <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 7h12m0 0l-4-4m4 4l-4 4m0 6H4m0 0l4 4m-4-4l4-4" />
      </svg>
    ),
    desc: 'Trace routes',
  },
  {
    href: '/query',
    label: 'Knowledge Query',
    icon: (
      <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" />
      </svg>
    ),
    desc: 'Ask questions',
  },
  {
    href: '/counterfactual',
    label: 'Counterfactual',
    icon: (
      <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
      </svg>
    ),
    desc: 'What would have been',
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
    <nav className="w-56 min-h-screen bg-white border-r border-zinc-200 flex flex-col py-5 shrink-0">
      {/* Logo */}
      <div className="px-4 mb-6">
        <div className="flex items-center gap-2 mb-1">
          <div className="h-6 w-6 bg-indigo-600 rounded flex items-center justify-center">
            <span className="text-white text-xs font-bold">CE</span>
          </div>
          <span className="text-sm font-bold text-zinc-900">Causal Engine</span>
        </div>
        <p className="text-[10px] text-zinc-400 pl-8">Critical Minerals</p>
      </div>

      {/* Nav links */}
      <div className="flex flex-col gap-0.5 px-2 flex-1">
        {links.map(({ href, label, icon, desc }) => {
          const active = href === '/' ? path === '/' : path.startsWith(href);
          return (
            <Link
              key={href}
              href={href}
              className={`group flex items-center gap-3 px-3 py-2.5 rounded-lg transition-all ${
                active
                  ? 'bg-indigo-50 text-indigo-700'
                  : 'text-zinc-500 hover:text-zinc-900 hover:bg-zinc-50'
              }`}
            >
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
