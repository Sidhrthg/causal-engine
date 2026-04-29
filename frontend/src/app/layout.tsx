import type { Metadata } from 'next';
import { Inter } from 'next/font/google';
import './globals.css';
import Nav from '@/components/Nav';

const inter = Inter({ subsets: ['latin'] });

export const metadata: Metadata = {
  title: 'Causal Engine — Critical Minerals',
  description: 'Causal supply chain intelligence for critical minerals',
};

// Inline script to apply theme before React hydrates — prevents flash of
// wrong theme. Runs synchronously in the document head.
const themeInitScript = `
  (function () {
    try {
      var stored = localStorage.getItem('theme');
      if (stored === 'dark') document.documentElement.classList.add('dark');
    } catch (e) {}
  })();
`;

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className="bg-white dark:bg-zinc-950">
      <head>
        <script dangerouslySetInnerHTML={{ __html: themeInitScript }} />
      </head>
      <body className={`${inter.className} bg-zinc-50 dark:bg-zinc-950 text-zinc-900 dark:text-zinc-100`}>
        <div className="flex min-h-screen">
          <Nav />
          <main className="flex-1 overflow-auto">{children}</main>
        </div>
      </body>
    </html>
  );
}
