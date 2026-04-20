'use client';

import { useState, useRef, useEffect, useCallback } from 'react';
import { queryKnowledgeBase } from '@/lib/api';
import type { ChatMessage, SourceChunk } from '@/lib/types';

// ─── Example questions by category ───────────────────────────────────────────
const EXAMPLES = [
  { cat: 'Graphite', q: 'How does China control graphite anode processing?' },
  { cat: 'Graphite', q: 'What caused the 2012 graphite price spike?' },
  { cat: 'Lithium', q: 'Why is Australia the largest lithium producer?' },
  { cat: 'Lithium', q: 'What are the main substitutes for lithium in batteries?' },
  { cat: 'Cobalt', q: 'Explain DRC artisanal cobalt mining risks.' },
  { cat: 'Cobalt', q: 'How much cobalt does the DRC produce?' },
  { cat: 'Policy', q: 'What is the FEOC rule and how does it affect South Korean anode processors?' },
  { cat: 'Policy', q: 'What export restrictions has China imposed on critical minerals?' },
];

const CATEGORIES = ['All', 'Graphite', 'Lithium', 'Cobalt', 'Policy'];

// ─── Source panel ─────────────────────────────────────────────────────────────
function SourcesPanel({ sources }: { sources: SourceChunk[] }) {
  const [open, setOpen] = useState(false);
  if (!sources.length) return null;

  return (
    <div className="mt-3 pt-3 border-t border-zinc-100">
      <button
        onClick={() => setOpen(!open)}
        className="flex items-center gap-1.5 text-[11px] text-zinc-400 hover:text-zinc-600 transition-colors"
      >
        <svg
          className={`w-3 h-3 transition-transform ${open ? 'rotate-90' : ''}`}
          fill="none"
          stroke="currentColor"
          viewBox="0 0 24 24"
        >
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
        </svg>
        {sources.length} source{sources.length !== 1 ? 's' : ''} retrieved
      </button>

      {open && (
        <div className="mt-2.5 flex flex-col gap-2">
          {sources.map((s, i) => (
            <div key={i} className="rounded-lg border border-zinc-100 bg-zinc-50/80 p-3">
              <div className="flex items-center justify-between mb-1.5">
                <span className="text-[10px] font-mono text-zinc-400 truncate max-w-[220px]">
                  {s.source || 'unknown source'}
                </span>
                <div className="flex items-center gap-1 ml-2 shrink-0">
                  <div
                    className="h-1.5 rounded-full bg-indigo-400"
                    style={{ width: `${Math.max(s.similarity * 32, 4)}px` }}
                  />
                  <span className="text-[10px] text-zinc-400">
                    {(s.similarity * 100).toFixed(0)}%
                  </span>
                </div>
              </div>
              <p className="text-xs text-zinc-600 leading-relaxed line-clamp-3">{s.text}</p>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

// ─── Copy button ──────────────────────────────────────────────────────────────
function CopyButton({ text }: { text: string }) {
  const [copied, setCopied] = useState(false);
  const copy = () => {
    navigator.clipboard.writeText(text).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    });
  };
  return (
    <button
      onClick={copy}
      className="p-1 text-zinc-300 hover:text-zinc-500 transition-colors rounded"
      title="Copy"
    >
      {copied ? (
        <svg className="w-3.5 h-3.5 text-emerald-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
        </svg>
      ) : (
        <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z" />
        </svg>
      )}
    </button>
  );
}

// ─── Message renderer (basic markdown-ish) ────────────────────────────────────
function MessageText({ text }: { text: string }) {
  // Render **bold** and line breaks
  const parts = text.split(/(\*\*[^*]+\*\*)/g);
  return (
    <p className="text-sm leading-relaxed whitespace-pre-wrap">
      {parts.map((part, i) =>
        part.startsWith('**') && part.endsWith('**') ? (
          <strong key={i} className="font-semibold">
            {part.slice(2, -2)}
          </strong>
        ) : (
          <span key={i}>{part}</span>
        )
      )}
    </p>
  );
}

// ─── Main page ────────────────────────────────────────────────────────────────
export default function QueryPage() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState('');
  const [commodity, setCommodity] = useState('');
  const [loading, setLoading] = useState(false);
  const [category, setCategory] = useState('All');
  const bottomRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, loading]);

  const sendMessage = useCallback(
    async (question: string) => {
      const q = question.trim();
      if (!q || loading) return;

      setMessages((prev) => [
        ...prev,
        { role: 'user', content: q, timestamp: new Date() },
      ]);
      setInput('');
      setLoading(true);

      try {
        const resp = await queryKnowledgeBase({
          question: q,
          commodity: commodity || undefined,
        });
        setMessages((prev) => [
          ...prev,
          {
            role: 'assistant',
            content: resp.answer || 'No answer returned.',
            sources: resp.sources,
            timestamp: new Date(),
          },
        ]);
      } catch (err) {
        setMessages((prev) => [
          ...prev,
          {
            role: 'assistant',
            content: `Could not reach the backend: ${err instanceof Error ? err.message : 'Unknown error'}.\n\nMake sure the FastAPI server is running:\n  uvicorn api:app --reload`,
            timestamp: new Date(),
          },
        ]);
      } finally {
        setLoading(false);
        setTimeout(() => inputRef.current?.focus(), 100);
      }
    },
    [loading, commodity]
  );

  const filteredExamples =
    category === 'All' ? EXAMPLES : EXAMPLES.filter((e) => e.cat === category);

  return (
    <div className="flex flex-col h-screen bg-zinc-50">
      {/* Header */}
      <div className="border-b border-zinc-200 bg-white px-6 py-4 shrink-0">
        <div className="max-w-3xl flex items-center justify-between">
          <div>
            <div className="flex items-center gap-2 mb-0.5">
              <p className="text-[10px] font-semibold text-indigo-600 uppercase tracking-widest">
                HippoRAG · USGS MCS · CEPII · IEA
              </p>
            </div>
            <h1 className="text-lg font-bold text-zinc-900">Knowledge Query</h1>
          </div>
          <div className="flex items-center gap-3">
            <div className="flex items-center gap-2">
              <label className="text-xs text-zinc-400">Commodity</label>
              <select
                value={commodity}
                onChange={(e) => setCommodity(e.target.value)}
                className="text-sm border border-zinc-200 rounded-lg px-2.5 py-1.5 bg-white focus:outline-none focus:ring-2 focus:ring-indigo-500"
              >
                <option value="">All</option>
                {['graphite', 'lithium', 'cobalt', 'nickel', 'copper'].map((c) => (
                  <option key={c} value={c}>{c}</option>
                ))}
              </select>
            </div>
            {messages.length > 0 && (
              <button
                onClick={() => setMessages([])}
                className="text-xs text-zinc-400 hover:text-zinc-600 transition-colors"
              >
                Clear chat
              </button>
            )}
          </div>
        </div>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto px-6 py-6">
        <div className="max-w-3xl mx-auto">
          {/* Empty state */}
          {messages.length === 0 && !loading && (
            <div className="py-8">
              <div className="text-center mb-6">
                <div className="h-12 w-12 bg-indigo-50 rounded-xl flex items-center justify-center mx-auto mb-3">
                  <svg className="w-6 h-6 text-indigo-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
                  </svg>
                </div>
                <p className="text-sm font-medium text-zinc-700 mb-1">Ask anything about critical minerals</p>
                <p className="text-xs text-zinc-400">
                  1,661 chunks · 4,368 triples · 3,132 entities · text-embedding-3-large
                </p>
              </div>

              {/* Category filter */}
              <div className="flex gap-1.5 justify-center mb-4">
                {CATEGORIES.map((cat) => (
                  <button
                    key={cat}
                    onClick={() => setCategory(cat)}
                    className={`text-xs px-3 py-1 rounded-full border transition-all ${
                      category === cat
                        ? 'bg-indigo-600 border-indigo-600 text-white'
                        : 'bg-white border-zinc-200 text-zinc-500 hover:border-zinc-300'
                    }`}
                  >
                    {cat}
                  </button>
                ))}
              </div>

              {/* Example questions */}
              <div className="grid grid-cols-2 gap-2">
                {filteredExamples.map(({ q, cat }) => (
                  <button
                    key={q}
                    onClick={() => sendMessage(q)}
                    className="text-left p-3 bg-white rounded-lg border border-zinc-200 hover:border-indigo-300 hover:shadow-sm transition-all group"
                  >
                    <span className="text-[10px] font-semibold text-indigo-400 uppercase tracking-wider block mb-1">
                      {cat}
                    </span>
                    <span className="text-xs text-zinc-700 group-hover:text-zinc-900 leading-relaxed">
                      {q}
                    </span>
                  </button>
                ))}
              </div>
            </div>
          )}

          {/* Messages */}
          <div className="flex flex-col gap-5">
            {messages.map((msg, i) => (
              <div
                key={i}
                className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
              >
                {msg.role === 'user' ? (
                  <div className="max-w-[75%]">
                    <div className="bg-indigo-600 text-white text-sm rounded-2xl rounded-tr-sm px-4 py-2.5 leading-relaxed">
                      {msg.content}
                    </div>
                    <p className="text-[10px] text-zinc-300 mt-1 text-right px-1">
                      {msg.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                    </p>
                  </div>
                ) : (
                  <div className="max-w-[85%]">
                    <div className="bg-white border border-zinc-200 rounded-2xl rounded-tl-sm px-4 py-4 shadow-sm">
                      <div className="flex items-start justify-between gap-2">
                        <MessageText text={msg.content} />
                        <CopyButton text={msg.content} />
                      </div>
                      {msg.sources && <SourcesPanel sources={msg.sources} />}
                    </div>
                    <p className="text-[10px] text-zinc-300 mt-1 px-1">
                      {msg.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                    </p>
                  </div>
                )}
              </div>
            ))}

            {/* Loading */}
            {loading && (
              <div className="flex justify-start">
                <div className="bg-white border border-zinc-200 rounded-2xl rounded-tl-sm px-4 py-3.5 shadow-sm">
                  <div className="flex gap-1.5 items-center">
                    {[0, 1, 2].map((i) => (
                      <div
                        key={i}
                        className="h-2 w-2 bg-indigo-300 rounded-full animate-bounce"
                        style={{ animationDelay: `${i * 120}ms` }}
                      />
                    ))}
                  </div>
                </div>
              </div>
            )}
          </div>

          <div ref={bottomRef} />
        </div>
      </div>

      {/* Input bar */}
      <div className="border-t border-zinc-200 bg-white px-6 py-4 shrink-0">
        <form
          onSubmit={(e) => {
            e.preventDefault();
            sendMessage(input);
          }}
          className="max-w-3xl mx-auto flex gap-2 items-end"
        >
          <textarea
            ref={inputRef}
            rows={1}
            value={input}
            onChange={(e) => {
              setInput(e.target.value);
              e.target.style.height = 'auto';
              e.target.style.height = `${Math.min(e.target.scrollHeight, 120)}px`;
            }}
            onKeyDown={(e) => {
              if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendMessage(input);
              }
            }}
            placeholder="Ask about mineral supply chains, price drivers, trade patterns… (Enter to send)"
            disabled={loading}
            className="flex-1 text-sm border border-zinc-200 rounded-xl px-4 py-2.5 resize-none focus:outline-none focus:ring-2 focus:ring-indigo-500 disabled:opacity-50 leading-relaxed"
            style={{ minHeight: '42px', maxHeight: '120px' }}
          />
          <button
            type="submit"
            disabled={loading || !input.trim()}
            className="flex items-center gap-1.5 px-4 py-2.5 bg-indigo-600 text-white text-sm font-semibold rounded-xl hover:bg-indigo-700 disabled:opacity-40 transition-colors shadow-sm shrink-0"
          >
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8" />
            </svg>
            Send
          </button>
        </form>
        <p className="text-center text-[10px] text-zinc-300 mt-2">
          Shift+Enter for new line · Enter to send
        </p>
      </div>
    </div>
  );
}
