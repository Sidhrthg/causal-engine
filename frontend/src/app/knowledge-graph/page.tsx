'use client';

import { useEffect, useMemo, useRef, useState, useCallback } from 'react';
import HowToUse from '@/components/HowToUse';
import { getKnowledgeGraph } from '@/lib/api';
import type { KGEntity, KGRelationship } from '@/lib/types';

const COMMODITIES = ['', 'graphite', 'lithium', 'cobalt', 'nickel', 'copper', 'soybeans'];

const TYPE_COLORS: Record<string, string> = {
  commodity:          '#6366f1',  // indigo
  country:            '#10b981',  // emerald
  policy:             '#f59e0b',  // amber
  technology:         '#3b82f6',  // blue
  industry:           '#8b5cf6',  // violet
  economic_indicator: '#ef4444',  // red
  region:             '#06b6d4',  // cyan
  event:              '#f97316',  // orange
  risk_factor:        '#ec4899',  // pink
};

const REL_COLORS: Record<string, string> = {
  produces:     '#10b981',
  exports_to:   '#6366f1',
  regulates:    '#f59e0b',
  disrupts:     '#ef4444',
  causes:       '#f97316',
  depends_on:   '#8b5cf6',
  mitigates:    '#06b6d4',
  consumes:     '#3b82f6',
  enables:      '#ec4899',
};

interface Node extends KGEntity {
  x: number;
  y: number;
  vx: number;
  vy: number;
  degree: number;
}

interface Edge {
  source: Node;
  target: Node;
  relation_type: string;
}

function buildGraph(
  entities: KGEntity[],
  relationships: KGRelationship[],
  width: number,
  height: number,
): { nodes: Node[]; edges: Edge[] } {
  const degMap: Record<string, number> = {};
  for (const r of relationships) {
    degMap[r.source_id] = (degMap[r.source_id] ?? 0) + 1;
    degMap[r.target_id] = (degMap[r.target_id] ?? 0) + 1;
  }

  // Group by type for initial layout (concentric rings)
  const typeOrder = ['commodity','country','policy','industry','technology','economic_indicator','region','event','risk_factor'];
  const byType: Record<string, KGEntity[]> = {};
  for (const e of entities) {
    (byType[e.entity_type] ??= []).push(e);
  }

  const nodeMap: Record<string, Node> = {};
  const cx = width / 2, cy = height / 2;
  let ringIdx = 0;

  for (const type of typeOrder) {
    const group = byType[type] ?? [];
    if (!group.length) continue;
    const radius = ringIdx === 0 ? 0 : 80 + ringIdx * 95;
    group.forEach((e, i) => {
      const angle = (2 * Math.PI * i) / group.length - Math.PI / 2;
      const jitter = (Math.random() - 0.5) * 20;
      nodeMap[e.id] = {
        ...e,
        x: group.length === 1 ? cx : cx + (radius + jitter) * Math.cos(angle),
        y: group.length === 1 ? cy : cy + (radius + jitter) * Math.sin(angle),
        vx: 0,
        vy: 0,
        degree: degMap[e.id] ?? 0,
      };
    });
    ringIdx++;
  }

  const edges: Edge[] = [];
  for (const r of relationships) {
    const s = nodeMap[r.source_id];
    const t = nodeMap[r.target_id];
    if (s && t) edges.push({ source: s, target: t, relation_type: r.relation_type });
  }

  return { nodes: Object.values(nodeMap), edges };
}

function runForce(nodes: Node[], edges: Edge[], iterations = 80): void {
  const k = 60;
  for (let iter = 0; iter < iterations; iter++) {
    const alpha = 1 - iter / iterations;

    // Repulsion
    for (let i = 0; i < nodes.length; i++) {
      for (let j = i + 1; j < nodes.length; j++) {
        const dx = nodes[j].x - nodes[i].x;
        const dy = nodes[j].y - nodes[i].y;
        const d = Math.sqrt(dx * dx + dy * dy) || 1;
        const f = (k * k) / d * alpha * 0.5;
        nodes[i].vx -= (dx / d) * f;
        nodes[i].vy -= (dy / d) * f;
        nodes[j].vx += (dx / d) * f;
        nodes[j].vy += (dy / d) * f;
      }
    }

    // Attraction along edges
    for (const e of edges) {
      const dx = e.target.x - e.source.x;
      const dy = e.target.y - e.source.y;
      const d = Math.sqrt(dx * dx + dy * dy) || 1;
      const f = (d / k) * alpha;
      e.source.vx += (dx / d) * f;
      e.source.vy += (dy / d) * f;
      e.target.vx -= (dx / d) * f;
      e.target.vy -= (dy / d) * f;
    }

    for (const n of nodes) {
      n.x += Math.max(-10, Math.min(10, n.vx));
      n.y += Math.max(-10, Math.min(10, n.vy));
      n.vx *= 0.7;
      n.vy *= 0.7;
    }
  }
}

export default function KnowledgeGraphPage() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [commodity, setCommodity] = useState('');
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selected, setSelected] = useState<Node | null>(null);
  const [stats, setStats] = useState({ nodes: 0, edges: 0 });
  const [relFilter, setRelFilter] = useState<string>('');
  const [search, setSearch] = useState('');
  const [focusMode, setFocusMode] = useState(false);
  const [hoveredEdge, setHoveredEdge] = useState<Edge | null>(null);
  const [tooltip, setTooltip] = useState<{ x: number; y: number; text: string } | null>(null);
  const graphRef = useRef<{ nodes: Node[]; edges: Edge[] } | null>(null);
  const transformRef = useRef({ scale: 1, x: 0, y: 0 });
  const dragRef = useRef<{ active: boolean; lastX: number; lastY: number }>({ active: false, lastX: 0, lastY: 0 });

  // Pre-compute search matches and 1-hop neighbour set for the selected node
  const searchMatches = useMemo(() => {
    const q = search.trim().toLowerCase();
    if (!q || !graphRef.current) return new Set<string>();
    return new Set(graphRef.current.nodes
      .filter((n) => n.id.toLowerCase().includes(q) ||
                    n.aliases?.some((a) => a.toLowerCase().includes(q)))
      .map((n) => n.id));
  }, [search]);

  const neighborhood = useMemo(() => {
    if (!selected || !focusMode || !graphRef.current) return null;
    const neigh = new Set<string>([selected.id]);
    for (const e of graphRef.current.edges) {
      if (e.source.id === selected.id) neigh.add(e.target.id);
      else if (e.target.id === selected.id) neigh.add(e.source.id);
    }
    return neigh;
  }, [selected, focusMode]);

  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    const graph = graphRef.current;
    if (!canvas || !graph) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const { scale, x: tx, y: ty } = transformRef.current;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.save();
    ctx.translate(tx, ty);
    ctx.scale(scale, scale);

    const isInFocus = (id: string) => !neighborhood || neighborhood.has(id);
    const isSearchMatch = (id: string) => searchMatches.size === 0 || searchMatches.has(id);

    // Draw edges
    for (const e of graph.edges) {
      if (relFilter && e.relation_type !== relFilter) continue;
      const inFocus = isInFocus(e.source.id) && isInFocus(e.target.id);
      const isHovered = hoveredEdge === e;
      const baseColor = REL_COLORS[e.relation_type] ?? '#94a3b8';
      ctx.beginPath();
      ctx.moveTo(e.source.x, e.source.y);
      ctx.lineTo(e.target.x, e.target.y);
      // Hovered or selected-touching edges: full opacity + thicker
      if (isHovered) {
        ctx.strokeStyle = baseColor;
        ctx.lineWidth = 2.4;
      } else if (selected && (e.source.id === selected.id || e.target.id === selected.id)) {
        ctx.strokeStyle = baseColor + 'cc';
        ctx.lineWidth = 1.6;
      } else if (!inFocus) {
        ctx.strokeStyle = baseColor + '14'; // very dim
        ctx.lineWidth = 0.6;
      } else {
        ctx.strokeStyle = baseColor + '66';
        ctx.lineWidth = 0.8;
      }
      ctx.stroke();
    }

    // Draw nodes
    for (const n of graph.nodes) {
      const r = 4 + Math.sqrt(n.degree) * 2;
      const color = TYPE_COLORS[n.entity_type] ?? '#94a3b8';
      const inFocus = isInFocus(n.id);
      const matched = isSearchMatch(n.id);
      const dimmed = !inFocus || (search && !matched);
      const highlighted = search && matched;

      ctx.globalAlpha = dimmed ? 0.18 : 1.0;
      ctx.beginPath();
      ctx.arc(n.x, n.y, highlighted ? r + 2 : r, 0, 2 * Math.PI);
      ctx.fillStyle = n === selected ? '#fff' : color;
      ctx.strokeStyle = color;
      ctx.lineWidth = n === selected ? 2.5 : highlighted ? 2 : 1;
      ctx.fill();
      ctx.stroke();
      ctx.globalAlpha = 1.0;

      // Label for high-degree, selected, or search-matched nodes
      if (n.degree > 6 || n === selected || highlighted) {
        ctx.fillStyle = highlighted ? '#dc2626' : '#1e293b';
        ctx.font = `${(n === selected || highlighted) ? 'bold ' : ''}${10 / scale + 2}px system-ui`;
        ctx.fillText(n.id.replace(/_/g, ' '), n.x + r + 2, n.y + 4);
      }
    }

    ctx.restore();
  }, [selected, relFilter, neighborhood, searchMatches, search, hoveredEdge]);

  const loadGraph = useCallback(async () => {
    setLoading(true);
    setError(null);
    setSelected(null);
    try {
      const data = await getKnowledgeGraph(commodity || undefined);
      const W = canvasRef.current?.width ?? 900;
      const H = canvasRef.current?.height ?? 650;
      const { nodes, edges } = buildGraph(data.entities, data.relationships ?? [], W, H);
      runForce(nodes, edges);
      graphRef.current = { nodes, edges };
      transformRef.current = { scale: 1, x: 0, y: 0 };
      setStats({ nodes: nodes.length, edges: edges.length });
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to load');
    } finally {
      setLoading(false);
    }
  }, [commodity]);

  useEffect(() => { loadGraph(); }, [loadGraph]);
  useEffect(() => { if (!loading) draw(); }, [loading, draw]);

  // Canvas click → select node
  const handleClick = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    const graph = graphRef.current;
    if (!graph) return;
    const rect = (e.target as HTMLCanvasElement).getBoundingClientRect();
    const { scale, x: tx, y: ty } = transformRef.current;
    const mx = (e.clientX - rect.left - tx) / scale;
    const my = (e.clientY - rect.top - ty) / scale;
    let closest: Node | null = null;
    let minD = 20;
    for (const n of graph.nodes) {
      const d = Math.hypot(n.x - mx, n.y - my);
      if (d < minD) { minD = d; closest = n; }
    }
    setSelected(closest);
  }, []);

  // Pan
  const handleMouseDown = (e: React.MouseEvent) => {
    dragRef.current = { active: true, lastX: e.clientX, lastY: e.clientY };
  };
  const handleMouseMove = (e: React.MouseEvent) => {
    // Pan branch
    if (dragRef.current.active) {
      transformRef.current.x += e.clientX - dragRef.current.lastX;
      transformRef.current.y += e.clientY - dragRef.current.lastY;
      dragRef.current.lastX = e.clientX;
      dragRef.current.lastY = e.clientY;
      draw();
      return;
    }
    // Hover branch — find nearest edge for tooltip
    const graph = graphRef.current;
    if (!graph) return;
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const { scale, x: tx, y: ty } = transformRef.current;
    const mx = (e.clientX - rect.left - tx) / scale;
    const my = (e.clientY - rect.top - ty) / scale;

    // Distance from point to line segment
    const distToSeg = (px: number, py: number, x1: number, y1: number, x2: number, y2: number) => {
      const dx = x2 - x1, dy = y2 - y1;
      const len2 = dx * dx + dy * dy;
      if (len2 === 0) return Math.hypot(px - x1, py - y1);
      let t = ((px - x1) * dx + (py - y1) * dy) / len2;
      t = Math.max(0, Math.min(1, t));
      return Math.hypot(px - (x1 + t * dx), py - (y1 + t * dy));
    };

    let closestEdge: Edge | null = null;
    let minD = 6 / scale;
    for (const ed of graph.edges) {
      if (relFilter && ed.relation_type !== relFilter) continue;
      const d = distToSeg(mx, my, ed.source.x, ed.source.y, ed.target.x, ed.target.y);
      if (d < minD) { minD = d; closestEdge = ed; }
    }

    if (closestEdge !== hoveredEdge) {
      setHoveredEdge(closestEdge);
      if (closestEdge) {
        setTooltip({
          x: e.clientX - rect.left,
          y: e.clientY - rect.top,
          text: `${closestEdge.source.id.replace(/_/g, ' ')} → ${closestEdge.target.id.replace(/_/g, ' ')}: ${closestEdge.relation_type.replace(/_/g, ' ')}`,
        });
      } else {
        setTooltip(null);
      }
      draw();
    } else if (closestEdge && tooltip) {
      // Just move the tooltip with the cursor
      setTooltip({ x: e.clientX - rect.left, y: e.clientY - rect.top, text: tooltip.text });
    }
  };
  const handleMouseUp = () => { dragRef.current.active = false; };

  // Zoom
  const handleWheel = (e: React.WheelEvent) => {
    e.preventDefault();
    const factor = e.deltaY < 0 ? 1.1 : 0.9;
    transformRef.current.scale = Math.max(0.2, Math.min(5, transformRef.current.scale * factor));
    draw();
  };

  const relTypes = Array.from(new Set(graphRef.current?.edges.map((e) => e.relation_type) ?? [])).sort();

  return (
    <div className="flex flex-col h-screen bg-zinc-50">
      {/* Header */}
      <div className="border-b border-zinc-200 dark:border-zinc-800 bg-white dark:bg-zinc-900 px-6 py-3 shrink-0">
        <div className="flex items-center justify-between">
          <div>
            <p className="text-[10px] font-semibold text-indigo-600 uppercase tracking-widest mb-0.5">
              Option A · Causal Knowledge Graph
            </p>
            <h1 className="text-lg font-bold text-zinc-900">
              Knowledge Graph
              <span className="ml-2 text-sm font-normal text-zinc-400">
                {stats.nodes} nodes · {stats.edges} edges
              </span>
            </h1>
          </div>
          <div className="flex items-center gap-2">
            {/* Search box */}
            <div className="relative">
              <input
                type="text"
                placeholder="Search nodes…"
                value={search}
                onChange={(e) => setSearch(e.target.value)}
                className="text-sm border border-zinc-200 dark:border-zinc-800 rounded-lg pl-8 pr-2 py-1.5 bg-white dark:bg-zinc-900 focus:outline-none focus:ring-2 focus:ring-indigo-500 w-44"
              />
              <svg className="w-3.5 h-3.5 absolute left-2.5 top-2.5 text-zinc-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-4.35-4.35M11 18a7 7 0 100-14 7 7 0 000 14z" />
              </svg>
              {search && (
                <button
                  onClick={() => setSearch('')}
                  className="absolute right-2 top-1.5 text-zinc-400 hover:text-zinc-600 text-sm"
                >
                  ×
                </button>
              )}
            </div>
            <select
              value={commodity}
              onChange={(e) => setCommodity(e.target.value)}
              className="text-sm border border-zinc-200 dark:border-zinc-800 rounded-lg px-2.5 py-1.5 bg-white dark:bg-zinc-900 focus:outline-none focus:ring-2 focus:ring-indigo-500"
            >
              {COMMODITIES.map((c) => <option key={c} value={c}>{c || 'All commodities'}</option>)}
            </select>
            <select
              value={relFilter}
              onChange={(e) => { setRelFilter(e.target.value); draw(); }}
              className="text-sm border border-zinc-200 dark:border-zinc-800 rounded-lg px-2.5 py-1.5 bg-white dark:bg-zinc-900 focus:outline-none focus:ring-2 focus:ring-indigo-500"
            >
              <option value="">All relations</option>
              {relTypes.map((r) => <option key={r} value={r}>{r}</option>)}
            </select>
            <button
              onClick={() => setFocusMode(!focusMode)}
              disabled={!selected}
              title={selected ? `Toggle focus — show only ${selected.id.replace(/_/g, ' ')} + neighbours` : 'Click a node first to enable focus mode'}
              className={`text-xs px-3 py-1.5 border rounded-lg transition-colors ${
                focusMode && selected
                  ? 'bg-indigo-600 border-indigo-600 text-white'
                  : 'border-zinc-200 dark:border-zinc-800 text-zinc-600 dark:text-zinc-400 hover:bg-zinc-50 dark:hover:bg-zinc-800 disabled:opacity-40 disabled:cursor-not-allowed'
              }`}
            >
              Focus
            </button>
            <button
              onClick={loadGraph}
              className="text-xs px-3 py-1.5 border border-zinc-200 dark:border-zinc-800 rounded-lg hover:bg-zinc-50 dark:hover:bg-zinc-800 text-zinc-600 dark:text-zinc-400"
            >
              Reset
            </button>
          </div>
        </div>
      </div>

      <div className="flex flex-1 overflow-hidden">
        {/* Canvas */}
        <div className="flex-1 relative">
          {loading && (
            <div className="absolute inset-0 flex items-center justify-center bg-zinc-50 dark:bg-zinc-950/80 z-10">
              <div className="text-sm text-zinc-500">Building graph…</div>
            </div>
          )}
          {error && (
            <div className="absolute inset-0 flex items-center justify-center bg-zinc-50 dark:bg-zinc-950/80 z-10">
              <div className="text-sm text-red-500">{error}</div>
            </div>
          )}
          <canvas
            ref={canvasRef}
            width={1000}
            height={700}
            className="w-full h-full cursor-grab active:cursor-grabbing"
            onClick={handleClick}
            onMouseDown={handleMouseDown}
            onMouseMove={handleMouseMove}
            onMouseUp={handleMouseUp}
            onMouseLeave={() => { handleMouseUp(); setHoveredEdge(null); setTooltip(null); }}
            onWheel={handleWheel}
          />
          {tooltip && (
            <div
              className="absolute pointer-events-none px-2 py-1 text-[11px] bg-zinc-900 dark:bg-zinc-100 text-white dark:text-zinc-900 rounded shadow-lg z-20 whitespace-nowrap"
              style={{ left: tooltip.x + 12, top: tooltip.y + 12 }}
            >
              {tooltip.text}
            </div>
          )}
          <p className="absolute bottom-3 left-4 text-[10px] text-zinc-300 dark:text-zinc-600">
            Scroll to zoom · drag to pan · click node to inspect · hover edge for relation · search/focus in header
          </p>
        </div>

        {/* Sidebar */}
        <div className="w-60 border-l border-zinc-200 dark:border-zinc-800 bg-white dark:bg-zinc-900 p-4 flex flex-col gap-4 overflow-y-auto shrink-0">
          <HowToUse
            id="knowledge-graph"
            steps={[
              <>Use the <strong>commodity dropdown</strong> in the header to focus the graph (or pick All).</>,
              <><strong>Scroll</strong> to zoom, <strong>drag</strong> to pan, <strong>click</strong> any node to inspect its type, degree, and properties.</>,
              <>Filter edges by relation type (produces, regulates, exports_to, etc.) to declutter the view.</>,
            ]}
            tip="Larger dots = higher-degree nodes (more connections). The legend shows which color maps to which entity type."
          />

          {/* Legend */}
          <div>
            <p className="text-[10px] font-semibold text-zinc-400 uppercase tracking-wider mb-2">Node Types</p>
            <div className="flex flex-col gap-1.5">
              {Object.entries(TYPE_COLORS).map(([type, color]) => (
                <div key={type} className="flex items-center gap-2">
                  <div className="h-3 w-3 rounded-full shrink-0" style={{ backgroundColor: color }} />
                  <span className="text-xs text-zinc-600">{type.replace(/_/g, ' ')}</span>
                </div>
              ))}
            </div>
          </div>

          <div>
            <p className="text-[10px] font-semibold text-zinc-400 uppercase tracking-wider mb-2">Edge Types</p>
            <div className="flex flex-col gap-1.5">
              {Object.entries(REL_COLORS).map(([rel, color]) => (
                <div key={rel} className="flex items-center gap-2">
                  <div className="h-0.5 w-5 shrink-0 rounded" style={{ backgroundColor: color }} />
                  <span className="text-xs text-zinc-600">{rel.replace(/_/g, ' ')}</span>
                </div>
              ))}
            </div>
          </div>

          {/* Selected node details */}
          {selected && (
            <div className="border-t border-zinc-100 dark:border-zinc-800 pt-4">
              <p className="text-[10px] font-semibold text-zinc-400 uppercase tracking-wider mb-2">Selected</p>
              <div className="bg-zinc-50 dark:bg-zinc-950 rounded-lg p-3">
                <p className="text-sm font-bold text-zinc-800 dark:text-zinc-200 mb-1">{selected.id.replace(/_/g, ' ')}</p>
                <div
                  className="text-[10px] font-semibold px-2 py-0.5 rounded-full inline-block mb-2 text-white"
                  style={{ backgroundColor: TYPE_COLORS[selected.entity_type] ?? '#94a3b8' }}
                >
                  {selected.entity_type}
                </div>
                <p className="text-[11px] text-zinc-500 mb-1">Degree: {selected.degree}</p>
                {selected.aliases && selected.aliases.length > 0 && (
                  <p className="text-[10px] text-zinc-400">
                    Aliases: {selected.aliases.join(', ')}
                  </p>
                )}
                {selected.properties && Object.keys(selected.properties).length > 0 && (
                  <div className="mt-2">
                    {Object.entries(selected.properties).map(([k, v]) => (
                      <div key={k} className="text-[10px] text-zinc-500">
                        <span className="font-medium">{k}:</span>{' '}
                        {Array.isArray(v) ? v.join(', ') : String(v)}
                      </div>
                    ))}
                  </div>
                )}
                <button
                  onClick={() => setSelected(null)}
                  className="mt-2 text-[10px] text-zinc-400 hover:text-zinc-600"
                >
                  Dismiss
                </button>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
