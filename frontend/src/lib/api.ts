import type {
  TransshipmentResponse,
  QueryResponse,
  KGResponse,
  ExtractShockResponse,
  PredictFromTextResponse,
} from './types';

async function post<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error((err as { detail: string }).detail || 'Request failed');
  }
  return res.json() as Promise<T>;
}

export async function analyzeTransshipment(params: {
  commodity: string;
  source: string;
  destination: string;
  year: number;
  event_years: number[];
  max_hops: number;
  data_path: string;
  nominal_restriction: number;
}): Promise<TransshipmentResponse> {
  return post<TransshipmentResponse>('/api/transshipment', params);
}

export async function queryKnowledgeBase(params: {
  question: string;
  commodity?: string;
  top_k?: number;
}): Promise<QueryResponse> {
  return post<QueryResponse>('/api/query', params);
}

export async function getCommodities(): Promise<{
  commodities: string[];
  hs_codes: Record<string, string>;
}> {
  const res = await fetch('/api/commodities');
  if (!res.ok) throw new Error('Failed to fetch commodities');
  return res.json();
}

export async function getScenarios(): Promise<{ scenarios: import('./types').ScenarioMeta[] }> {
  const res = await fetch('/api/scenarios');
  if (!res.ok) throw new Error('Failed to fetch scenarios');
  return res.json();
}

export async function getKnowledgeGraph(commodity?: string): Promise<KGResponse> {
  const params = new URLSearchParams();
  if (commodity) params.set('commodity', commodity);
  const res = await fetch(`/api/knowledge-graph?${params}`);
  if (!res.ok) throw new Error('Failed to fetch knowledge graph');
  return res.json();
}

export async function extractShocks(params: {
  text: string;
  use_llm?: boolean;
  default_duration?: number;
}): Promise<ExtractShockResponse> {
  return post<ExtractShockResponse>('/api/extract-shock', params);
}

export async function predictFromText(params: {
  text: string;
  commodity: string;
  start_year: number;
  end_year: number;
  use_llm?: boolean;
  baseline_P0?: number;
  baseline_K0?: number;
}): Promise<PredictFromTextResponse> {
  return post<PredictFromTextResponse>('/api/predict-from-text', params);
}

export async function runCounterfactual(params: {
  scenario: string;
  cf_type: 'substitution' | 'fringe' | 'trajectory';
  cf_elasticity?: number;
  cf_cap?: number;
  cf_capacity_share?: number;
  cf_entry_price?: number;
  shock_overrides?: Record<string, Record<string, number>>;
  use_calibrated?: boolean;
}): Promise<import('./types').CounterfactualResponse> {
  return post('/api/counterfactual', params);
}

export async function renderScenario(
  payload: import('./types').ScenarioPayload,
): Promise<import('./types').ScenarioResult> {
  return post<import('./types').ScenarioResult>(
    '/api/kg/render-scenario',
    payload,
  );
}

export async function getScenarioPresets(): Promise<import('./types').ScenarioPresetsResponse> {
  const res = await fetch('/api/kg/scenario-presets');
  if (!res.ok) throw new Error('Failed to fetch scenario presets');
  return res.json();
}
