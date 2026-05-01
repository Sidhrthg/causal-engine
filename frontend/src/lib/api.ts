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
  const data = await res.json();
  // Backend currently returns list[str] of YAML filenames; convert to ScenarioMeta-like.
  // When backend is upgraded to return objects, this passthrough handles both shapes.
  const scenarios = (data.scenarios || []).map((s: unknown) => {
    if (typeof s === 'string') {
      const name = s.replace(/\.ya?ml$/, '');
      const lower = name.toLowerCase();
      const commodity = ['graphite','rare_earths','rare-earths','cobalt','lithium','nickel','uranium','germanium','gallium','copper','soybeans']
        .find((c) => lower.includes(c.replace('-','_'))) || 'unknown';
      return {
        name,
        commodity,
        description: '',
        start_year: 2024,
        end_year: 2032,
        has_shocks: lower.includes('ban') || lower.includes('restrict') || lower.includes('shock'),
        calibrated: lower.includes('calibrated'),
      } as import('./types').ScenarioMeta;
    }
    return s as import('./types').ScenarioMeta;
  });
  return { scenarios };
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

export async function startRenderScenario(
  payload: import('./types').ScenarioPayload,
): Promise<{ job_id: string; status: string }> {
  return post('/api/kg/render-scenario/start', payload);
}

export async function getRenderScenarioStatus(
  jobId: string,
): Promise<{
  job_id: string;
  status: 'running' | 'done' | 'failed';
  result?: import('./types').ScenarioResult;
  error?: string;
}> {
  const res = await fetch(
    `/api/kg/render-scenario/status?job_id=${encodeURIComponent(jobId)}`,
  );
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error((err as { detail: string }).detail || 'Status check failed');
  }
  return res.json();
}

export async function renderScenarioAsync(
  payload: import('./types').ScenarioPayload,
  opts: { intervalMs?: number; timeoutMs?: number } = {},
): Promise<import('./types').ScenarioResult> {
  const intervalMs = opts.intervalMs ?? 4000;
  const timeoutMs = opts.timeoutMs ?? 5 * 60 * 1000;
  const { job_id } = await startRenderScenario(payload);
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    await new Promise((r) => setTimeout(r, intervalMs));
    const s = await getRenderScenarioStatus(job_id);
    if (s.status === 'done' && s.result) return s.result;
    if (s.status === 'failed') throw new Error(s.error || 'Render failed');
  }
  throw new Error('Render timed out after 5 minutes');
}

export async function getScenarioPresets(): Promise<import('./types').ScenarioPresetsResponse> {
  const res = await fetch('/api/kg/scenario-presets');
  if (!res.ok) throw new Error('Failed to fetch scenario presets');
  return res.json();
}

export async function enrichKG(params: {
  query: string;
  top_k?: number;
}): Promise<import('./types').KGEnrichResponse> {
  return post<import('./types').KGEnrichResponse>('/api/kg/enrich', params);
}

export async function batchEnrichKG(params: {
  top_k?: number;
}): Promise<import('./types').KGEnrichResponse> {
  return post<import('./types').KGEnrichResponse>('/api/kg/batch-enrich', params);
}

export async function getTemporalComparison(
  commodity?: string,
): Promise<import('./types').TemporalComparisonResponse> {
  const qs = commodity ? `?commodity=${encodeURIComponent(commodity)}` : '';
  const res = await fetch(`/api/kg/temporal-comparison${qs}`);
  if (!res.ok) throw new Error('Failed to fetch temporal comparison');
  return res.json();
}

export async function getYearlyShares(
  commodity: string,
): Promise<import('./types').YearlySharesResponse> {
  const res = await fetch(`/api/kg/yearly-shares?commodity=${encodeURIComponent(commodity)}`);
  if (!res.ok) throw new Error('Failed to fetch yearly shares');
  return res.json();
}

export interface ForecastRequest {
  commodity: string;
  shock_year: number;
  severity: 'baseline' | 'mild_ban' | 'full_ban' | 'severe_ban' | 'custom';
  restriction_magnitude?: number;
  restriction_duration?: number;
  horizon_years?: number;
  demand_surge?: number;
  n_bootstrap?: number;
}

export interface ForecastPoint {
  year: number;
  price_index: number;
}

export interface ForecastResponse {
  commodity: string;
  shock_year: number;
  horizon_years: number;
  severity: string;
  restriction_magnitude: number;
  restriction_start: number;
  restriction_end: number;
  params_used: { alpha_P: number; eta_D: number; tau_K: number; g: number };
  baseline_path: ForecastPoint[];
  scenario_path: ForecastPoint[];
  ci_band: { year: number; low: number; high: number }[] | null;
  direction: 'up' | 'down' | 'flat';
  peak_year: number | null;
  peak_index: number | null;
  peak_vs_baseline: number | null;
  normalization_year: number | null;
  normalization_lag_years: number | null;
  accuracy: {
    in_sample_DA: number | null;
    oos_DA: number | null;
    in_sample_episodes: string[];
    oos_pairs: string[];
  };
}

export async function runForecast(params: ForecastRequest): Promise<ForecastResponse> {
  return post<ForecastResponse>('/api/forecast/forward', params);
}

export async function getForecastCommodities(): Promise<{
  commodities: { commodity: string; in_sample_DA: number | null; oos_DA: number | null }[];
}> {
  const res = await fetch('/api/forecast/commodities');
  if (!res.ok) throw new Error('Failed to fetch forecast commodities');
  return res.json();
}

export async function getYearSnapshot(params: {
  commodity: string;
  year: number;
  shock_origin?: string;
}): Promise<import('./types').YearSnapshotResponse> {
  const qs = new URLSearchParams({
    commodity: params.commodity,
    year: String(params.year),
  });
  if (params.shock_origin) qs.set('shock_origin', params.shock_origin);
  const res = await fetch(`/api/kg/year-snapshot?${qs.toString()}`);
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error((err as { detail: string }).detail || 'Snapshot failed');
  }
  return res.json();
}
