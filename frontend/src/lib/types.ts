export interface RouteResult {
  rank: number;
  path: string[];
  bottleneck_t: number;
  pct_of_source: number;
  is_circumvention: boolean;
  non_producer_intermediaries: string[];
  hops: number;
}

export interface TransshipmentResponse {
  commodity: string;
  source: string;
  destination: string;
  year: number;
  routes: RouteResult[];
  circumvention_rate: number;
  circumvention_rate_ci: [number, number];
  nominal_restriction_t: number;
  detected_rerouted_t: number;
  significant_hubs: string[];
  notes: string[];
  summary: string;
}

export interface SourceChunk {
  text: string;
  source: string;
  similarity: number;
}

export interface QueryResponse {
  question: string;
  answer: string;
  sources: SourceChunk[];
  backend: string;
  episode_id: string;
}

export interface ChatMessage {
  role: 'user' | 'assistant';
  content: string;
  sources?: SourceChunk[];
  timestamp: Date;
}

// ─── Counterfactual ───────────────────────────────────────────────────────────

export interface ScenarioMeta {
  name: string;
  commodity: string;
  description: string;
  start_year: number;
  end_year: number;
  has_shocks: boolean;
  calibrated: boolean;
}

export interface TrajectoryRow {
  year: number;
  P: number;
  Q_total: number;
  shortage: number;
  tight: number;
  K: number;
  D: number;
  Q_sub: number;
  Q_fringe: number;
}

export interface CounterfactualResponse {
  scenario: string;
  cf_type: string;
  description: string;
  factual: TrajectoryRow[];
  counterfactual: TrajectoryRow[];
  ate: Record<string, number>;
  factual_params: Record<string, number>;
  cf_params: Record<string, unknown>;
}
