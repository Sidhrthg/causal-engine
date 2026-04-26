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

// ─── Knowledge Graph ──────────────────────────────────────────────────────────

export interface KGEntity {
  id: string;
  entity_type: string;
  name?: string;
  aliases?: string[];
  properties?: Record<string, unknown>;
}

export interface KGRelationship {
  source_id: string;
  target_id: string;
  relation_type: string;
  weight?: number;
  confidence?: number;
}

export interface KGResponse {
  entities: KGEntity[];
  relationships?: KGRelationship[];
  metadata: {
    num_entities: number;
    num_relationships: number;
    entity_types: string[];
    relation_types: string[];
    filtered_by?: string;
  };
}

export interface ExtractedShock {
  commodity: string;
  shock: {
    type: string;
    start_year: number;
    end_year: number;
    magnitude: number;
  };
  affected_entities: string[];
  reasoning: string;
  evidence: string;
  confidence: number;
}

export interface ExtractShockResponse {
  n_shocks_extracted: number;
  extraction_method: string;
  shocks: ExtractedShock[];
}

export interface PredictFromTextResponse {
  commodity: string;
  text_length: number;
  n_shocks_extracted: number;
  n_shocks_applied: number;
  extraction_method: string;
  extracted_shocks: ExtractedShock[];
  trajectory: TrajectoryRow[];
  metrics: Record<string, number>;
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

// ─── Scenario Builder ─────────────────────────────────────────────────────────

export interface ScenarioPayload {
  year: number;
  shock_origin: string;
  commodity: string;
  title: string;
  scenario_id?: string;
}

export interface ScenarioResult {
  scenario_id: string;
  image_url: string;
  node_count: number;
  focal_count: number;
  edge_count: number;
  impact_count: number;
  effective_share: number | null;
  binding: string | null;
  query: string;
  skipped: boolean;
}

export interface ScenarioPreset {
  scenario_id: string;
  kind: 'validation' | 'predictive';
  year: number;
  shock_origin: string;
  commodity: string;
  title: string;
  image_url: string;
  available: boolean;
}

export interface ScenarioPresetsResponse {
  validation: ScenarioPreset[];
  predictive: ScenarioPreset[];
}

// ─── KG Enrichment ────────────────────────────────────────────────────────────

export interface KGEnrichResponse {
  result: string;
}
