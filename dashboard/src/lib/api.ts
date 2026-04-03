const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8099";

async function fetchAPI<T>(path: string): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, { next: { revalidate: 60 } });
  if (!res.ok) throw new Error(`API error: ${res.status} ${res.statusText}`);
  return res.json();
}

// Types
export interface Prediction {
  ticker: string;
  horizon: number;
  timestamp: string;
  best_model: string | null;
  best_prediction: {
    direction: string;
    label: number;
    confidence: number;
    probabilities: { down: number; flat: number; up: number };
  } | null;
  predictions: Record<string, any>;
  regime: { state: number; label: string } | null;
  error: string | null;
}

export interface PriceData {
  ticker: string;
  count: number;
  data: { time: string; open: number; high: number; low: number; close: number; volume: number }[];
}

export interface ModelResult {
  model: string;
  horizon: number;
  accuracy: number;
  directional_accuracy: number;
  macro_f1: number;
  sharpe_ratio: number;
  max_drawdown: number;
  win_rate: number;
  zero_shot: boolean;
}

export interface PipelineStatus {
  database_connected: boolean;
  tables: Record<string, number>;
  last_price_update: string | null;
  last_news_update: string | null;
  models_available: number;
}

export interface SentimentArticle {
  time: string;
  headline: string;
  source: string;
  sentiment_finbert: number | null;
  sentiment_llm: number | null;
}

// API functions
export const api = {
  health: () => fetchAPI<{ status: string; timestamp: string; models_loaded: number }>("/health"),
  predictions: (horizon = 5) => fetchAPI<Prediction[]>(`/api/predictions/latest?horizon=${horizon}`),
  prediction: (ticker: string, horizon = 5) => fetchAPI<Prediction>(`/api/predictions/${ticker}?horizon=${horizon}`),
  prices: (ticker: string, days = 365) => fetchAPI<PriceData>(`/api/prices/${ticker}?days=${days}`),
  models: () => fetchAPI<{ models: ModelResult[] }>("/api/models/results"),
  sentiment: (limit = 50) => fetchAPI<{ count: number; articles: SentimentArticle[] }>(`/api/sentiment/latest?limit=${limit}`),
  pipelineStatus: () => fetchAPI<PipelineStatus>("/api/pipeline/status"),
};
