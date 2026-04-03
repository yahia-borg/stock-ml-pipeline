"use client";

import { useQuery } from "@tanstack/react-query";
import { api } from "@/lib/api";

function SentimentDot({ score }: { score: number }) {
  const color = score > 0.2 ? "bg-emerald-400" : score < -0.2 ? "bg-red-400" : "bg-amber-400";
  return <span className={`inline-block w-2 h-2 rounded-full ${color}`} />;
}

export default function SentimentPage() {
  const { data, isLoading } = useQuery({
    queryKey: ["sentiment"],
    queryFn: () => api.sentiment(100),
  });

  const articles = data?.articles || [];
  const scored = articles.filter(a => a.sentiment_finbert !== null);
  const avg = scored.length > 0 ? scored.reduce((s, a) => s + (a.sentiment_finbert || 0), 0) / scored.length : 0;
  const bullish = scored.filter(a => (a.sentiment_finbert || 0) > 0.1).length;
  const bearish = scored.filter(a => (a.sentiment_finbert || 0) < -0.1).length;

  if (isLoading) return (
    <div className="flex items-center justify-center h-[60vh]">
      <div className="flex items-center gap-3 text-[hsl(var(--muted-foreground))]">
        <div className="w-4 h-4 border-2 border-emerald-400 border-t-transparent rounded-full animate-spin" />
        <span className="text-sm">Loading sentiment...</span>
      </div>
    </div>
  );

  return (
    <div className="space-y-5">
      <div>
        <h2 className="text-lg font-bold text-white">Sentiment Feed</h2>
        <p className="text-xs text-[hsl(var(--muted-foreground))] mt-0.5">FinBERT-scored financial headlines</p>
      </div>

      <div className="grid grid-cols-4 gap-3">
        {[
          { label: "Articles", value: String(scored.length), color: "" },
          { label: "Avg Sentiment", value: avg.toFixed(3), color: avg > 0 ? "text-emerald-400" : avg < 0 ? "text-red-400" : "text-amber-400" },
          { label: "Bullish", value: String(bullish), color: "text-emerald-400" },
          { label: "Bearish", value: String(bearish), color: "text-red-400" },
        ].map((c) => (
          <div key={c.label} className="rounded-xl border border-border/40 bg-[hsl(var(--card))] p-4">
            <p className="text-[10px] uppercase tracking-wider text-[hsl(var(--muted-foreground))]">{c.label}</p>
            <p className={`text-xl font-bold mt-1 ${c.color || "text-white"}`}>{c.value}</p>
          </div>
        ))}
      </div>

      <div className="rounded-xl border border-border/40 bg-[hsl(var(--card))] overflow-hidden divide-y divide-border/20">
        {articles.length === 0 ? (
          <div className="py-16 text-center text-sm text-[hsl(var(--muted-foreground))]">
            No sentiment data. Run: <code className="bg-[hsl(var(--muted))] px-2 py-0.5 rounded text-xs">make collect-news && make score-sentiment</code>
          </div>
        ) : (
          articles.map((a, i) => (
            <div key={i} className="px-4 py-3 hover:bg-[hsl(var(--card-hover))] transition-colors flex items-start gap-3">
              <div className="pt-1.5">
                <SentimentDot score={a.sentiment_finbert || 0} />
              </div>
              <div className="flex-1 min-w-0">
                <p className="text-[13px] text-white truncate">{a.headline}</p>
                <div className="flex gap-3 mt-1 text-[10px] text-[hsl(var(--muted-foreground))]">
                  <span>{a.source}</span>
                  <span>{new Date(a.time).toLocaleString()}</span>
                </div>
              </div>
              {a.sentiment_finbert !== null && (
                <span className={`shrink-0 font-mono text-xs font-medium px-2 py-0.5 rounded-md ${
                  a.sentiment_finbert > 0.1 ? "badge-green" : a.sentiment_finbert < -0.1 ? "badge-red" : "badge-yellow"
                }`}>
                  {a.sentiment_finbert > 0 ? "+" : ""}{a.sentiment_finbert.toFixed(3)}
                </span>
              )}
            </div>
          ))
        )}
      </div>
    </div>
  );
}
