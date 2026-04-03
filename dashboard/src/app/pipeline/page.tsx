"use client";

import { useQuery } from "@tanstack/react-query";
import { api } from "@/lib/api";

function StatusDot({ ok }: { ok: boolean }) {
  return <span className={`inline-block w-2 h-2 rounded-full ${ok ? "bg-emerald-400 shadow-[0_0_6px_hsl(142,71%,45%,0.5)]" : "bg-red-400 shadow-[0_0_6px_hsl(0,72%,51%,0.5)]"}`} />;
}

export default function PipelinePage() {
  const { data: status, isLoading } = useQuery({
    queryKey: ["pipeline-status"],
    queryFn: () => api.pipelineStatus(),
    refetchInterval: 30_000,
  });
  const { data: health } = useQuery({
    queryKey: ["health"],
    queryFn: () => api.health(),
    refetchInterval: 30_000,
  });

  if (isLoading) return (
    <div className="flex items-center justify-center h-[60vh]">
      <div className="flex items-center gap-3 text-[hsl(var(--muted-foreground))]">
        <div className="w-4 h-4 border-2 border-emerald-400 border-t-transparent rounded-full animate-spin" />
        <span className="text-sm">Checking pipeline...</span>
      </div>
    </div>
  );

  return (
    <div className="space-y-5">
      <div>
        <h2 className="text-lg font-bold text-white">Pipeline Health</h2>
        <p className="text-xs text-[hsl(var(--muted-foreground))] mt-0.5">System status and data freshness</p>
      </div>

      {/* Status cards */}
      <div className="grid grid-cols-3 gap-3">
        {[
          { label: "API Server", ok: !!health, detail: health ? "Running" : "Down" },
          { label: "Database", ok: status?.database_connected || false, detail: status?.database_connected ? "Connected" : "Disconnected" },
          { label: "Models", ok: (status?.models_available || 0) > 0, detail: `${status?.models_available || 0} loaded` },
        ].map((s) => (
          <div key={s.label} className="rounded-xl border border-border/40 bg-[hsl(var(--card))] p-4 flex items-center gap-3">
            <StatusDot ok={s.ok} />
            <div>
              <p className="text-[10px] uppercase tracking-wider text-[hsl(var(--muted-foreground))]">{s.label}</p>
              <p className="text-sm font-medium text-white">{s.detail}</p>
            </div>
          </div>
        ))}
      </div>

      {/* Tables */}
      <div className="rounded-xl border border-border/40 bg-[hsl(var(--card))] overflow-hidden">
        <div className="px-5 py-3 border-b border-border/30">
          <h3 className="text-xs font-semibold text-[hsl(var(--muted-foreground))] uppercase tracking-wider">Database Tables</h3>
        </div>
        <table className="w-full">
          <thead>
            <tr className="text-[10px] uppercase tracking-wider text-[hsl(var(--muted-foreground))] border-b border-border/20">
              <th className="text-left py-2.5 px-5">Table</th>
              <th className="text-right py-2.5 px-5">Rows</th>
              <th className="text-right py-2.5 px-5">Status</th>
            </tr>
          </thead>
          <tbody>
            {Object.entries(status?.tables || {}).map(([table, count]) => (
              <tr key={table} className="border-b border-border/10 hover:bg-[hsl(var(--card-hover))] transition-colors">
                <td className="py-2.5 px-5 font-mono text-xs text-white">{table}</td>
                <td className="py-2.5 px-5 text-right font-mono text-xs text-[hsl(var(--muted-foreground))]">{(count as number).toLocaleString()}</td>
                <td className="py-2.5 px-5 text-right"><StatusDot ok={(count as number) > 0} /></td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Timestamps */}
      <div className="rounded-xl border border-border/40 bg-[hsl(var(--card))] p-5 space-y-3">
        <h3 className="text-xs font-semibold text-[hsl(var(--muted-foreground))] uppercase tracking-wider">Last Updates</h3>
        {[
          { label: "Price data", value: status?.last_price_update },
          { label: "News articles", value: status?.last_news_update },
          { label: "API health", value: health?.timestamp },
        ].map((r) => (
          <div key={r.label} className="flex justify-between text-xs">
            <span className="text-[hsl(var(--muted-foreground))]">{r.label}</span>
            <span className="text-white font-mono">{r.value ? new Date(r.value).toLocaleString() : "Never"}</span>
          </div>
        ))}
      </div>

      {/* Commands */}
      <div className="rounded-xl border border-border/40 bg-[hsl(var(--card))] p-5">
        <h3 className="text-xs font-semibold text-[hsl(var(--muted-foreground))] uppercase tracking-wider mb-3">Quick Commands</h3>
        <div className="grid grid-cols-2 gap-1.5">
          {[
            { cmd: "make daily-update", desc: "Update prices + news" },
            { cmd: "make score-sentiment", desc: "Score new headlines" },
            { cmd: "make features", desc: "Rebuild features" },
            { cmd: "make train-baseline", desc: "Retrain models" },
            { cmd: "make backfill", desc: "Full data backfill" },
            { cmd: "make compare-models", desc: "Compare all models" },
          ].map((c) => (
            <div key={c.cmd} className="flex items-center gap-3 py-1.5">
              <code className="text-[11px] bg-[hsl(var(--muted))] px-2.5 py-1 rounded-lg font-mono text-emerald-400 shrink-0">{c.cmd}</code>
              <span className="text-[11px] text-[hsl(var(--muted-foreground))]">{c.desc}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
