"use client";

import { useQuery } from "@tanstack/react-query";
import { api, ModelResult } from "@/lib/api";
import { formatPercent, formatNumber } from "@/lib/utils";
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from "recharts";

function StatRow({ label, value, color }: { label: string; value: string; color?: string }) {
  return (
    <div className="flex justify-between items-center py-1.5">
      <span className="text-[11px] text-[hsl(var(--muted-foreground))]">{label}</span>
      <span className={`text-[12px] font-mono font-medium ${color || "text-white"}`}>{value}</span>
    </div>
  );
}

function MetricBar({ value, max = 1, color }: { value: number; max?: number; color: string }) {
  return (
    <div className="w-full h-1 bg-[hsl(var(--muted))] rounded-full overflow-hidden">
      <div className="h-full rounded-full transition-all" style={{ width: `${Math.min((value / max) * 100, 100)}%`, backgroundColor: color }} />
    </div>
  );
}

export default function ModelsPage() {
  const { data, isLoading, error } = useQuery({
    queryKey: ["model-results"],
    queryFn: () => api.models(),
    staleTime: 300_000,
  });

  if (isLoading) return (
    <div className="flex items-center justify-center h-[60vh]">
      <div className="flex items-center gap-3 text-[hsl(var(--muted-foreground))]">
        <div className="w-4 h-4 border-2 border-emerald-400 border-t-transparent rounded-full animate-spin" />
        <span className="text-sm">Loading model results...</span>
      </div>
    </div>
  );

  const models = data?.models || [];
  const chartData = models.map(m => ({
    name: `${m.model} ${m.horizon}d`,
    acc: +(m.accuracy * 100).toFixed(1),
    dir: +(m.directional_accuracy * 100).toFixed(1),
  }));

  return (
    <div className="space-y-5">
      <div>
        <h2 className="text-lg font-bold text-white">Model Performance</h2>
        <p className="text-xs text-[hsl(var(--muted-foreground))] mt-0.5">{models.length} models evaluated across walk-forward folds</p>
      </div>

      {models.length === 0 ? (
        <div className="rounded-xl border border-border/40 bg-[hsl(var(--card))] p-16 text-center">
          <p className="text-sm text-[hsl(var(--muted-foreground))]">No model results yet</p>
          <code className="text-xs bg-[hsl(var(--muted))] px-3 py-1 rounded-lg mt-2 inline-block">make train-baseline</code>
        </div>
      ) : (
        <>
          {/* Comparison chart */}
          <div className="rounded-xl border border-border/40 bg-[hsl(var(--card))] p-5">
            <h3 className="text-xs font-semibold text-[hsl(var(--muted-foreground))] uppercase tracking-wider mb-4">Accuracy vs Directional Accuracy</h3>
            <ResponsiveContainer width="100%" height={260}>
              <BarChart data={chartData} barGap={2}>
                <CartesianGrid strokeDasharray="3 3" stroke="hsl(220,14%,12%)" vertical={false} />
                <XAxis dataKey="name" tick={{ fontSize: 10, fill: "hsl(220,10%,45%)" }} axisLine={false} tickLine={false} />
                <YAxis tick={{ fontSize: 10, fill: "hsl(220,10%,45%)" }} axisLine={false} tickLine={false} domain={[0, 70]} />
                <Tooltip contentStyle={{ backgroundColor: "hsl(220,16%,7%)", border: "1px solid hsl(220,14%,14%)", borderRadius: "10px", fontSize: "12px", color: "#fff" }} />
                <Bar dataKey="acc" name="Accuracy %" fill="#3b82f6" radius={[4, 4, 0, 0]} />
                <Bar dataKey="dir" name="Dir Acc %" fill="#22c55e" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>

          {/* Model cards grid */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
            {models.map((m) => {
              const isPositiveSharpe = m.sharpe_ratio > 0;
              return (
                <div key={`${m.model}-${m.horizon}`} className="rounded-xl border border-border/40 bg-[hsl(var(--card))] p-4 hover:bg-[hsl(var(--card-hover))] transition-all">
                  <div className="flex items-center justify-between mb-3">
                    <div className="flex items-center gap-2">
                      <div className="w-8 h-8 rounded-lg bg-[hsl(var(--secondary))] flex items-center justify-center text-[10px] font-bold text-white">
                        {m.model.slice(0,2).toUpperCase()}
                      </div>
                      <div>
                        <p className="text-sm font-semibold text-white">{m.model.toUpperCase()}</p>
                        <p className="text-[10px] text-[hsl(var(--muted-foreground))]">{m.horizon}d horizon &middot; {m.n_folds} folds</p>
                      </div>
                    </div>
                    {m.zero_shot && (
                      <span className="text-[9px] px-2 py-0.5 rounded-full badge-blue">zero-shot</span>
                    )}
                  </div>

                  <div className="space-y-1 mb-3">
                    <div className="flex justify-between text-[10px] text-[hsl(var(--muted-foreground))]">
                      <span>Accuracy</span>
                      <span className="text-white font-mono">{(m.accuracy * 100).toFixed(1)}%</span>
                    </div>
                    <MetricBar value={m.accuracy} color="#3b82f6" />

                    <div className="flex justify-between text-[10px] text-[hsl(var(--muted-foreground))] mt-2">
                      <span>Dir Accuracy</span>
                      <span className="text-emerald-400 font-mono">{(m.directional_accuracy * 100).toFixed(1)}%</span>
                    </div>
                    <MetricBar value={m.directional_accuracy} color="#22c55e" />

                    <div className="flex justify-between text-[10px] text-[hsl(var(--muted-foreground))] mt-2">
                      <span>Macro F1</span>
                      <span className="text-white font-mono">{(m.macro_f1 * 100).toFixed(1)}%</span>
                    </div>
                    <MetricBar value={m.macro_f1} color="#a855f7" />
                  </div>

                  <div className="grid grid-cols-2 gap-x-4 gap-y-0.5 pt-3 border-t border-border/30">
                    <StatRow label="Sharpe" value={formatNumber(m.sharpe_ratio)} color={isPositiveSharpe ? "text-emerald-400" : "text-red-400"} />
                    <StatRow label="Win Rate" value={`${(m.win_rate * 100).toFixed(1)}%`} color={m.win_rate > 0.5 ? "text-emerald-400" : "text-red-400"} />
                    <StatRow label="Max DD" value={`${(m.max_drawdown * 100).toFixed(1)}%`} color="text-red-400" />
                    <StatRow label="Folds" value={String(m.n_folds)} />
                  </div>
                </div>
              );
            })}
          </div>
        </>
      )}
    </div>
  );
}
