"use client";

import { useQuery } from "@tanstack/react-query";
import { useState } from "react";
import dynamic from "next/dynamic";
import { api, Prediction } from "@/lib/api";
import { formatPercent } from "@/lib/utils";
import { AreaChart, Area, ResponsiveContainer } from "recharts";

const CandlestickChart = dynamic(() => import("@/components/CandlestickChart"), { ssr: false });

const REGIONS = ["All", "US", "Egypt", "Saudi", "UAE", "Qatar", "Kuwait"] as const;
const CHART_RANGES = [
  { label: "1W", days: 7 }, { label: "1M", days: 30 }, { label: "3M", days: 90 },
  { label: "6M", days: 180 }, { label: "1Y", days: 365 }, { label: "ALL", days: 9999 },
];

const REGION_MAP: Record<string, string[]> = {
  US: ["AAPL","MSFT","GOOGL","AMZN","NVDA","META","TSLA","BRK-B","JPM","V","SPY","QQQ","IWM","GLD","TLT","HYG","XLF","XLE","XLK","XLV","XLI"],
  Egypt: ["COMI.CA","TMGH.CA","EAST.CA","HRHO.CA","ORAS.CA","SWDY.CA","FWRY.CA","EFIH.CA","ETEL.CA","PHDC.CA","ESRS.CA","ABUK.CA","EGPT"],
  Saudi: ["2222.SR","1120.SR","7010.SR","2010.SR","1180.SR","5110.SR","1010.SR","1150.SR","2082.SR","1211.SR","KSA"],
  UAE: ["ETISALAT.AB","DFM.AE","EMAAR.AE","EMIRATESNBD.AE","DIB.AE","DEWA.AE","UAE"],
  Qatar: ["QNBK.QA","IQCD.QA","ORDS.QA","QAT"],
  Kuwait: ["NBK.KW"],
};

function Spark({ dir }: { dir: string }) {
  const c = dir === "up" ? "#22c55e" : dir === "down" ? "#ef4444" : "#eab308";
  const pts = dir === "up" ? [3,5,4,6,5,7,6,8,9,11] : dir === "down" ? [11,9,10,7,8,5,6,4,3,2] : [5,6,5,7,5,6,5,6,5,6];
  return (
    <ResponsiveContainer width={50} height={22}>
      <AreaChart data={pts.map(v=>({v}))} margin={{top:2,right:0,bottom:2,left:0}}>
        <defs><linearGradient id={`s${dir}`} x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stopColor={c} stopOpacity={0.2}/><stop offset="100%" stopColor={c} stopOpacity={0}/></linearGradient></defs>
        <Area type="monotone" dataKey="v" stroke={c} strokeWidth={1.5} fill={`url(#s${dir})`} dot={false}/>
      </AreaChart>
    </ResponsiveContainer>
  );
}

export default function Home() {
  const [region, setRegion] = useState("All");
  const [ticker, setTicker] = useState("SPY");
  const [days, setDays] = useState(90);
  const [chartMode, setChartMode] = useState<"candle"|"line">("line");

  const { data: preds, isLoading } = useQuery({ queryKey: ["preds"], queryFn: () => api.predictions(5) });
  const { data: status } = useQuery({ queryKey: ["status"], queryFn: () => api.pipelineStatus() });
  const { data: priceData } = useQuery({ queryKey: ["px", ticker, days], queryFn: () => api.prices(ticker, days), staleTime: 300_000 });

  if (isLoading) return (
    <div className="flex items-center justify-center h-[60vh] text-[hsl(var(--muted-foreground))] text-sm gap-3">
      <div className="w-4 h-4 border-2 border-emerald-400 border-t-transparent rounded-full animate-spin"/>Loading...
    </div>
  );

  const all = preds || [];
  const filtered = region === "All" ? all : all.filter(p => REGION_MAP[region]?.includes(p.ticker));
  const wp = filtered.filter(p => p.best_prediction);
  const prices = priceData?.data || [];
  const last = prices[prices.length - 1];
  const first = prices[0];
  const chg = last && first ? ((last.close - first.close) / first.close) * 100 : 0;
  const up = chg >= 0;
  const hi = prices.length ? Math.max(...prices.map(p=>p.high)) : 0;
  const lo = prices.length ? Math.min(...prices.map(p=>p.low)) : 0;
  const tp = all.find(p => p.ticker === ticker);

  return (
    <div className="space-y-5">
      {/* Breadcrumb */}
      <div className="flex items-center gap-2 text-sm">
        <span className="text-[hsl(var(--muted-foreground))]">Invest</span>
        <span className="text-[hsl(var(--muted-foreground))]">/</span>
        <span className="text-white font-medium">{ticker}</span>
      </div>

      {/* ── Ticker header card ── */}
      <div className="rounded-2xl bg-[hsl(var(--card))] border border-white/[0.04] p-6">
        {/* Ticker info row */}
        <div className="flex items-center gap-4 mb-5">
          <div className="w-11 h-11 rounded-xl bg-[hsl(var(--secondary))] flex items-center justify-center text-sm font-bold text-white">
            {ticker.replace(/\.(CA|SR|AB|AE|QA|KW)/g,"").slice(0,2)}
          </div>
          <div className="flex-1">
            <div className="flex items-center gap-3">
              <select value={ticker} onChange={e => setTicker(e.target.value)}
                className="bg-transparent text-white text-base font-semibold outline-none cursor-pointer appearance-none">
                {all.map(p => <option key={p.ticker} value={p.ticker} className="bg-[#161a24]">{p.ticker.replace(/\.(CA|SR|AB|AE|QA|KW)/g,"")}</option>)}
                {all.length === 0 && <option value="SPY">SPY</option>}
              </select>
              {tp?.best_prediction && (
                <span className={`text-[11px] font-medium px-2.5 py-0.5 rounded-lg ${
                  tp.best_prediction.direction === "up" ? "badge-green" : tp.best_prediction.direction === "down" ? "badge-red" : "badge-yellow"
                }`}>AI: {tp.best_prediction.direction}</span>
              )}
            </div>
            <p className="text-xs text-[hsl(var(--muted-foreground))]">{ticker}</p>
          </div>
          {/* Chart mode toggle */}
          <div className="flex gap-1 bg-[hsl(var(--secondary))] rounded-lg p-0.5">
            {(["line","candle"] as const).map(m => (
              <button key={m} onClick={() => setChartMode(m)}
                className={`px-3 py-1 rounded-md text-[11px] transition-all ${chartMode === m ? "bg-white/10 text-white" : "text-[hsl(var(--muted-foreground))]"}`}>
                {m === "line" ? "Line" : "Candle"}
              </button>
            ))}
          </div>
        </div>

        {/* Chart area */}
        <div className="rounded-xl bg-[hsl(var(--background))] border border-white/[0.03] p-4">
          {/* Price overlay */}
          <div className="flex items-end justify-between mb-2">
            <div>
              {last && (
                <>
                  <p className="text-3xl font-bold text-white tracking-tight">
                    ${last.close.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
                  </p>
                  <p className={`text-sm font-medium mt-0.5 ${up ? "text-emerald-400" : "text-red-400"}`}>
                    {up ? "+" : ""}{(last.close - first.close).toFixed(2)} ({chg.toFixed(2)}%)
                  </p>
                </>
              )}
            </div>
            {/* Time range pills */}
            <div className="flex gap-0.5">
              {CHART_RANGES.map(r => (
                <button key={r.label} onClick={() => setDays(r.days)}
                  className={`px-2.5 py-1 rounded-lg text-[11px] font-medium transition-all ${
                    days === r.days ? "bg-white/10 text-white" : "text-[hsl(var(--muted-foreground))] hover:text-white"
                  }`}>{r.label}</button>
              ))}
            </div>
          </div>

          {/* Chart */}
          {prices.length > 0 ? (
            <CandlestickChart data={prices} height={320} mode={chartMode} />
          ) : (
            <div className="h-[320px] flex items-center justify-center text-[hsl(var(--muted-foreground))] text-sm">No data for {ticker}</div>
          )}
        </div>

        {/* Stats row below chart */}
        <div className="flex gap-8 mt-4 px-1">
          {[
            { label: "High", value: `$${hi.toLocaleString(undefined,{minimumFractionDigits:2})}` },
            { label: "Low", value: `$${lo.toLocaleString(undefined,{minimumFractionDigits:2})}` },
            { label: "Change", value: `${up?"+":""}${chg.toFixed(2)}%`, color: up ? "text-emerald-400" : "text-red-400" },
            ...(last ? [{ label: "Volume", value: `${(last.volume/1e6).toFixed(1)}M` }] : []),
          ].map(s => (
            <div key={s.label}>
              <p className="text-[10px] text-[hsl(var(--muted-foreground))] uppercase tracking-wider">{s.label}</p>
              <p className={`text-sm font-semibold mt-0.5 ${"color" in s ? s.color : "text-white"}`}>{s.value}</p>
            </div>
          ))}
        </div>
      </div>

      {/* ── Tabs: Overview, Forecast ── */}
      <div className="flex gap-1">
        {["Overview", "Forecast"].map(t => (
          <span key={t} className={`px-4 py-2 rounded-xl text-[12px] font-medium cursor-default ${
            t === "Overview" ? "bg-[hsl(var(--secondary))] text-white" : "text-[hsl(var(--muted-foreground))]"
          }`}>{t}</span>
        ))}
      </div>

      {/* ── Right-side market trend + table ── */}
      <div className="grid grid-cols-12 gap-4">
        {/* Predictions table */}
        <div className="col-span-8">
          <div className="rounded-2xl bg-[hsl(var(--card))] border border-white/[0.04] overflow-hidden">
            <div className="flex items-center justify-between px-5 py-3 border-b border-white/[0.04]">
              <h3 className="text-sm font-semibold text-white">Predictions</h3>
              <div className="flex gap-1">
                {REGIONS.map(r => (
                  <button key={r} onClick={() => setRegion(r)}
                    className={`px-3 py-1 rounded-lg text-[10px] font-medium transition-all ${
                      region === r ? "bg-white/[0.08] text-white" : "text-[hsl(var(--muted-foreground))] hover:text-white"
                    }`}>{r}</button>
                ))}
              </div>
            </div>
            <table className="w-full">
              <thead>
                <tr className="text-[10px] uppercase tracking-wider text-[hsl(var(--muted-foreground))] border-b border-white/[0.03]">
                  <th className="text-left py-2.5 px-5">Asset</th>
                  <th className="text-left py-2.5 px-3">Direction</th>
                  <th className="text-right py-2.5 px-3">Confidence</th>
                  <th className="text-right py-2.5 px-3">P(Up)</th>
                  <th className="text-right py-2.5 px-3">P(Down)</th>
                  <th className="text-right py-2.5 px-3">Trend</th>
                </tr>
              </thead>
              <tbody>
                {filtered.map(p => {
                  const pr = p.best_prediction; if(!pr) return null;
                  const d = pr.direction;
                  const sh = p.ticker.replace(/\.(CA|SR|AB|AE|DU|QA|KW)/g,"");
                  const sel = p.ticker === ticker;
                  return (
                    <tr key={p.ticker} onClick={() => setTicker(p.ticker)}
                      className={`border-b border-white/[0.02] cursor-pointer transition-colors ${sel ? "bg-white/[0.03]" : "hover:bg-white/[0.02]"}`}>
                      <td className="py-2.5 px-5">
                        <div className="flex items-center gap-2.5">
                          <div className={`w-7 h-7 rounded-lg flex items-center justify-center text-[9px] font-bold ${
                            d==="up"?"bg-emerald-500/10 text-emerald-400":d==="down"?"bg-red-500/10 text-red-400":"bg-amber-500/10 text-amber-400"
                          }`}>{sh.slice(0,2)}</div>
                          <span className="text-[13px] font-medium text-white">{sh}</span>
                        </div>
                      </td>
                      <td className="py-2.5 px-3">
                        <span className={`text-xs font-medium ${d==="up"?"text-emerald-400":d==="down"?"text-red-400":"text-amber-400"}`}>
                          {d==="up"?"\u25B2":d==="down"?"\u25BC":"\u25C6"} {d}
                        </span>
                      </td>
                      <td className="py-2.5 px-3 text-right text-[11px] font-mono text-[hsl(var(--muted-foreground))]">{(pr.confidence*100).toFixed(1)}%</td>
                      <td className="py-2.5 px-3 text-right text-[11px] font-mono text-emerald-400">+{(pr.probabilities.up*100).toFixed(1)}%</td>
                      <td className="py-2.5 px-3 text-right text-[11px] font-mono text-red-400">-{(pr.probabilities.down*100).toFixed(1)}%</td>
                      <td className="py-2.5 px-3 text-right"><Spark dir={d}/></td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
            {filtered.length===0 && <div className="py-12 text-center text-sm text-[hsl(var(--muted-foreground))]">No predictions for {region}</div>}
          </div>
        </div>

        {/* Market trend sidebar */}
        <div className="col-span-4 space-y-4">
          <div className="rounded-2xl bg-[hsl(var(--card))] border border-white/[0.04] p-4">
            <p className="text-xs font-semibold text-white mb-3">Market Trend</p>
            <div className="space-y-0.5">
              {wp.slice(0,8).map(p => {
                const pr = p.best_prediction!; const d = pr.direction;
                const sh = p.ticker.replace(/\.(CA|SR|AB|AE|DU|QA|KW)/g,"");
                return (
                  <div key={p.ticker} onClick={() => setTicker(p.ticker)}
                    className={`flex items-center gap-2.5 py-2 px-2 rounded-xl cursor-pointer transition-colors ${
                      p.ticker===ticker ? "bg-white/[0.04]" : "hover:bg-white/[0.02]"
                    }`}>
                    <div className={`w-7 h-7 rounded-lg flex items-center justify-center text-[9px] font-bold ${
                      d==="up"?"bg-emerald-500/10 text-emerald-400":d==="down"?"bg-red-500/10 text-red-400":"bg-amber-500/10 text-amber-400"
                    }`}>{sh.slice(0,2)}</div>
                    <div className="flex-1 min-w-0">
                      <p className="text-[12px] font-medium text-white truncate">{sh}</p>
                      <p className="text-[10px] text-[hsl(var(--muted-foreground))]">{p.ticker}</p>
                    </div>
                    <Spark dir={d}/>
                    <span className={`text-[11px] font-medium ${d==="up"?"text-emerald-400":d==="down"?"text-red-400":"text-amber-400"}`}>
                      {d==="up"?"+":d==="down"?"-":""}{(pr.confidence*100).toFixed(0)}%
                    </span>
                  </div>
                );
              })}
            </div>
          </div>

          {/* Quick stats */}
          <div className="rounded-2xl bg-[hsl(var(--card))] border border-white/[0.04] p-4">
            <p className="text-xs font-semibold text-white mb-3">Summary</p>
            {[
              { l: "Total tickers", v: String(wp.length), c: "text-white" },
              { l: "Bullish", v: String(wp.filter(p=>p.best_prediction!.direction==="up").length), c: "text-emerald-400" },
              { l: "Bearish", v: String(wp.filter(p=>p.best_prediction!.direction==="down").length), c: "text-red-400" },
              { l: "Models", v: String(status?.models_available||0), c: "text-white" },
            ].map(s => (
              <div key={s.l} className="flex justify-between py-1.5">
                <span className="text-[11px] text-[hsl(var(--muted-foreground))]">{s.l}</span>
                <span className={`text-[13px] font-semibold ${s.c}`}>{s.v}</span>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
