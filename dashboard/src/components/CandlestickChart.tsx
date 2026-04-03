"use client";

import { useEffect, useRef } from "react";
import { createChart, ColorType } from "lightweight-charts";

interface OHLCData {
  time: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

interface Props {
  data: OHLCData[];
  height?: number;
  mode?: "candle" | "line";
}

export default function CandlestickChart({ data, height = 320, mode = "candle" }: Props) {
  const containerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<any>(null);

  useEffect(() => {
    if (!containerRef.current || data.length === 0) return;

    if (chartRef.current) {
      chartRef.current.remove();
      chartRef.current = null;
    }

    const chart = createChart(containerRef.current, {
      height,
      layout: {
        background: { type: ColorType.Solid, color: "transparent" },
        textColor: "#555d6e",
        fontSize: 11,
      },
      grid: {
        vertLines: { visible: false },
        horzLines: { color: "#1e2230", style: 1 },
      },
      crosshair: {
        vertLine: { color: "#3a3f4e", width: 1, style: 2, labelBackgroundColor: "#2a2f3e" },
        horzLine: { color: "#3a3f4e", width: 1, style: 2, labelBackgroundColor: "#2a2f3e" },
      },
      rightPriceScale: {
        borderVisible: false,
        scaleMargins: { top: 0.08, bottom: 0.15 },
      },
      timeScale: {
        borderVisible: false,
        timeVisible: false,
      },
    });

    chartRef.current = chart;

    const formatted = data.map(d => ({
      time: d.time.split("T")[0],
      open: d.open,
      high: d.high,
      low: d.low,
      close: d.close,
      value: d.close,
    }));

    if (mode === "candle") {
      const series = chart.addCandlestickSeries({
        upColor: "#22c55e",
        downColor: "#ef4444",
        borderUpColor: "#22c55e",
        borderDownColor: "#ef4444",
        wickUpColor: "#3a7d5a",
        wickDownColor: "#7d3a3a",
      });
      series.setData(formatted);
    } else {
      // Clean line chart like the Apple screenshot
      const isUp = data[data.length - 1].close >= data[0].close;
      const lineColor = isUp ? "#22c55e" : "#ef4444";

      const series = chart.addAreaSeries({
        lineColor,
        lineWidth: 2,
        topColor: isUp ? "rgba(34, 197, 94, 0.08)" : "rgba(239, 68, 68, 0.08)",
        bottomColor: "transparent",
        crosshairMarkerRadius: 4,
        crosshairMarkerBorderColor: lineColor,
        crosshairMarkerBackgroundColor: "#1a1f2e",
      });
      series.setData(formatted.map(d => ({ time: d.time, value: d.close })));
    }

    // Volume
    const volSeries = chart.addHistogramSeries({
      priceFormat: { type: "volume" },
      priceScaleId: "vol",
      color: "#2a2f3e",
    });
    chart.priceScale("vol").applyOptions({
      scaleMargins: { top: 0.88, bottom: 0 },
    });
    volSeries.setData(formatted.map(d => ({
      time: d.time,
      value: data.find(x => x.time.split("T")[0] === d.time)?.volume || 0,
      color: d.close >= d.open ? "rgba(34, 197, 94, 0.15)" : "rgba(239, 68, 68, 0.15)",
    })));

    chart.timeScale().fitContent();

    const onResize = () => {
      if (containerRef.current) chart.applyOptions({ width: containerRef.current.clientWidth });
    };
    window.addEventListener("resize", onResize);
    return () => { window.removeEventListener("resize", onResize); chart.remove(); chartRef.current = null; };
  }, [data, height, mode]);

  return <div ref={containerRef} className="w-full" />;
}
