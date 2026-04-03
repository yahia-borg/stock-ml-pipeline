import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export function formatPercent(value: number): string {
  return `${(value * 100).toFixed(2)}%`;
}

export function formatNumber(value: number, decimals = 2): string {
  return value.toFixed(decimals);
}

export function directionColor(direction: string): string {
  switch (direction) {
    case "up": return "text-green-500";
    case "down": return "text-red-500";
    default: return "text-yellow-500";
  }
}

export function directionBg(direction: string): string {
  switch (direction) {
    case "up": return "bg-green-500/10 text-green-500 border-green-500/20";
    case "down": return "bg-red-500/10 text-red-500 border-red-500/20";
    default: return "bg-yellow-500/10 text-yellow-500 border-yellow-500/20";
  }
}

export function sentimentColor(score: number): string {
  if (score > 0.2) return "text-green-500";
  if (score < -0.2) return "text-red-500";
  return "text-muted-foreground";
}
