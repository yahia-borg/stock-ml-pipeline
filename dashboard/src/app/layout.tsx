import type { Metadata } from "next";
import { Providers } from "./providers";
import "./globals.css";

export const metadata: Metadata = {
  title: "StockVision AI",
  description: "ML-powered stock forecasting",
};

const NAV = [
  { href: "/", label: "Home" },
  { href: "/models", label: "Models" },
  { href: "/sentiment", label: "Sentiment" },
  { href: "/pipeline", label: "Pipeline" },
];

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="min-h-screen font-sans">
        <Providers>
          <div className="flex min-h-screen">
            {/* Sidebar */}
            <aside className="w-[180px] min-h-screen flex flex-col py-6 px-4 border-r border-white/[0.06]">
              <div className="w-10 h-10 rounded-xl bg-[hsl(var(--secondary))] flex items-center justify-center mb-8">
                <svg className="w-5 h-5 text-emerald-400" fill="currentColor" viewBox="0 0 20 20">
                  <path d="M12 7a1 1 0 110-2h5a1 1 0 011 1v5a1 1 0 11-2 0V8.414l-4.293 4.293a1 1 0 01-1.414 0L8 10.414l-4.293 4.293a1 1 0 01-1.414-1.414l5-5a1 1 0 011.414 0L11 10.586 14.586 7H12z" />
                </svg>
              </div>

              <nav className="flex-1 flex flex-col gap-1">
                {NAV.map((item) => (
                  <a key={item.href} href={item.href}
                    className="px-3 py-2 rounded-xl text-[13px] text-[hsl(var(--muted-foreground))] hover:text-white hover:bg-white/[0.04] transition-all duration-150">
                    {item.label}
                  </a>
                ))}
              </nav>

              <div className="text-[10px] text-[hsl(var(--muted-foreground))] px-3">
                StockVision AI
              </div>
            </aside>

            {/* Main */}
            <main className="flex-1 overflow-auto">
              <div className="max-w-[1360px] mx-auto px-6 py-5">
                {children}
              </div>
            </main>
          </div>
        </Providers>
      </body>
    </html>
  );
}
