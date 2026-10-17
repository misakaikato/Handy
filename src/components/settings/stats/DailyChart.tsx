import React from "react";
import { useTranslation } from "react-i18next";
import type { DailyStat } from "@/bindings";

interface DailyChartProps {
  daily: DailyStat[];
}

const VIEW_WIDTH = 640;
const VIEW_HEIGHT = 180;
const PADDING_LEFT = 8;
const PADDING_RIGHT = 8;
const PADDING_TOP = 16;
const AXIS_HEIGHT = 18;
const PLOT_HEIGHT = VIEW_HEIGHT - PADDING_TOP - AXIS_HEIGHT;

/**
 * Compact daily-activity chart: one bar per day for characters dictated, with
 * an overlaid line for average total latency. Pure SVG, no chart dependency;
 * each series is normalized to its own maximum.
 */
export const DailyChart: React.FC<DailyChartProps> = ({ daily }) => {
  const { t, i18n } = useTranslation();

  const n = daily.length;
  if (n === 0) return null;

  const maxChars = Math.max(...daily.map((d) => d.characters), 1);
  const latencies = daily
    .filter((d) => d.dictations > 0)
    .map((d) => d.avg_total_ms);
  const maxLatency = Math.max(...latencies, 1);
  const hasLatency = latencies.length > 0;

  const plotWidth = VIEW_WIDTH - PADDING_LEFT - PADDING_RIGHT;
  const barSpace = plotWidth / n;
  const barWidth = Math.max(2, Math.min(barSpace * 0.7, 28));
  const barX = (i: number) =>
    PADDING_LEFT + barSpace * i + (barSpace - barWidth) / 2;
  const barY = (chars: number) =>
    PADDING_TOP + PLOT_HEIGHT * (1 - chars / maxChars);
  const barH = (chars: number) => (PLOT_HEIGHT * chars) / maxChars;

  const linePoint = (i: number, avgMs: number) => ({
    x: PADDING_LEFT + barSpace * i + barSpace / 2,
    y: PADDING_TOP + PLOT_HEIGHT * (1 - avgMs / maxLatency),
  });
  const linePoints = daily
    .map((d, i) => (d.dictations > 0 ? linePoint(i, d.avg_total_ms) : null))
    .filter((p): p is { x: number; y: number } => p !== null);
  const polyline = linePoints.map((p) => `${p.x},${p.y}`).join(" ");

  const labelEvery = Math.max(1, Math.ceil(n / 6));
  const labelFormatter = new Intl.DateTimeFormat(i18n.language, {
    month: "numeric",
    day: "numeric",
  });

  return (
    <div className="px-4 py-3">
      <div className="flex items-center gap-4 mb-2 text-xs text-mid-gray">
        <span className="inline-flex items-center gap-1.5">
          <span
            className="inline-block w-3 h-3 rounded-sm"
            style={{ background: "var(--color-logo-primary)" }}
          />
          {t("settings.stats.chart.characters")}
        </span>
        {hasLatency && (
          <span className="inline-flex items-center gap-1.5">
            <span
              className="inline-block w-3 h-0.5"
              style={{ background: "var(--color-text)" }}
            />
            {t("settings.stats.chart.latency")}
          </span>
        )}
      </div>
      <svg
        viewBox={`0 0 ${VIEW_WIDTH} ${VIEW_HEIGHT}`}
        className="w-full h-auto"
        role="img"
        aria-label={t("settings.stats.chart.title")}
      >
        <line
          x1={PADDING_LEFT}
          y1={PADDING_TOP + PLOT_HEIGHT}
          x2={VIEW_WIDTH - PADDING_RIGHT}
          y2={PADDING_TOP + PLOT_HEIGHT}
          style={{ stroke: "var(--color-mid-gray)", strokeOpacity: 0.4 }}
        />
        {daily.map((d, i) => (
          <rect
            key={d.date}
            x={barX(i)}
            y={barY(d.characters)}
            width={barWidth}
            height={Math.max(barH(d.characters), d.characters > 0 ? 2 : 0)}
            rx={1.5}
            style={{
              fill: "var(--color-logo-primary)",
              fillOpacity: 0.75,
            }}
          >
            <title>{`${d.date}: ${d.characters} / ${Math.round(d.avg_total_ms)}ms`}</title>
          </rect>
        ))}
        {linePoints.length > 1 && (
          <polyline
            points={polyline}
            fill="none"
            style={{
              stroke: "var(--color-text)",
              strokeWidth: 1.5,
              strokeOpacity: 0.8,
            }}
          />
        )}
        {daily.map((d, i) =>
          i % labelEvery === 0 || i === n - 1 ? (
            <text
              key={`label-${d.date}`}
              x={PADDING_LEFT + barSpace * i + barSpace / 2}
              y={VIEW_HEIGHT - 4}
              textAnchor="middle"
              style={{ fill: "var(--color-mid-gray)", fontSize: 10 }}
            >
              {labelFormatter.format(new Date(`${d.date}T00:00:00`))}
            </text>
          ) : null,
        )}
      </svg>
    </div>
  );
};
