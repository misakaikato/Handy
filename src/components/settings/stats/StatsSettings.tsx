import React, { useCallback, useEffect, useState } from "react";
import { save } from "@tauri-apps/plugin-dialog";
import { useTranslation } from "react-i18next";
import { toast } from "sonner";
import {
  commands,
  events,
  type LearnedWordEvent,
  type StatsSummary,
} from "@/bindings";
import { Button } from "../../ui/Button";
import { SettingsGroup } from "../../ui/SettingsGroup";
import { DailyChart } from "./DailyChart";

const RANGES = [1, 7, 30] as const;
type RangeDays = (typeof RANGES)[number];

const LEARNED_WORDS_LIMIT = 10;

type Summaries = Partial<Record<RangeDays, StatsSummary | null>>;

const rangeLabelKey: Record<RangeDays, string> = {
  1: "settings.stats.periods.today",
  7: "settings.stats.periods.week",
  30: "settings.stats.month",
};

const numberFormatter = (locale: string) =>
  new Intl.NumberFormat(locale, { maximumFractionDigits: 0 });

const ms = (value: number, locale: string) =>
  `${numberFormatter(locale).format(Math.round(value))}ms`;

const StatCard: React.FC<{
  label: string;
  value: string;
}> = ({ label, value }) => (
  <div className="flex-1 min-w-28 px-4 py-3 flex flex-col gap-1">
    <p className="text-xs text-mid-gray">{label}</p>
    <p className="text-xl font-medium tabular-nums">{value}</p>
  </div>
);

export const StatsSettings: React.FC = () => {
  const { t, i18n } = useTranslation();
  const [range, setRange] = useState<RangeDays>(7);
  const [summaries, setSummaries] = useState<Summaries>({});
  const [learnedWords, setLearnedWords] = useState<LearnedWordEvent[] | null>(
    null,
  );
  const [loadError, setLoadError] = useState(false);

  const fmt = numberFormatter(i18n.language);

  const load = useCallback(async () => {
    setLoadError(false);
    try {
      const [today, week, month, learned] = await Promise.all([
        commands.getDictationStats(1),
        commands.getDictationStats(7),
        commands.getDictationStats(30),
        commands.getRecentLearnedWords(LEARNED_WORDS_LIMIT),
      ]);
      if (
        today.status !== "ok" ||
        week.status !== "ok" ||
        month.status !== "ok" ||
        learned.status !== "ok"
      ) {
        throw new Error("stats command failed");
      }
      setSummaries({
        1: today.data,
        7: week.data,
        30: month.data,
      });
      setLearnedWords(learned.data);
    } catch (error) {
      console.error("Failed to load stats:", error);
      setLoadError(true);
    }
  }, []);

  useEffect(() => {
    load();
  }, [load]);

  // A new dictation (or a learned-word correction) invalidates everything.
  useEffect(() => {
    const unlisten = events.historyUpdatePayload.listen(() => {
      load();
    });
    return () => {
      unlisten.then((fn) => fn());
    };
  }, [load]);

  const summary = summaries[range];

  const handleExport = async () => {
    let path: string | null = null;
    try {
      path = await save({
        title: t("settings.stats.export"),
        defaultPath: "handy-stats.csv",
        filters: [{ name: "CSV", extensions: ["csv"] }],
      });
    } catch (error) {
      console.error("Save dialog failed:", error);
    }
    if (!path) return;
    const result = await commands.exportStatsCsv(path);
    if (result.status === "ok") {
      toast.success(t("settings.stats.exportSuccess", { count: result.data }));
    } else {
      toast.error(t("settings.stats.exportError"));
    }
  };

  return (
    <div className="max-w-3xl w-full mx-auto space-y-6">
      <div className="flex items-center justify-between px-4">
        <h2 className="text-xs font-medium text-mid-gray uppercase tracking-wide">
          {t("settings.stats.title")}
        </h2>
        <Button variant="secondary" size="sm" onClick={handleExport}>
          {t("settings.stats.export")}
        </Button>
      </div>

      <div className="flex gap-1 px-4">
        {RANGES.map((r) => (
          <Button
            key={r}
            variant={r === range ? "primary-soft" : "secondary"}
            size="sm"
            onClick={() => setRange(r)}
          >
            {t(rangeLabelKey[r])}
          </Button>
        ))}
      </div>

      {loadError && (
        <div className="px-4 py-3 text-sm text-warning">
          {t("settings.stats.loadError")}
        </div>
      )}

      {summary && (
        <>
          <div className="bg-background border border-mid-gray/20 rounded-lg flex flex-wrap divide-x divide-mid-gray/20">
            <StatCard
              label={t("settings.stats.dictations")}
              value={fmt.format(summary.dictations)}
            />
            <StatCard
              label={t("settings.stats.characters")}
              value={fmt.format(summary.characters)}
            />
            <StatCard
              label={t("settings.stats.avgTotalLatency")}
              value={ms(summary.total.avg_ms, i18n.language)}
            />
          </div>

          <SettingsGroup title={t("settings.stats.stageTimings.title")}>
            <div className="px-4 py-3">
              {summary.dictations === 0 ? (
                <p className="text-sm text-text/50 py-2">
                  {t("settings.stats.noData")}
                </p>
              ) : (
                <table className="w-full text-sm">
                  <thead>
                    <tr className="text-xs text-mid-gray text-start">
                      <th className="text-start font-medium pb-2">
                        {t("settings.stats.stageTimings.stage")}
                      </th>
                      <th className="text-end font-medium pb-2">
                        {t("settings.stats.stageTimings.avg")}
                      </th>
                      <th className="text-end font-medium pb-2">
                        {t("settings.stats.stageTimings.p50")}
                      </th>
                      <th className="text-end font-medium pb-2">
                        {t("settings.stats.stageTimings.p90")}
                      </th>
                    </tr>
                  </thead>
                  <tbody className="tabular-nums">
                    {(
                      [
                        ["transcribe", summary.transcribe],
                        ["postProcess", summary.post_process],
                        ["firstToken", summary.first_token],
                        ["output", summary.output],
                        ["total", summary.total],
                      ] as const
                    ).map(([key, stage]) => (
                      <tr key={key} className="border-t border-mid-gray/10">
                        <td className="py-1.5">
                          {t(`settings.stats.stages.${key}`)}
                          {stage.count < summary.dictations && (
                            <span className="text-xs text-mid-gray ml-1.5">
                              ({fmt.format(stage.count)})
                            </span>
                          )}
                        </td>
                        <td className="text-end py-1.5">
                          {stage.count > 0
                            ? ms(stage.avg_ms, i18n.language)
                            : "–"}
                        </td>
                        <td className="text-end py-1.5">
                          {stage.count > 0 ? fmt.format(stage.p50_ms) : "–"}
                        </td>
                        <td className="text-end py-1.5">
                          {stage.count > 0 ? fmt.format(stage.p90_ms) : "–"}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              )}
              {summary.dictations > 0 && (
                <div className="flex gap-4 mt-3 text-xs text-mid-gray">
                  <span>
                    {t("settings.stats.streamedCount", {
                      count: summary.streamed,
                    })}
                  </span>
                  <span>
                    {t("settings.stats.postProcessedCount", {
                      count: summary.post_processed,
                    })}
                  </span>
                </div>
              )}
            </div>
          </SettingsGroup>

          <SettingsGroup title={t("settings.stats.chart.title")}>
            {summary.dictations === 0 ? (
              <p className="px-4 py-3 text-sm text-text/50">
                {t("settings.stats.noData")}
              </p>
            ) : (
              <DailyChart daily={summary.daily} />
            )}
          </SettingsGroup>
        </>
      )}

      <SettingsGroup title={t("settings.stats.learnedWords.title")}>
        <div className="px-4 py-3">
          {!learnedWords || learnedWords.length === 0 ? (
            <p className="text-sm text-text/50 py-2">
              {t("settings.stats.learnedWords.empty")}
            </p>
          ) : (
            <div className="flex flex-wrap gap-1.5">
              {learnedWords.map((event) => (
                <span
                  key={`${event.timestamp}-${event.word}`}
                  className="inline-flex items-center gap-1.5 px-2 py-1 rounded-md bg-mid-gray/10 border border-mid-gray/20 text-sm"
                  title={new Date(event.timestamp * 1000).toLocaleString(
                    i18n.language,
                  )}
                >
                  <span>{event.word}</span>
                  <span className="text-xs text-mid-gray">
                    {new Intl.RelativeTimeFormat(i18n.language, {
                      numeric: "auto",
                    }).format(
                      Math.round(
                        (event.timestamp * 1000 - Date.now()) / 86400000,
                      ),
                      "day",
                    )}
                  </span>
                </span>
              ))}
            </div>
          )}
        </div>
      </SettingsGroup>
    </div>
  );
};
