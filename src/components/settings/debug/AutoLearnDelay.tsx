import React from "react";
import { useTranslation } from "react-i18next";
import { Slider } from "../../ui/Slider";
import { useSettings } from "../../../hooks/useSettings";
import { useOsType } from "../../../hooks/useOsType";

type AutoLearnTimingKey =
  | "auto_learn_delay_ms"
  | "auto_learn_notification_duration_ms";

interface AutoLearnTimingProps {
  descriptionMode?: "inline" | "tooltip";
  grouped?: boolean;
  settingKey?: AutoLearnTimingKey;
  labelKey?: string;
  descriptionKey?: string;
}

/**
 * Debug-gated sliders for the auto-learn timing: the wait before reading the
 * focused field, and how long the learned-words toast stays visible.
 * macOS only, matching the feature.
 */
export const AutoLearnDelay: React.FC<AutoLearnTimingProps> = ({
  descriptionMode = "tooltip",
  grouped = false,
  settingKey = "auto_learn_delay_ms",
  labelKey = "settings.debug.autoLearnDelay.title",
  descriptionKey = "settings.debug.autoLearnDelay.description",
}) => {
  const osType = useOsType();
  const { t } = useTranslation();
  const { settings, updateSetting, resetSetting, isUpdating } = useSettings();

  if (osType !== "macos") {
    return null;
  }

  return (
    <Slider
      value={
        settings?.[settingKey] ??
        (settingKey === "auto_learn_delay_ms" ? 8000 : 5000)
      }
      onChange={(value: number) => updateSetting(settingKey, value)}
      onReset={() => resetSetting(settingKey)}
      isResetting={isUpdating(settingKey)}
      min={1000}
      max={30000}
      step={500}
      label={t(labelKey)}
      description={t(descriptionKey)}
      descriptionMode={descriptionMode}
      grouped={grouped}
      formatValue={(v) => `${(v / 1000).toFixed(1)}s`}
    />
  );
};
