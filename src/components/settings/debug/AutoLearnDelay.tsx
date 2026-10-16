import React from "react";
import { useTranslation } from "react-i18next";
import { Slider } from "../../ui/Slider";
import { useSettings } from "../../../hooks/useSettings";
import { useOsType } from "../../../hooks/useOsType";

/**
 * Delay before the backend reads the focused text field looking for
 * corrections to learn. Debug-gated companion of the auto-learn toggle.
 */
export const AutoLearnDelay: React.FC<{
  descriptionMode?: "inline" | "tooltip";
  grouped?: boolean;
}> = ({ descriptionMode = "tooltip", grouped = false }) => {
  const osType = useOsType();
  const { t } = useTranslation();
  const { settings, updateSetting, resetSetting, isUpdating } = useSettings();

  if (osType !== "macos") {
    return null;
  }

  return (
    <Slider
      value={settings?.auto_learn_delay_ms ?? 8000}
      onChange={(value: number) => updateSetting("auto_learn_delay_ms", value)}
      onReset={() => resetSetting("auto_learn_delay_ms")}
      isResetting={isUpdating("auto_learn_delay_ms")}
      min={1000}
      max={30000}
      step={500}
      label={t("settings.debug.autoLearnDelay.title")}
      description={t("settings.debug.autoLearnDelay.description")}
      descriptionMode={descriptionMode}
      grouped={grouped}
      formatValue={(v) => `${(v / 1000).toFixed(1)}s`}
    />
  );
};
