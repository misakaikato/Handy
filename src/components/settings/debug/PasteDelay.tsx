import React from "react";
import { useTranslation } from "react-i18next";
import { Slider } from "../../ui/Slider";
import { useSettings } from "../../../hooks/useSettings";

type PasteDelayKey =
  | "paste_delay_ms"
  | "paste_delay_after_ms"
  | "paste_modifier_hold_ms";

// Mirrors the Rust `default_*_paste_*` fns so the slider shows the effective
// value before settings have loaded.
const PASTE_DELAY_DEFAULTS: Record<PasteDelayKey, number> = {
  paste_delay_ms: 50,
  paste_delay_after_ms: 50,
  paste_modifier_hold_ms: 80,
};

interface PasteDelayProps {
  descriptionMode?: "tooltip" | "inline";
  grouped?: boolean;
  settingKey?: PasteDelayKey;
  labelKey?: string;
  descriptionKey?: string;
}

export const PasteDelay: React.FC<PasteDelayProps> = ({
  descriptionMode = "tooltip",
  grouped = false,
  settingKey = "paste_delay_ms",
  labelKey = "settings.debug.pasteDelay.title",
  descriptionKey = "settings.debug.pasteDelay.description",
}) => {
  const { t } = useTranslation();
  const { settings, updateSetting, resetSetting, isUpdating } = useSettings();

  const handleDelayChange = (value: number) => {
    updateSetting(settingKey, value);
  };

  return (
    <Slider
      value={settings?.[settingKey] ?? PASTE_DELAY_DEFAULTS[settingKey]}
      onChange={handleDelayChange}
      onReset={() => resetSetting(settingKey)}
      isResetting={isUpdating(settingKey)}
      min={0}
      max={5000}
      step={10}
      label={t(labelKey)}
      description={t(descriptionKey)}
      descriptionMode={descriptionMode}
      grouped={grouped}
      formatValue={(v) => `${v}ms`}
    />
  );
};
