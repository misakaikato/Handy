import React from "react";
import { useTranslation } from "react-i18next";
import { Slider } from "../ui/Slider";
import { useSettings } from "../../hooks/useSettings";
import { useOsType } from "../../hooks/useOsType";

interface CursorContextCharsProps {
  descriptionMode?: "inline" | "tooltip";
  grouped?: boolean;
}

/**
 * Per-side character limit of the cursor context passed to post-processing.
 * macOS only, matching the feature it configures.
 */
export const CursorContextChars: React.FC<CursorContextCharsProps> = React.memo(
  ({ descriptionMode = "tooltip", grouped = false }) => {
    const osType = useOsType();
    const { t } = useTranslation();
    const { settings, updateSetting, resetSetting, isUpdating } = useSettings();

    if (osType !== "macos") {
      return null;
    }

    return (
      <Slider
        value={settings?.cursor_context_chars ?? 500}
        onChange={(value: number) =>
          updateSetting("cursor_context_chars", value)
        }
        onReset={() => resetSetting("cursor_context_chars")}
        isResetting={isUpdating("cursor_context_chars")}
        min={50}
        max={5000}
        step={50}
        label={t("settings.advanced.cursorContextChars.title")}
        description={t("settings.advanced.cursorContextChars.description")}
        descriptionMode={descriptionMode}
        grouped={grouped}
        formatValue={(v) => `${v}`}
      />
    );
  },
);
CursorContextChars.displayName = "CursorContextChars";
