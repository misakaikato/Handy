import React from "react";
import { useTranslation } from "react-i18next";
import { ToggleSwitch } from "../ui/ToggleSwitch";
import { useSettings } from "../../hooks/useSettings";
import { useOsType } from "../../hooks/useOsType";

interface CursorContextProps {
  descriptionMode?: "inline" | "tooltip";
  grouped?: boolean;
}

/**
 * Cursor context for post-processing: before the LLM rewrite, read the text
 * around the cursor in the focused field (macOS Accessibility API) and pass
 * it to the model as reference-only context. macOS only; password fields are
 * always skipped and any read failure is a silent skip.
 */
export const CursorContext: React.FC<CursorContextProps> = React.memo(
  ({ descriptionMode = "tooltip", grouped = false }) => {
    const osType = useOsType();
    const { t } = useTranslation();
    const { getSetting, updateSetting, isUpdating } = useSettings();

    if (osType !== "macos") {
      return null;
    }

    const enabled = getSetting("use_cursor_context") ?? false;

    return (
      <ToggleSwitch
        checked={enabled}
        onChange={(nextEnabled) =>
          updateSetting("use_cursor_context", nextEnabled)
        }
        isUpdating={isUpdating("use_cursor_context")}
        label={t("settings.advanced.cursorContext.title")}
        description={t("settings.advanced.cursorContext.description")}
        descriptionMode={descriptionMode}
        grouped={grouped}
      />
    );
  },
);
CursorContext.displayName = "CursorContext";
