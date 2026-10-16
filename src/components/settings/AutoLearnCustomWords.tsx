import React from "react";
import { useTranslation } from "react-i18next";
import { ToggleSwitch } from "../ui/ToggleSwitch";
import { useSettings } from "../../hooks/useSettings";
import { useOsType } from "../../hooks/useOsType";

interface AutoLearnCustomWordsProps {
  descriptionMode?: "inline" | "tooltip";
  grouped?: boolean;
}

/**
 * Auto-learning of custom words from the user's manual corrections after a
 * paste. macOS only for now: the backend reads the focused text field
 * through the Accessibility API, which no other platform implements yet.
 */
export const AutoLearnCustomWords: React.FC<AutoLearnCustomWordsProps> =
  React.memo(({ descriptionMode = "tooltip", grouped = false }) => {
    const osType = useOsType();
    const { t } = useTranslation();
    const { getSetting, updateSetting, isUpdating } = useSettings();

    if (osType !== "macos") {
      return null;
    }

    const enabled = getSetting("auto_learn_custom_words") ?? false;

    return (
      <ToggleSwitch
        checked={enabled}
        onChange={(nextEnabled) =>
          updateSetting("auto_learn_custom_words", nextEnabled)
        }
        isUpdating={isUpdating("auto_learn_custom_words")}
        label={t("settings.advanced.autoLearnCustomWords.title")}
        description={t("settings.advanced.autoLearnCustomWords.description")}
        descriptionMode={descriptionMode}
        grouped={grouped}
      />
    );
  });
