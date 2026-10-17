import React from "react";
import { useTranslation } from "react-i18next";
import { ToggleSwitch } from "../ui/ToggleSwitch";
import { useSettings } from "../../hooks/useSettings";

interface StreamPostProcessProps {
  descriptionMode?: "inline" | "tooltip";
  grouped?: boolean;
}

/**
 * Streaming rewrite output: when enabled, the LLM rewrite is requested with
 * stream:true (SSE) and typed at the cursor incrementally as chunks arrive,
 * instead of being pasted once the full response is in.
 */
export const StreamPostProcess: React.FC<StreamPostProcessProps> = React.memo(
  ({ descriptionMode = "tooltip", grouped = false }) => {
    const { t } = useTranslation();
    const { getSetting, updateSetting, isUpdating } = useSettings();

    const enabled = getSetting("stream_post_process") ?? false;

    return (
      <ToggleSwitch
        checked={enabled}
        onChange={(nextEnabled) =>
          updateSetting("stream_post_process", nextEnabled)
        }
        isUpdating={isUpdating("stream_post_process")}
        label={t("settings.postProcessing.streaming.title")}
        description={t("settings.postProcessing.streaming.description")}
        descriptionMode={descriptionMode}
        grouped={grouped}
      />
    );
  },
);
StreamPostProcess.displayName = "StreamPostProcess";
