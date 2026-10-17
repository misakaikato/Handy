import React, { useMemo } from "react";
import { useTranslation } from "react-i18next";
import type { PostProcessThinking } from "@/bindings";
import { useSettings } from "@/hooks/useSettings";
import { Dropdown, type DropdownOption } from "@/components/ui/Dropdown";
import { SettingContainer } from "@/components/ui/SettingContainer";

interface ThinkingControlProps {
  descriptionMode?: "tooltip" | "inline";
  grouped?: boolean;
}

/**
 * Thinking control for the post-processing model: Default sends nothing, Off
 * disables thinking (chat_template_kwargs / enable_thinking), and
 * Low/Medium/High request a reasoning effort. Applies to OpenAI-compatible
 * (including custom) providers.
 */
export const ThinkingControl: React.FC<ThinkingControlProps> = ({
  descriptionMode = "tooltip",
  grouped = false,
}) => {
  const { t } = useTranslation();
  const { getSetting, updateSetting, isUpdating } = useSettings();
  const selected = getSetting("post_process_thinking") ?? "default";

  const options = useMemo<DropdownOption[]>(
    () =>
      (["default", "off", "low", "medium", "high"] as const).map((value) => ({
        value,
        label: t(`settings.postProcessing.thinking.options.${value}`),
      })),
    [t],
  );

  return (
    <SettingContainer
      title={t("settings.postProcessing.thinking.title")}
      description={t("settings.postProcessing.thinking.description")}
      descriptionMode={descriptionMode}
      grouped={grouped}
      layout="horizontal"
    >
      <Dropdown
        options={options}
        selectedValue={selected}
        onSelect={(value) =>
          updateSetting("post_process_thinking", value as PostProcessThinking)
        }
        disabled={isUpdating("post_process_thinking")}
      />
    </SettingContainer>
  );
};
