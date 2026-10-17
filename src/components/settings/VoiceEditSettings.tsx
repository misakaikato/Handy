import React, { useEffect, useState } from "react";
import { useTranslation } from "react-i18next";
import { useSettings } from "../../hooks/useSettings";
import { ToggleSwitch } from "../ui/ToggleSwitch";
import { SettingContainer } from "../ui/SettingContainer";
import { Textarea } from "@/components/ui";
import { Button } from "../ui/Button";
import { ResetButton } from "../ui/ResetButton";
import { ShortcutInput } from "./ShortcutInput";

/**
 * Voice edit of selected text: a dedicated shortcut that captures the
 * selection, records a dictated instruction, has the post-processing LLM
 * rewrite the selection, and pastes the result over it. The prompt is
 * user-editable with ${selected} / ${instruction} placeholders.
 */
export const VoiceEditSettings: React.FC = () => {
  const { t } = useTranslation();
  const { getSetting, updateSetting, resetSetting, isUpdating } = useSettings();
  const [draftPrompt, setDraftPrompt] = useState("");

  const enabled = getSetting("voice_edit_enabled") ?? false;
  const savedPrompt = getSetting("voice_edit_prompt") ?? "";

  useEffect(() => {
    setDraftPrompt(savedPrompt);
  }, [savedPrompt]);

  const isDirty = draftPrompt.trim() !== savedPrompt.trim();

  const handleSavePrompt = () => {
    updateSetting("voice_edit_prompt", draftPrompt);
  };

  const handleResetPrompt = () => {
    resetSetting("voice_edit_prompt");
  };

  return (
    <>
      <ToggleSwitch
        checked={enabled}
        onChange={(nextEnabled) =>
          updateSetting("voice_edit_enabled", nextEnabled)
        }
        isUpdating={isUpdating("voice_edit_enabled")}
        label={t("settings.postProcessing.voiceEdit.title")}
        description={t("settings.postProcessing.voiceEdit.description")}
        descriptionMode="tooltip"
        grouped={true}
      />

      {enabled && (
        <>
          <SettingContainer
            title={t("settings.postProcessing.voiceEdit.shortcut.title")}
            description={t(
              "settings.postProcessing.voiceEdit.shortcut.description",
            )}
            descriptionMode="tooltip"
            layout="horizontal"
            grouped={true}
          >
            <ShortcutInput
              shortcutId="voice_edit"
              descriptionMode="tooltip"
              grouped={true}
            />
          </SettingContainer>

          <SettingContainer
            title={t("settings.postProcessing.voiceEdit.prompt.title")}
            description={t(
              "settings.postProcessing.voiceEdit.prompt.description",
            )}
            descriptionMode="tooltip"
            layout="stacked"
            grouped={true}
          >
            <div className="space-y-3">
              <Textarea
                value={draftPrompt}
                onChange={(e) => setDraftPrompt(e.target.value)}
                placeholder={t(
                  "settings.postProcessing.voiceEdit.prompt.placeholder",
                )}
              />
              <div className="flex gap-2">
                <Button
                  onClick={handleSavePrompt}
                  variant="primary"
                  size="md"
                  disabled={
                    !draftPrompt.trim() ||
                    !isDirty ||
                    isUpdating("voice_edit_prompt")
                  }
                >
                  {t("settings.postProcessing.voiceEdit.prompt.save")}
                </Button>
                <ResetButton
                  onClick={handleResetPrompt}
                  disabled={isUpdating("voice_edit_prompt")}
                  ariaLabel={t(
                    "settings.postProcessing.voiceEdit.prompt.reset",
                  )}
                />
              </div>
            </div>
          </SettingContainer>
        </>
      )}
    </>
  );
};
