import React, { useState } from "react";
import { open, save } from "@tauri-apps/plugin-dialog";
import { useTranslation } from "react-i18next";
import { toast } from "sonner";
import { commands, type WordlistPreview } from "@/bindings";
import { useSettings } from "../../hooks/useSettings";
import { Input } from "../ui/Input";
import { Button } from "../ui/Button";
import { SettingContainer } from "../ui/SettingContainer";
interface CustomWordsProps {
  descriptionMode?: "inline" | "tooltip";
  grouped?: boolean;
}

const normalizeCustomWord = (word: string) =>
  word
    .replace(/[<>"']/g, "")
    .replace(/\s+/g, " ")
    .trim();

interface ImportPreviewState {
  fileName: string;
  preview: WordlistPreview;
}

export const CustomWords: React.FC<CustomWordsProps> = React.memo(
  ({ descriptionMode = "tooltip", grouped = false }) => {
    const { t } = useTranslation();
    const { getSetting, updateSetting, isUpdating, refreshSettings } =
      useSettings();
    const [newWord, setNewWord] = useState("");
    const [importPreview, setImportPreview] =
      useState<ImportPreviewState | null>(null);
    const [importBusy, setImportBusy] = useState(false);
    const customWords = getSetting("custom_words") || [];
    const normalizedWord = normalizeCustomWord(newWord);

    const handleAddWord = () => {
      if (normalizedWord && normalizedWord.length <= 50) {
        if (customWords.includes(normalizedWord)) {
          toast.error(
            t("settings.advanced.customWords.duplicate", {
              word: normalizedWord,
            }),
          );
          return;
        }
        updateSetting("custom_words", [...customWords, normalizedWord]);
        setNewWord("");
      }
    };

    const handleRemoveWord = (wordToRemove: string) => {
      updateSetting(
        "custom_words",
        customWords.filter((word) => word !== wordToRemove),
      );
    };

    const handleKeyPress = (e: React.KeyboardEvent) => {
      if (e.key === "Enter") {
        e.preventDefault();
        handleAddWord();
      }
    };

    const handleImport = async () => {
      let path: string | null = null;
      try {
        path = await open({
          multiple: false,
          directory: false,
          filters: [
            {
              name: "Word list",
              extensions: ["txt", "csv", "json"],
            },
          ],
        });
      } catch (error) {
        console.error("Import dialog failed:", error);
      }
      if (!path) return;

      const result = await commands.importWordlistPreview(path);
      if (result.status !== "ok") {
        toast.error(t("settings.advanced.customWords.importError"));
        return;
      }
      if (result.data.words.length === 0) {
        toast.info(t("settings.advanced.customWords.importEmpty"));
        return;
      }
      const fileName = path.split(/[\\/]/).pop() ?? path;
      setImportPreview({ fileName, preview: result.data });
    };

    const handleApplyImport = async (replace: boolean) => {
      if (!importPreview) return;
      const { preview } = importPreview;
      setImportBusy(true);
      try {
        const result = await commands.applyWordlistImport(
          preview.words,
          replace,
        );
        if (result.status !== "ok") {
          throw new Error(String(result.error));
        }
        await refreshSettings();
        toast.success(
          t("settings.advanced.customWords.importSuccess", {
            count: replace ? preview.words.length : preview.new_count,
          }),
        );
        setImportPreview(null);
      } catch (error) {
        console.error("Failed to apply import:", error);
        toast.error(t("settings.advanced.customWords.importError"));
      } finally {
        setImportBusy(false);
      }
    };

    const handleExport = async (format: "txt" | "json") => {
      let path: string | null = null;
      try {
        path = await save({
          defaultPath: `handy-custom-words.${format}`,
          filters: [
            format === "json"
              ? { name: "JSON", extensions: ["json"] }
              : { name: "Text", extensions: ["txt"] },
          ],
        });
      } catch (error) {
        console.error("Export dialog failed:", error);
      }
      if (!path) return;

      const result = await commands.exportWordlist(path, format);
      if (result.status === "ok") {
        toast.success(
          t("settings.advanced.customWords.exportSuccess", {
            count: result.data,
          }),
        );
      } else {
        toast.error(t("settings.advanced.customWords.exportError"));
      }
    };

    return (
      <>
        <SettingContainer
          title={t("settings.advanced.customWords.title")}
          description={t("settings.advanced.customWords.description")}
          descriptionMode={descriptionMode}
          grouped={grouped}
        >
          <div className="flex items-center gap-2">
            <Input
              type="text"
              className="max-w-40"
              value={newWord}
              onChange={(e) => setNewWord(e.target.value)}
              onKeyDown={handleKeyPress}
              placeholder={t("settings.advanced.customWords.placeholder")}
              variant="compact"
              disabled={isUpdating("custom_words")}
            />
            <Button
              onClick={handleAddWord}
              disabled={
                !normalizedWord ||
                normalizedWord.length > 50 ||
                isUpdating("custom_words")
              }
              variant="primary"
              size="md"
            >
              {t("settings.advanced.customWords.add")}
            </Button>
          </div>
        </SettingContainer>
        <div
          className={`px-4 p-2 ${
            grouped ? "" : "rounded-lg border border-mid-gray/20"
          } flex flex-wrap items-center gap-2`}
        >
          <Button
            onClick={handleImport}
            disabled={importBusy || isUpdating("custom_words")}
            variant="secondary"
            size="sm"
          >
            {t("settings.advanced.customWords.import")}
          </Button>
          <Button
            onClick={() => handleExport("txt")}
            disabled={customWords.length === 0 || isUpdating("custom_words")}
            variant="secondary"
            size="sm"
          >
            {t("settings.advanced.customWords.exportTxt")}
          </Button>
          <Button
            onClick={() => handleExport("json")}
            disabled={customWords.length === 0 || isUpdating("custom_words")}
            variant="secondary"
            size="sm"
          >
            {t("settings.advanced.customWords.exportJson")}
          </Button>
        </div>
        {importPreview && (
          <div
            className={`px-4 p-3 ${
              grouped ? "" : "rounded-lg border border-mid-gray/20"
            } flex flex-col gap-3`}
          >
            <p className="text-sm">
              {t("settings.advanced.customWords.importSummary", {
                file: importPreview.fileName,
                total: importPreview.preview.words.length,
                new: importPreview.preview.new_count,
                duplicate:
                  importPreview.preview.words.length -
                  importPreview.preview.new_count,
              })}
            </p>
            <div className="flex flex-wrap gap-2">
              <Button
                onClick={() => handleApplyImport(false)}
                disabled={importBusy || isUpdating("custom_words")}
                variant="primary"
                size="sm"
              >
                {t("settings.advanced.customWords.importMerge")}
              </Button>
              <Button
                onClick={() => handleApplyImport(true)}
                disabled={importBusy || isUpdating("custom_words")}
                variant="secondary"
                size="sm"
              >
                {t("settings.advanced.customWords.importReplace")}
              </Button>
              <Button
                onClick={() => setImportPreview(null)}
                disabled={importBusy}
                variant="ghost"
                size="sm"
              >
                {t("settings.advanced.customWords.importCancel")}
              </Button>
            </div>
          </div>
        )}
        {customWords.length > 0 && (
          <div
            className={`px-4 p-2 ${
              grouped ? "" : "rounded-lg border border-mid-gray/20"
            } flex flex-wrap gap-1`}
          >
            {customWords.map((word) => (
              <Button
                key={word}
                onClick={() => handleRemoveWord(word)}
                disabled={isUpdating("custom_words")}
                variant="secondary"
                size="sm"
                className="inline-flex items-center gap-1 cursor-pointer"
                aria-label={t("settings.advanced.customWords.remove", { word })}
              >
                <span>{word}</span>
                <svg
                  className="w-3 h-3"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M6 18L18 6M6 6l12 12"
                  />
                </svg>
              </Button>
            ))}
          </div>
        )}
      </>
    );
  },
);
