import React, { useEffect, useState } from "react";
import { listen } from "@tauri-apps/api/event";
import { useTranslation } from "react-i18next";
import { commands } from "@/bindings";
import "./AutoLearnToast.css";

/** Payload of the backend's `learned-words` event. */
interface LearnedWordsEvent {
  id: number;
  words: string[];
}

/**
 * Toast shown after auto-learning: lists the learned words with an Undo
 * button. Visibility is driven entirely by the backend (it shows the window,
 * emits this event and hides it again after the configured duration).
 */
const AutoLearnToast: React.FC = () => {
  const { t } = useTranslation();
  const [learned, setLearned] = useState<LearnedWordsEvent | null>(null);

  useEffect(() => {
    const unlisten = listen<LearnedWordsEvent>("learned-words", (event) => {
      setLearned(event.payload);
    });
    return () => {
      unlisten.then((fn) => fn());
    };
  }, []);

  if (!learned) {
    return null;
  }

  const handleUndo = async () => {
    const eventId = learned.id;
    setLearned(null);
    try {
      const result = await commands.undoAutoLearn(eventId);
      if (result.status === "error") {
        console.error("Failed to undo auto-learn:", result.error);
      }
    } catch (error) {
      console.error("Failed to undo auto-learn:", error);
    }
  };

  return (
    <div className="toast-card" role="status">
      <span className="toast-words">
        {t("autoLearn.learned", {
          words: learned.words.join(t("autoLearn.separator")),
        })}
      </span>
      <button className="toast-undo" onClick={handleUndo}>
        {t("autoLearn.undo")}
      </button>
    </div>
  );
};

export default AutoLearnToast;
