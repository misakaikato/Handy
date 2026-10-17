//! Learns custom words from the corrections users make after Handy pastes.
//!
//! Flow: after a successful paste, wait `auto_learn_delay_ms`, read the
//! focused text field's value through the macOS Accessibility API, locate
//! the pasted region inside it, and diff it against what was pasted.
//! Replacement spans that look like a term fix — a short Han span whose
//! toneless pinyin matches, or a Latin token close by edit distance/Soundex
//! — are added to the `custom_words` setting so future transcriptions (and
//! the post-processing prompt glossary) use them.
//!
//! Everything that can fail (no accessibility permission, no focused text
//! field, the paste no longer present, an unsupported target app) results in
//! a silent skip: auto-learning must never disturb the user.
//!
//! The pure logic (region location, diffing, acceptance) lives in
//! [`logic`], is platform independent and unit-tested on every platform;
//! only the AX glue (crate::ax_reader) is macOS-specific.

#[cfg(any(target_os = "macos", test))]
mod logic;
#[cfg(target_os = "macos")]
mod notify;

#[cfg(target_os = "macos")]
use crate::settings::{get_settings, write_settings, PasteMethod};
#[cfg(target_os = "macos")]
use log::{debug, info};
use tauri::AppHandle;

#[cfg(target_os = "macos")]
use tauri::Emitter;

/// Entry point called right after a successful paste. When auto-learning is
/// enabled, waits on a background thread and compares the focused field
/// against what was pasted. Never blocks or errors into the paste path;
/// non-macOS platforms silently do nothing (macOS first).
pub fn schedule_after_paste(app: &AppHandle, pasted_text: String) {
    #[cfg(not(target_os = "macos"))]
    {
        let _ = (app, pasted_text);
    }

    #[cfg(target_os = "macos")]
    {
        let settings = get_settings(app);
        if !settings.auto_learn_custom_words || settings.paste_method == PasteMethod::None {
            return;
        }
        let pasted_text = pasted_text.trim().to_string();
        if pasted_text.is_empty() {
            return;
        }

        let delay =
            std::time::Duration::from_millis(settings.auto_learn_delay_ms.clamp(500, 60_000));
        let app = app.clone();
        std::thread::spawn(move || {
            std::thread::sleep(delay);
            learn_from_correction(&app, &pasted_text);
        });
    }
}

/// Reads the focused field, extracts learnable corrections and persists them.
#[cfg(target_os = "macos")]
fn learn_from_correction(app: &AppHandle, pasted: &str) {
    let Some(field_value) = crate::ax_reader::focused_field_value() else {
        debug!("Auto-learn: no readable focused text field; skipping");
        return;
    };

    let mut settings = get_settings(app);
    let learned = logic::extract_learned_words(pasted, &field_value, &settings.custom_words);
    if learned.is_empty() {
        return;
    }

    info!("Auto-learned custom words: {learned:?}");
    settings.custom_words.extend(learned.clone());
    write_settings(app, settings);

    // Let an open settings window pick up the new words.
    let _ = app.emit(
        "settings-changed",
        serde_json::json!({ "setting": "custom_words" }),
    );

    notify::show_learned_toast(app, learned);
}

/// Creates the learned-words toast window (hidden). No-op outside macOS,
/// where auto-learning is not wired up yet.
pub fn create_learned_toast(app: &AppHandle) {
    #[cfg(target_os = "macos")]
    notify::create_learned_toast(app);
    #[cfg(not(target_os = "macos"))]
    {
        let _ = app;
    }
}

/// Removes the words added by the given learn event from `custom_words`,
/// persists the change and hides the toast. Returns whether an event with
/// this id was still pending; always false outside macOS.
pub fn undo_last_learned(app: &AppHandle, event_id: u64) -> bool {
    #[cfg(target_os = "macos")]
    {
        notify::undo_last_learned(app, event_id)
    }
    #[cfg(not(target_os = "macos"))]
    {
        let _ = (app, event_id);
        false
    }
}
