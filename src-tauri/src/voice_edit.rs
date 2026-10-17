//! Voice editing of the selected text.
//!
//! A dedicated shortcut (binding id `voice_edit`) turns a spoken instruction
//! into an edit of whatever is selected in the focused app:
//!
//! 1. on trigger, the selection is captured — through the macOS
//!    Accessibility API when available (AXSelectedText, with secure fields
//!    always skipped), otherwise by simulating the platform's copy shortcut
//!    and reading the clipboard, restoring whatever was there before;
//! 2. the user's speech is recorded and transcribed as the instruction;
//! 3. `{selected text + instruction}` goes to the post-processing LLM with a
//!    dedicated, user-editable prompt;
//! 4. the result replaces the selection by pasting over it.
//!
//! Failures are silent skips (no selection, no provider configured, LLM
//! error) so a stray trigger never disturbs the user.

use crate::settings::{get_settings, APPLE_INTELLIGENCE_PROVIDER_ID};
use log::{debug, error};
use once_cell::sync::Lazy;
use std::sync::Mutex;
use std::time::Duration;
use tauri::AppHandle;
use tauri_plugin_clipboard_manager::ClipboardExt;

/// How long to wait after simulating the copy key before reading the
/// clipboard back.
const COPY_SETTLE_DELAY: Duration = Duration::from_millis(200);

/// The selection captured at trigger time, kept until the pipeline's stop
/// consumes it. Voice edit sessions are serialized by the transcription
/// coordinator like any recording, so one slot suffices.
static CAPTURED_SELECTION: Lazy<Mutex<Option<String>>> = Lazy::new(|| Mutex::new(None));

fn store_selection(text: String) {
    if let Ok(mut slot) = CAPTURED_SELECTION.lock() {
        *slot = Some(text);
    }
}

fn take_selection() -> Option<String> {
    CAPTURED_SELECTION
        .lock()
        .ok()
        .and_then(|mut slot| slot.take())
        .filter(|selection| !selection.trim().is_empty())
}

/// Captures and remembers the current selection; returns whether there was
/// one (recording only starts when there was).
pub fn capture_selection_for_recording(app: &AppHandle) -> bool {
    let Some(selection) = read_selected_text(app) else {
        debug!("Voice edit: no selectable text in the focused app; ignoring trigger");
        return false;
    };
    debug!(
        "Voice edit: captured a {}-char selection",
        selection.chars().count()
    );
    store_selection(selection);
    true
}

/// Reads the selected text of the focused app. macOS tries the Accessibility
/// API first (which also lets secure fields be skipped); every platform
/// falls back to simulating the copy shortcut and reading the clipboard,
/// restoring its previous contents afterwards.
fn read_selected_text(app: &AppHandle) -> Option<String> {
    read_selected_text_via_ax().or_else(|| read_selected_text_via_copy(app))
}

/// AX-selected text on macOS: anything else is `None` (which includes the
/// silent-skip failure modes of the AX reader itself).
fn read_selected_text_via_ax() -> Option<String> {
    #[cfg(target_os = "macos")]
    {
        let snapshot = crate::ax_reader::focused_element_snapshot()?;
        if crate::cursor_context::is_secure_text_field(
            snapshot.role.as_deref(),
            snapshot.subrole.as_deref(),
        ) {
            debug!("Voice edit: focused field is a secure text field; ignoring trigger");
            return None;
        }
        snapshot
            .selected_text
            .filter(|text| !text.trim().is_empty())
    }
    #[cfg(not(target_os = "macos"))]
    {
        None
    }
}

/// Copy-shortcut fallback: saves the clipboard, sends Cmd/Ctrl+C, reads the
/// selection back, restores the clipboard. A clipboard that did not change
/// means the copy found no selection.
fn read_selected_text_via_copy(app: &AppHandle) -> Option<String> {
    let clipboard = app.clipboard();
    let saved = clipboard.read_text().ok().filter(|t| !t.is_empty());

    let app_for_copy = app.clone();
    let copied = app
        .run_on_main_thread(move || {
            if let Err(e) = crate::clipboard::send_copy_key(&app_for_copy) {
                debug!("Voice edit: copy-key simulation failed: {e}");
            }
        })
        .is_ok();
    if !copied {
        return None;
    }

    std::thread::sleep(COPY_SETTLE_DELAY);
    let selected = clipboard.read_text().ok().filter(|t| !t.trim().is_empty());

    // Restore whatever was on the clipboard before the probe.
    match &saved {
        Some(previous) => {
            let _ = crate::clipboard::write_text_to_clipboard(app, previous);
        }
        None => {
            let _ = clipboard.clear();
        }
    }

    match selected {
        Some(text) if Some(&text) != saved.as_ref() => Some(text),
        _ => None,
    }
}

/// Builds the voice-edit prompt from the user's template. `${selected}` and
/// `${instruction}` are substituted when present; anything missing is
/// appended as a delimited block so a custom template cannot accidentally
/// drop one of the two inputs.
pub(crate) fn build_voice_edit_prompt(template: &str, selected: &str, instruction: &str) -> String {
    let mut prompt = template.trim().to_string();
    let has_selected = prompt.contains("${selected}");
    let has_instruction = prompt.contains("${instruction}");
    prompt = prompt.replace("${selected}", selected.trim());
    prompt = prompt.replace("${instruction}", instruction.trim());

    if !has_selected {
        prompt.push_str(&format!(
            "\n\n<selected_text>\n{}\n</selected_text>",
            selected.trim()
        ));
    }
    if !has_instruction {
        prompt.push_str(&format!(
            "\n\n<instruction>\n{}\n</instruction>",
            instruction.trim()
        ));
    }
    prompt
}

/// Applies the dictated instruction to the captured selection with the
/// post-processing LLM, returning the edited text to paste over it.
pub(crate) async fn apply_voice_edit(app: &AppHandle, instruction: &str) -> Option<String> {
    let settings = get_settings(app);
    let Some(selection) = take_selection() else {
        debug!("Voice edit: no selection was captured for this session");
        return None;
    };

    let Some(provider) = settings.active_post_process_provider().cloned() else {
        debug!("Voice edit skipped: no post-processing provider is selected");
        return None;
    };
    let model = settings
        .post_process_models
        .get(&provider.id)
        .cloned()
        .unwrap_or_default();
    if model.trim().is_empty() {
        debug!(
            "Voice edit skipped: provider '{}' has no model configured",
            provider.id
        );
        return None;
    }
    if provider.id == APPLE_INTELLIGENCE_PROVIDER_ID {
        debug!("Voice edit is not supported for the Apple Intelligence provider");
        return None;
    }
    let api_key = settings
        .post_process_api_keys
        .get(&provider.id)
        .cloned()
        .unwrap_or_default();

    let prompt = build_voice_edit_prompt(&settings.voice_edit_prompt, &selection, instruction);
    debug!(
        "Voice edit: applying a {}-char instruction to a {}-char selection via '{}' (model: {})",
        instruction.chars().count(),
        selection.chars().count(),
        provider.id,
        model
    );

    match crate::llm_client::send_chat_completion(
        &provider,
        api_key,
        &model,
        prompt,
        settings.post_process_thinking,
    )
    .await
    {
        Ok(Some(content)) => {
            let text = crate::actions::clean_llm_output(&content);
            if text.trim().is_empty() {
                debug!("Voice edit: model returned an empty result");
                None
            } else {
                Some(text)
            }
        }
        Ok(None) => {
            error!("Voice edit: LLM API response has no content");
            None
        }
        Err(e) => {
            error!("Voice edit failed: {e}");
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::settings::default_voice_edit_prompt;

    #[test]
    fn default_prompt_substitutes_both_placeholders() {
        let prompt = build_voice_edit_prompt(
            &default_voice_edit_prompt(),
            "  Hello world  ",
            "  make it shout  ",
        );
        assert!(prompt.contains("<selected_text>\nHello world\n</selected_text>"));
        assert!(prompt.contains("<instruction>\nmake it shout\n</instruction>"));
        assert!(!prompt.contains("${selected}"));
        assert!(!prompt.contains("${instruction}"));
    }

    #[test]
    fn template_without_placeholders_gets_blocks_appended() {
        let prompt = build_voice_edit_prompt("Edit this for me.", "the text", "the instruction");
        assert!(prompt.starts_with("Edit this for me."));
        assert!(prompt.contains("<selected_text>\nthe text\n</selected_text>"));
        assert!(prompt.contains("<instruction>\nthe instruction\n</instruction>"));
    }

    #[test]
    fn template_with_only_one_placeholder_appends_the_other() {
        let prompt = build_voice_edit_prompt("Rewrite: ${selected}", "abc", "shorten it");
        assert!(prompt.contains("Rewrite: abc"));
        assert!(!prompt.contains("${selected}"));
        assert!(prompt.contains("<instruction>\nshorten it\n</instruction>"));
        assert!(!prompt.contains("<selected_text>"));
    }

    #[test]
    fn default_prompt_keeps_language_and_outputs_only_the_result() {
        let prompt = default_voice_edit_prompt();
        assert!(prompt.contains("ONLY the edited result"));
        assert!(prompt.contains("original language"));
    }
}
