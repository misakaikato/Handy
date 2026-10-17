//! Toast window announcing auto-learned words, with an Undo action.
//!
//! A small non-activating NSPanel (same construction as the recording
//! overlay, reusing its positioning) shown near the overlay position after a
//! learn event. The webview renders "Learned: … [Undo]"; the backend hides
//! the panel again after the configured duration. Clicking Undo invokes the
//! `undo_auto_learn` command, which removes exactly this event's words from
//! `custom_words` and persists the change. macOS only — auto-learning itself
//! is macOS-only.

use log::{debug, warn};
use once_cell::sync::Lazy;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;
use std::time::Duration;
use tauri::{AppHandle, Emitter, Manager, WebviewUrl};

use tauri_nspanel::{tauri_panel, CollectionBehavior, PanelBuilder, PanelLevel, StyleMask};

use super::logic;

tauri_panel! {
    panel!(AutoLearnToastPanel {
        config: {
            // Non-key so showing the toast can never steal focus from the
            // text field the user just corrected. Mouse clicks still reach
            // the webview button.
            can_become_key_window: false,
            is_floating_panel: true
        }
    })
}

const TOAST_LABEL: &str = "auto_learn_toast";

// Toast card geometry (logical points); the CSS card is flush to the window
// bounds. Keep in sync with AutoLearnToast.css.
const TOAST_WIDTH: f64 = 380.0;
const TOAST_HEIGHT: f64 = 44.0;

/// The most recent learn event, addressable by id so a stale Undo (from a
/// toast already replaced by a newer event) cannot remove the wrong words.
/// `stat_row_ids` are the learned_words rows persisted for the Stats page.
struct PendingUndo {
    id: u64,
    words: Vec<String>,
    stat_row_ids: Vec<i64>,
}

static PENDING_UNDO: Lazy<Mutex<Option<PendingUndo>>> = Lazy::new(|| Mutex::new(None));
static NEXT_EVENT_ID: AtomicU64 = AtomicU64::new(1);

/// Generation counter bumped on every show and undo. The delayed `hide()`
/// only unmaps the window if no newer show happened meanwhile — same pattern
/// as the recording overlay's hide.
static TOAST_SHOW_GENERATION: AtomicU64 = AtomicU64::new(0);

/// Creates the toast panel at startup, hidden. Created eagerly (like the
/// recording overlay) so the webview and its event listener are ready long
/// before the first learn event arrives.
pub fn create_learned_toast(app_handle: &AppHandle) {
    let position =
        crate::overlay::calculate_overlay_position(app_handle, TOAST_WIDTH, TOAST_HEIGHT);

    let mut builder = PanelBuilder::<_, AutoLearnToastPanel>::new(app_handle, TOAST_LABEL)
        .url(WebviewUrl::App("src/auto-learn-toast/index.html".into()))
        .title("Learned Words")
        .level(PanelLevel::Status)
        .size(tauri::Size::Logical(tauri::LogicalSize {
            width: TOAST_WIDTH,
            height: TOAST_HEIGHT,
        }))
        .has_shadow(false)
        .transparent(true)
        .no_activate(true)
        .corner_radius(0.0)
        .style_mask(StyleMask::empty().borderless().nonactivating_panel())
        .with_window(|w| {
            w.decorations(false)
                .transparent(true)
                .focusable(false)
                .accept_first_mouse(true)
        })
        .collection_behavior(
            CollectionBehavior::new()
                .can_join_all_spaces()
                .full_screen_auxiliary(),
        );

    if let Some((x, y)) = position {
        builder = builder.position(tauri::Position::Logical(tauri::LogicalPosition { x, y }));
    }

    match builder.build() {
        Ok(panel) => panel.hide(),
        Err(e) => debug!("Failed to create learned-words toast panel: {}", e),
    }
}

/// Shows the toast for one learn event and schedules its hide.
///
/// Called from the learn thread; hops to the main thread for the panel and
/// monitor work.
pub(crate) fn show_learned_toast(app: &AppHandle, words: Vec<String>, stat_row_ids: Vec<i64>) {
    let duration_ms = crate::settings::get_settings(app)
        .auto_learn_notification_duration_ms
        .clamp(500, 60_000);
    let id = NEXT_EVENT_ID.fetch_add(1, Ordering::Relaxed);
    lock_pending(&PENDING_UNDO).replace(PendingUndo {
        id,
        words: words.clone(),
        stat_row_ids,
    });

    let handle = app.clone();
    let _ = app.run_on_main_thread(move || {
        show_on_main(&handle, id, words, duration_ms);
    });
}

/// Undo entry point (also called from the command on other platforms as a
/// no-op). Returns whether an event with this id was still pending.
pub(crate) fn undo_last_learned(app: &AppHandle, event_id: u64) -> bool {
    let pending = match lock_pending(&PENDING_UNDO).take_if(|p| p.id == event_id) {
        Some(pending) => pending,
        None => {
            debug!("Auto-learn undo ignored: no pending event {event_id}");
            return false;
        }
    };

    let mut settings = crate::settings::get_settings(app);
    let removed = logic::remove_learned_words(&pending.words, &mut settings.custom_words);
    if removed > 0 {
        crate::settings::write_settings(app, settings);
        let _ = app.emit(
            "settings-changed",
            serde_json::json!({ "setting": "custom_words" }),
        );
        debug!("Auto-learn undo removed {removed} word(s)");
    }
    // Mirror the undo in the stats history so "recently learned" no longer
    // lists the undone words.
    if let Some(hm) = app.try_state::<std::sync::Arc<crate::managers::history::HistoryManager>>() {
        if let Err(e) = hm.mark_learned_words_undone(&pending.stat_row_ids) {
            debug!("Auto-learn undo could not update stats rows: {e}");
        }
    }
    hide_toast(app);
    true
}

/// Hides the toast window and invalidates any scheduled hide.
fn hide_toast(app: &AppHandle) {
    TOAST_SHOW_GENERATION.fetch_add(1, Ordering::SeqCst);
    if let Some(window) = app.get_webview_window(TOAST_LABEL) {
        let _ = window.hide();
    }
}

fn show_on_main(app: &AppHandle, event_id: u64, words: Vec<String>, duration_ms: u64) {
    TOAST_SHOW_GENERATION.fetch_add(1, Ordering::SeqCst);

    let Some(window) = app.get_webview_window(TOAST_LABEL) else {
        warn!("Learned-words toast window missing; skipping notification");
        return;
    };

    // Re-position on every show: the cursor may be on a different monitor
    // than at startup, and the recording overlay may sit Top or Bottom.
    if let Some((x, y)) = crate::overlay::calculate_overlay_position(app, TOAST_WIDTH, TOAST_HEIGHT)
    {
        let _ = window.set_position(tauri::Position::Logical(tauri::LogicalPosition { x, y }));
    }
    let _ = window.show();
    let _ = app.emit_to(
        TOAST_LABEL,
        "learned-words",
        serde_json::json!({ "id": event_id, "words": words }),
    );

    // Hide after the configured duration unless a newer event (or an undo)
    // bumped the generation in the meantime.
    let scheduled_at = TOAST_SHOW_GENERATION.load(Ordering::SeqCst);
    let handle = app.clone();
    std::thread::spawn(move || {
        std::thread::sleep(Duration::from_millis(duration_ms));
        if TOAST_SHOW_GENERATION.load(Ordering::SeqCst) != scheduled_at {
            return;
        }
        if let Some(window) = handle.get_webview_window(TOAST_LABEL) {
            let _ = window.hide();
        }
    });
}

/// Locks a pending-undo mutex, tolerating a poisoned lock (undo state is
/// advisory; never worth propagating an error to the user).
fn lock_pending(
    pending: &Mutex<Option<PendingUndo>>,
) -> std::sync::MutexGuard<'_, Option<PendingUndo>> {
    pending
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}
