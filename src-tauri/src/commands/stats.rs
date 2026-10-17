//! Frontend commands for the Stats page: range summaries, recently learned
//! words, and CSV export of the raw metric rows.

use crate::managers::history::HistoryManager;
use crate::stats::{self, LearnedWordEvent, StatsSummary};
use chrono::Local;
use std::sync::Arc;
use tauri::{AppHandle, Manager, State};

/// Aggregated statistics for the last `range_days` calendar days (1 = today).
#[tauri::command]
#[specta::specta]
pub async fn get_dictation_stats(
    range_days: u32,
    history_manager: State<'_, Arc<HistoryManager>>,
) -> Result<StatsSummary, String> {
    let range_days = range_days.clamp(1, 365);
    let today = Local::now().date_naive();
    let first_day = today - chrono::Days::new(range_days as u64 - 1);
    // Fetch from local midnight of the oldest day; `summarize` filters by
    // local calendar date anyway, so DST imprecision cannot double-count.
    let since = stats::local_midnight_unix(first_day);
    let metrics = history_manager
        .get_metrics_since(since)
        .map_err(|e| e.to_string())?;
    Ok(stats::summarize(&metrics, range_days, today))
}

/// The most recent words learned from user corrections (not undone).
#[tauri::command]
#[specta::specta]
pub async fn get_recent_learned_words(
    limit: Option<usize>,
    history_manager: State<'_, Arc<HistoryManager>>,
) -> Result<Vec<LearnedWordEvent>, String> {
    let limit = limit.unwrap_or(10).clamp(1, 100);
    history_manager
        .get_recent_learned_words(limit)
        .map_err(|e| e.to_string())
}

/// Writes every recorded metric row to `path` as CSV (the frontend picks the
/// path with a save dialog). Returns the number of exported rows.
#[tauri::command]
#[specta::specta]
pub async fn export_stats_csv(app: AppHandle, path: String) -> Result<u32, String> {
    let history_manager = app.state::<Arc<HistoryManager>>();
    let mut metrics = history_manager
        .get_all_metrics()
        .map_err(|e| format!("Failed to read metrics: {e}"))?;
    let count = metrics.len() as u32;
    let csv = stats::metrics_to_csv(&mut metrics);
    std::fs::write(&path, csv).map_err(|e| format!("Failed to write export file: {e}"))?;
    Ok(count)
}
