//! Custom-words import/export: parsing, normalization and merge logic plus
//! the two commands the Custom Words UI drives (preview an import file,
//! export the current list).
//!
//! All transformation lives in pure, unit-tested functions; the commands
//! only glue file IO and settings together. Normalization mirrors the
//! Custom Words UI (`normalizeCustomWord`): strip `<>"'`, collapse
//! whitespace, trim, cap at 50 characters.

use crate::settings::get_settings;
use log::info;
use serde::Serialize;
use specta::Type;
use std::fs;
use std::path::Path;
use tauri::{AppHandle, Emitter};

/// Mirrors the frontend's 50-character cap on custom words.
pub const MAX_WORD_LEN: usize = 50;

/// A parsed import file, ready for preview: normalized, deduplicated words
/// (in file order) plus how many of them are new relative to the current
/// custom words.
#[derive(Clone, Debug, Serialize, Type)]
pub struct WordlistPreview {
    /// All importable words from the file, deduplicated within the file.
    pub words: Vec<String>,
    /// Raw non-empty entries seen in the file (before normalization dropped
    /// or merged any).
    pub total_in_file: usize,
    /// Words not already present in the current custom words.
    pub new_count: usize,
}

/// Normalizes one raw word exactly like the Custom Words UI: strip `<>"'`,
/// collapse runs of whitespace to single spaces, trim, cap the length.
/// Returns `None` when nothing (valid) is left.
pub fn normalize_word(raw: &str) -> Option<String> {
    let mut normalized = String::with_capacity(raw.len());
    let mut last_was_space = true;
    for ch in raw.chars() {
        if ch.is_whitespace() {
            if !last_was_space {
                normalized.push(' ');
                last_was_space = true;
            }
        } else if !matches!(ch, '<' | '>' | '"' | '\'') {
            normalized.push(ch);
            last_was_space = false;
        }
    }
    let trimmed = normalized.trim_end();
    if trimmed.is_empty() {
        return None;
    }
    // Cap by characters, matching the UI's `length <= 50` check.
    if trimmed.chars().count() > MAX_WORD_LEN {
        return None;
    }
    Some(trimmed.to_string())
}

/// Case-insensitive key for dedupe; the first casing encountered wins.
fn dedupe_key(word: &str) -> String {
    word.to_lowercase()
}

/// Whether the first non-empty line of a CSV is a lone header such as
/// "word", "words", "term" or "vocabulary" (optionally quoted).
fn is_csv_header(line: &str) -> bool {
    let trimmed = line.trim().trim_matches('"');
    matches!(
        trimmed.to_ascii_lowercase().as_str(),
        "word" | "words" | "term" | "terms" | "vocabulary" | "custom_words"
    )
}

/// Parses plain-text or CSV content: one word per line, or several per line
/// separated by commas, semicolons or tabs. An optional leading header row
/// is skipped. Returns normalized words deduplicated case-insensitively
/// (first casing wins), plus the count of raw non-empty entries seen.
pub fn parse_wordlist_text(content: &str) -> (Vec<String>, usize) {
    let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
    let mut words = Vec::new();
    let mut total = 0usize;
    let mut first_non_empty = true;

    for line in content.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        if first_non_empty {
            first_non_empty = false;
            if is_csv_header(line) {
                continue;
            }
        }
        for field in line.split([',', ';', '\t']) {
            let Some(word) = normalize_word(field) else {
                continue;
            };
            total += 1;
            if seen.insert(dedupe_key(&word)) {
                words.push(word);
            }
        }
    }

    (words, total)
}

/// Parses a JSON wordlist: either an array of strings, or an object with a
/// `"words"` / `"custom_words"` array of strings. Non-string entries are
/// rejected so a malformed file cannot silently drop words.
pub fn parse_wordlist_json(content: &str) -> Result<(Vec<String>, usize), String> {
    let value: serde_json::Value =
        serde_json::from_str(content).map_err(|e| format!("invalid JSON: {e}"))?;
    let array = match &value {
        serde_json::Value::Array(items) => items.as_slice(),
        serde_json::Value::Object(map) => map
            .get("words")
            .or_else(|| map.get("custom_words"))
            .and_then(|v| v.as_array())
            .map(|items| items.as_slice())
            .ok_or_else(|| {
                "expected an array of words or an object with a \"words\" array".to_string()
            })?,
        _ => {
            return Err(
                "expected an array of words or an object with a \"words\" array".to_string(),
            )
        }
    };

    let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
    let mut words = Vec::new();
    let mut total = 0usize;
    for entry in array {
        let raw = entry
            .as_str()
            .ok_or_else(|| "expected every word to be a string".to_string())?;
        let Some(word) = normalize_word(raw) else {
            continue;
        };
        total += 1;
        if seen.insert(dedupe_key(&word)) {
            words.push(word);
        }
    }
    Ok((words, total))
}

/// Parses a wordlist file by extension: `.json` as JSON, everything else
/// (`.txt`, `.csv`, ...) as delimited text.
pub fn parse_wordlist(content: &str, file_name: &str) -> Result<(Vec<String>, usize), String> {
    let content = content.strip_prefix('\u{feff}').unwrap_or(content);
    if Path::new(file_name)
        .extension()
        .is_some_and(|ext| ext.eq_ignore_ascii_case("json"))
    {
        parse_wordlist_json(content)
    } else {
        Ok(parse_wordlist_text(content))
    }
}

/// How many of `incoming` are missing from `existing` (case-insensitive).
pub fn count_new_words(existing: &[String], incoming: &[String]) -> usize {
    let present: std::collections::HashSet<String> =
        existing.iter().map(|w| dedupe_key(w)).collect();
    incoming
        .iter()
        .filter(|w| !present.contains(&dedupe_key(w)))
        .count()
}

/// Merges `incoming` into `existing`: existing order is preserved, new
/// words are appended in import order, dedupe is case-insensitive (an
/// existing casing always wins over an imported one). Returns the merged
/// list and the number of words added.
pub fn merge_words(existing: &[String], incoming: &[String]) -> (Vec<String>, usize) {
    let mut merged = existing.to_vec();
    let mut seen: std::collections::HashSet<String> =
        existing.iter().map(|w| dedupe_key(w)).collect();
    let mut added = 0usize;
    for word in incoming {
        if seen.insert(dedupe_key(word)) {
            merged.push(word.clone());
            added += 1;
        }
    }
    (merged, added)
}

/// Renders words as one-per-line plain text.
pub fn words_to_txt(words: &[String]) -> String {
    if words.is_empty() {
        return String::new();
    }
    let mut out = words.join("\n");
    out.push('\n');
    out
}

/// Renders words as a pretty-printed JSON array (round-trips through
/// `parse_wordlist_json`).
pub fn words_to_json(words: &[String]) -> Result<String, String> {
    serde_json::to_string_pretty(words).map_err(|e| format!("failed to serialize words: {e}"))
}

/// Reads an import file and previews it against the current custom words.
#[tauri::command]
#[specta::specta]
pub fn import_wordlist_preview(app: AppHandle, path: String) -> Result<WordlistPreview, String> {
    let content = fs::read_to_string(&path).map_err(|e| format!("failed to read {path}: {e}"))?;
    let (words, total_in_file) = parse_wordlist(&content, &path)?;
    let existing = get_settings(&app).custom_words;
    let new_count = count_new_words(&existing, &words);
    info!(
        "Wordlist import preview: {} words in file, {} new",
        words.len(),
        new_count
    );
    Ok(WordlistPreview {
        words,
        total_in_file,
        new_count,
    })
}

/// Applies an import the user confirmed: merge into the current custom words
/// (case-insensitive dedupe, existing casing wins) or replace the list.
/// Returns the final word list; emits `settings-changed` so open windows
/// pick it up even without an optimistic frontend update.
#[tauri::command]
#[specta::specta]
pub fn apply_wordlist_import(
    app: AppHandle,
    words: Vec<String>,
    replace: bool,
) -> Result<Vec<String>, String> {
    let mut settings = get_settings(&app);
    let final_words = if replace {
        words
    } else {
        merge_words(&settings.custom_words, &words).0
    };
    settings.custom_words = final_words.clone();
    crate::settings::write_settings(&app, settings);
    let _ = app.emit(
        "settings-changed",
        serde_json::json!({ "setting": "custom_words" }),
    );
    Ok(final_words)
}

/// Writes the current custom words to `path` in the given format
/// ("txt" or "json"). Returns the number of words written.
#[tauri::command]
#[specta::specta]
pub fn export_wordlist(app: AppHandle, path: String, format: String) -> Result<usize, String> {
    let words = get_settings(&app).custom_words;
    let content = match format.as_str() {
        "txt" => words_to_txt(&words),
        "json" => words_to_json(&words)?,
        other => return Err(format!("unknown export format '{other}'")),
    };
    fs::write(&path, content).map_err(|e| format!("failed to write {path}: {e}"))?;
    info!("Exported {} custom words to {path}", words.len());
    Ok(words.len())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_strips_punctuation_and_collapses_whitespace() {
        assert_eq!(
            normalize_word("  Handy  One "),
            Some("Handy One".to_string())
        );
        assert_eq!(normalize_word("a<b>c\"d'e"), Some("abcde".to_string()));
        assert_eq!(
            normalize_word("tab\tand\nnewline"),
            Some("tab and newline".to_string())
        );
    }

    #[test]
    fn normalize_rejects_empty_and_overlong_words() {
        assert_eq!(normalize_word(""), None);
        assert_eq!(normalize_word("   "), None);
        assert_eq!(normalize_word("\"<>\""), None);
        assert_eq!(
            normalize_word(&"x".repeat(MAX_WORD_LEN)),
            Some("x".repeat(MAX_WORD_LEN))
        );
        assert_eq!(normalize_word(&"x".repeat(MAX_WORD_LEN + 1)), None);
    }

    #[test]
    fn parse_text_takes_one_word_per_line_or_delimited() {
        let (words, total) = parse_wordlist_text("alpha\nbeta, gamma\tdelta\n");
        assert_eq!(words, vec!["alpha", "beta", "gamma", "delta"]);
        assert_eq!(total, 4);
    }

    #[test]
    fn parse_text_skips_a_lone_header_row() {
        let (words, total) = parse_wordlist_text("word\nalpha\nbeta\n");
        assert_eq!(words, vec!["alpha", "beta"]);
        assert_eq!(total, 2);
    }

    #[test]
    fn parse_text_header_check_ignores_leading_blank_lines() {
        let (words, _) = parse_wordlist_text("\n\nword\nalpha\n");
        assert_eq!(words, vec!["alpha"]);
    }

    #[test]
    fn parse_text_dedupes_case_insensitively_keeping_first_casing() {
        let (words, total) = parse_wordlist_text("Handy\nhandy\nHANDY\nother");
        assert_eq!(words, vec!["Handy", "other"]);
        // every raw entry still counts toward the file total
        assert_eq!(total, 4);
    }

    #[test]
    fn parse_text_normalizes_and_drops_invalid_entries() {
        let (words, total) = parse_wordlist_text("  spaced  word \n\"quoted\"\n\"\"\n<>\n");
        assert_eq!(words, vec!["spaced word", "quoted"]);
        assert_eq!(total, 2);
    }

    #[test]
    fn parse_json_accepts_arrays_and_word_objects() {
        let (words, total) = parse_wordlist_json(r#"["alpha", "beta"]"#).expect("parse array");
        assert_eq!(words, vec!["alpha", "beta"]);
        assert_eq!(total, 2);

        let (words, total) = parse_wordlist_json(r#"{"words": ["gamma"]}"#).expect("parse object");
        assert_eq!(words, vec!["gamma"]);
        assert_eq!(total, 1);

        let (words, _) = parse_wordlist_json(r#"{"custom_words": ["delta"]}"#)
            .expect("parse custom_words object");
        assert_eq!(words, vec!["delta"]);
    }

    #[test]
    fn parse_json_rejects_malformed_input() {
        assert!(parse_wordlist_json("not json").is_err());
        assert!(parse_wordlist_json(r#"{"nope": []}"#).is_err());
        assert!(parse_wordlist_json(r#"["ok", 42]"#).is_err());
    }

    #[test]
    fn parse_dispatches_on_extension_and_strips_bom() {
        let (words, _) = parse_wordlist("[\"alpha\"]", "list.json").expect("json by extension");
        assert_eq!(words, vec!["alpha"]);

        let (words, _) = parse_wordlist("\u{feff}alpha\n", "list.txt").expect("text by extension");
        assert_eq!(words, vec!["alpha"]);

        let (words, _) = parse_wordlist("alpha,beta", "list.csv").expect("csv by extension");
        assert_eq!(words, vec!["alpha", "beta"]);
    }

    #[test]
    fn merge_preserves_existing_order_and_appends_new() {
        let existing = vec!["Handy".to_string(), "existing".to_string()];
        let incoming = vec!["handy".to_string(), "new".to_string()];
        let (merged, added) = merge_words(&existing, &incoming);
        assert_eq!(merged, vec!["Handy", "existing", "new"]);
        assert_eq!(added, 1);
    }

    #[test]
    fn count_new_words_is_case_insensitive() {
        let existing = vec!["Handy".to_string()];
        assert_eq!(
            count_new_words(&existing, &["handy".to_string(), "fresh".to_string()]),
            1
        );
        assert_eq!(count_new_words(&existing, &[]), 0);
    }

    #[test]
    fn txt_export_round_trips_through_text_parse() {
        let words = vec!["alpha".to_string(), "be ta".to_string()];
        let txt = words_to_txt(&words);
        assert_eq!(txt, "alpha\nbe ta\n");
        let (parsed, total) = parse_wordlist_text(&txt);
        assert_eq!(parsed, words);
        assert_eq!(total, 2);
    }

    #[test]
    fn json_export_round_trips_through_json_parse() {
        let words = vec!["alpha".to_string(), "be ta".to_string()];
        let json = words_to_json(&words).expect("serialize");
        let (parsed, total) = parse_wordlist_json(&json).expect("parse");
        assert_eq!(parsed, words);
        assert_eq!(total, 2);
    }

    #[test]
    fn empty_txt_export_is_empty_string() {
        assert_eq!(words_to_txt(&[]), "");
    }
}
