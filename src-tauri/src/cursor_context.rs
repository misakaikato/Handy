//! Reference-only cursor context for post-processing.
//!
//! When enabled, the post-processing LLM rewrite is preceded by one
//! Accessibility read of the focused text field: its full value plus the
//! selection range ([`crate::ax_reader`]). Up to `cursor_context_chars`
//! characters on each side of the selection are passed to the model inside a
//! `<cursor_context>` block that is explicitly labeled as context the model
//! must not repeat in its output — it exists so pronouns and half-sentences
//! can be resolved against the text the cursor actually sits in.
//!
//! Secure/password fields are always skipped, and every failure (no
//! accessibility permission, no focused text field, no selection range, a
//! non-text element) is a silent skip: context injection must never disturb
//! the dictation flow. macOS only — on other platforms the read returns
//! `None` and the setting has no effect.

use log::debug;

/// The text surrounding the selection in the focused field, already
/// truncated to the configured per-side limit.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct CursorContext {
    pub before: String,
    pub after: String,
}

/// Whether a focused element is a secure (password) field that must never be
/// read, judged from its AX role / AX subrole. Browsers and native apps
/// expose password inputs as `AXSecureTextField` in one of the two.
pub(crate) fn is_secure_text_field(role: Option<&str>, subrole: Option<&str>) -> bool {
    const SECURE: &str = "AXSecureTextField";
    role == Some(SECURE) || subrole == Some(SECURE)
}

/// Maps a UTF-16 code-unit offset (what the AX APIs report) onto the byte
/// index of the containing/boundary char, clamped to `text.len()`. An offset
/// landing inside a surrogate pair rounds up to the char after the pair; both
/// resolutions are valid char boundaries, so slicing stays safe.
pub(crate) fn utf16_to_char_index(text: &str, utf16_index: usize) -> usize {
    let mut units = 0usize;
    for (byte_index, ch) in text.char_indices() {
        if units >= utf16_index {
            return byte_index;
        }
        units += ch.len_utf16();
    }
    text.len()
}

/// Extracts up to `max_chars` characters on each side of the selection
/// (`range` is (location, length) in UTF-16 units into `value`), trimming
/// whitespace from the cut ends. Returns `None` when both sides end up empty
/// or the limits are degenerate.
pub(crate) fn build_cursor_context(
    value: &str,
    range: (usize, usize),
    max_chars: usize,
) -> Option<CursorContext> {
    if max_chars == 0 {
        return None;
    }
    let (location, length) = range;
    let selection_start = utf16_to_char_index(value, location);
    let selection_end =
        utf16_to_char_index(value, location.saturating_add(length)).max(selection_start);

    let before = value.get(..selection_start)?;
    let after = value.get(selection_end..)?;

    let before_tail: String = before.chars().rev().take(max_chars).collect();
    let before_tail: String = before_tail.chars().rev().collect();
    let after_head: String = after.chars().take(max_chars).collect();

    let context = CursorContext {
        before: before_tail.trim().to_string(),
        after: after_head.trim().to_string(),
    };
    if context.before.is_empty() && context.after.is_empty() {
        None
    } else {
        Some(context)
    }
}

/// Formats the context as a prompt suffix: a delimited block carrying the
/// surrounding text plus an explicit instruction that it is reference-only
/// and must never appear in the output.
pub(crate) fn context_prompt_block(context: &CursorContext) -> String {
    let mut block = String::from("\n\n<cursor_context>\n");
    if !context.before.is_empty() {
        block.push_str("Text before the cursor:\n");
        block.push_str(&context.before);
        block.push_str("\n\n");
    }
    if !context.after.is_empty() {
        block.push_str("Text after the cursor:\n");
        block.push_str(&context.after);
        block.push('\n');
    }
    block.push_str(
        "</cursor_context>\nThe <cursor_context> block shows the text surrounding the user's \
cursor. It is reference-only: use it to interpret the transcript, but never repeat, quote, \
translate, or continue its content in your output.",
    );
    block
}

/// Reads the cursor context from the focused field, off the async runtime
/// (AX messages block up to their 1s timeout). Silent skip on any failure.
pub(crate) async fn read_cursor_context(max_chars: u32) -> Option<CursorContext> {
    let max_chars = (max_chars as usize).clamp(1, 10_000);
    let result =
        tauri::async_runtime::spawn_blocking(move || read_cursor_context_blocking(max_chars)).await;
    match result {
        Ok(context) => context,
        Err(e) => {
            debug!("Cursor-context read task failed: {e}");
            None
        }
    }
}

fn read_cursor_context_blocking(max_chars: usize) -> Option<CursorContext> {
    let snapshot = crate::ax_reader::focused_element_snapshot()?;
    if is_secure_text_field(snapshot.role.as_deref(), snapshot.subrole.as_deref()) {
        debug!("Cursor-context skipped: focused field is a secure text field");
        return None;
    }
    let value = snapshot.value?;
    let range = snapshot.selected_range?;
    build_cursor_context(&value, range, max_chars)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn secure_fields_are_detected_by_role_or_subrole() {
        assert!(is_secure_text_field(Some("AXSecureTextField"), None));
        assert!(is_secure_text_field(None, Some("AXSecureTextField")));
        assert!(is_secure_text_field(
            Some("AXTextField"),
            Some("AXSecureTextField")
        ));
        assert!(!is_secure_text_field(Some("AXTextField"), None));
        assert!(!is_secure_text_field(None, None));
        assert!(!is_secure_text_field(
            Some("AXTextArea"),
            Some("AXSearchField")
        ));
    }

    #[test]
    fn utf16_offsets_map_onto_char_boundaries() {
        // ASCII: byte index == char index == UTF-16 index.
        assert_eq!(utf16_to_char_index("hello", 2), 2);
        // 'é' is one char / two bytes / one UTF-16 unit.
        assert_eq!(utf16_to_char_index("café au lait", 5), 6);
        // '😀' is one char / four bytes / two UTF-16 units.
        assert_eq!(utf16_to_char_index("a😀b", 0), 0);
        assert_eq!(utf16_to_char_index("a😀b", 1), 1);
        // Offset 2 lands inside the emoji; rounds to the char after it.
        assert_eq!(utf16_to_char_index("a😀b", 2), 5);
        assert_eq!(utf16_to_char_index("a😀b", 3), 5);
        // Out-of-bounds clamps to the end.
        assert_eq!(utf16_to_char_index("a😀b", 99), "a😀b".len());
        assert_eq!(utf16_to_char_index("", 3), 0);
    }

    #[test]
    fn context_wraps_the_selection() {
        let value = "The quick brown fox jumps over the lazy dog";
        // Selection "brown" (offsets 10..15), cursor context 5 chars each side.
        let context = build_cursor_context(value, (10, 5), 5).expect("context");
        assert_eq!(context.before, "uick");
        assert_eq!(context.after, "fox");
    }

    #[test]
    fn context_takes_last_n_chars_before_and_first_n_after() {
        let value = "0123456789";
        let context = build_cursor_context(value, (5, 0), 3).expect("context");
        assert_eq!(context.before, "234");
        assert_eq!(context.after, "567");
    }

    #[test]
    fn context_handles_cjk_and_emoji_offsets() {
        // "你好😀世界": UTF-16 units — 你(1)好(1)😀(2)世(1)界(1). Selection
        // starts at unit 2 (inside/after the emoji is unimportant: mapping
        // must stay on char boundaries either way).
        let value = "你好😀世界";
        let context = build_cursor_context(value, (2, 2), 10).expect("context");
        assert_eq!(context.before, "你好");
        assert!(context.after.contains('世'));
        assert!(context.after.contains('界'));
    }

    #[test]
    fn cursor_at_boundaries_yields_one_sided_context() {
        let value = "around the cursor";
        let context = build_cursor_context(value, (0, 0), 7).expect("context");
        assert_eq!(context.before, "");
        assert_eq!(context.after, "around");

        let context = build_cursor_context(value, (value.len(), 0), 7).expect("context");
        assert_eq!(context.before, "cursor");
        assert_eq!(context.after, "");
    }

    #[test]
    fn blank_context_is_none() {
        assert!(build_cursor_context("", (0, 0), 100).is_none());
        assert!(build_cursor_context("   ", (0, 0), 100).is_none());
        // Selection spanning the whole (whitespace-only) neighborhood.
        assert!(build_cursor_context("   ", (0, 3), 100).is_none());
    }

    #[test]
    fn zero_limit_is_none() {
        assert!(build_cursor_context("text around", (4, 0), 0).is_none());
    }

    #[test]
    fn cut_ends_are_trimmed() {
        let value = "word   |tail";
        let context = build_cursor_context(value, (7, 0), 6).expect("context");
        assert_eq!(context.before, "ord");
        assert_eq!(context.after, "|tail");
    }

    #[test]
    fn oversized_range_clamps_instead_of_panicking() {
        let value = "short";
        let context = build_cursor_context(value, (3, 500), 100).expect("context");
        assert_eq!(context.before, "sho");
        assert_eq!(context.after, "");
    }

    #[test]
    fn prompt_block_delimits_and_instructs() {
        let context = CursorContext {
            before: "before text".to_string(),
            after: "after text".to_string(),
        };
        let block = context_prompt_block(&context);
        assert!(block.starts_with("\n\n<cursor_context>\n"));
        assert!(block.contains("Text before the cursor:\nbefore text"));
        assert!(block.contains("Text after the cursor:\nafter text"));
        assert!(block.contains("</cursor_context>"));
        assert!(block.contains("never repeat, quote, translate, or continue"));
    }

    #[test]
    fn prompt_block_omits_empty_sides() {
        let one_sided = CursorContext {
            before: String::new(),
            after: "after".to_string(),
        };
        let block = context_prompt_block(&one_sided);
        assert!(!block.contains("Text before the cursor"));
        assert!(block.contains("Text after the cursor:\nafter"));
    }
}
