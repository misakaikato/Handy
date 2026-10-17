use enigo::{Enigo, Key, Keyboard, Mouse, Settings};
use std::sync::Mutex;
use tauri::{AppHandle, Manager};

#[cfg(target_os = "macos")]
mod macos {
    use super::Key;
    use log::{debug, warn};
    use std::ffi::c_void;

    type TisInputSourceRef = *const c_void;
    type CfDataRef = *const c_void;
    type CfStringRef = *const c_void;

    // kVK_ANSI_V. This is the behavior Handy used before layout-aware
    // resolution and remains the safest fallback if macOS cannot expose the
    // active layout.
    const ANSI_V_KEYCODE: u16 = 9;
    const KEYCODE_COUNT: u16 = 128;
    const UC_KEY_ACTION_DISPLAY: u16 = 3;
    const UC_KEY_TRANSLATE_NO_DEAD_KEYS_MASK: u32 = 1;
    // Carbon's cmdKey is bit 8. UCKeyTranslate expects Carbon modifiers shifted
    // right by 8, so Command is represented by bit 0 here.
    const COMMAND_MODIFIER_STATE: u32 = 1;

    #[link(name = "Carbon", kind = "framework")]
    unsafe extern "C" {
        fn TISCopyCurrentKeyboardLayoutInputSource() -> TisInputSourceRef;
        fn TISGetInputSourceProperty(
            input_source: TisInputSourceRef,
            property_key: CfStringRef,
        ) -> CfDataRef;
        static kTISPropertyUnicodeKeyLayoutData: CfStringRef;
        fn UCKeyTranslate(
            key_layout: *const u8,
            virtual_key_code: u16,
            key_action: u16,
            modifier_key_state: u32,
            keyboard_type: u32,
            key_translate_options: u32,
            dead_key_state: *mut u32,
            max_string_length: usize,
            actual_string_length: *mut usize,
            unicode_string: *mut u16,
        ) -> i32;
        fn LMGetKbdType() -> u8;
    }

    #[link(name = "CoreFoundation", kind = "framework")]
    unsafe extern "C" {
        fn CFDataGetBytePtr(data: CfDataRef) -> *const u8;
        fn CFRelease(value: *const c_void);
    }

    struct InputSource(TisInputSourceRef);

    impl Drop for InputSource {
        fn drop(&mut self) {
            if !self.0.is_null() {
                // SAFETY: TISCopyCurrentKeyboardLayoutInputSource returned this
                // retained reference, so this balances that ownership.
                unsafe { CFRelease(self.0) };
            }
        }
    }

    fn find_keycode(mut matches: impl FnMut(u16) -> bool) -> Option<u16> {
        (0..KEYCODE_COUNT).find(|&keycode| matches(keycode))
    }

    /// Resolves the physical key that macOS interprets as `v` while Command is
    /// held. Including Command is important: non-Latin layouts commonly map
    /// Cmd shortcuts to their ANSI equivalents, while standard Dvorak does not.
    ///
    /// TIS APIs must run on the main thread. Handy's paste path already enters
    /// through `AppHandle::run_on_main_thread` before reaching this function.
    fn resolve_command_v_keycode() -> Result<u16, String> {
        // SAFETY: This function is called on the macOS main thread. The returned
        // source follows the Create Rule and is released by InputSource::drop.
        let source = InputSource(unsafe { TISCopyCurrentKeyboardLayoutInputSource() });
        if source.0.is_null() {
            return Err("macOS returned no current keyboard layout input source".into());
        }

        // SAFETY: The source remains retained for the duration of the scan and
        // the property constant is provided by Carbon.
        let layout_data =
            unsafe { TISGetInputSourceProperty(source.0, kTISPropertyUnicodeKeyLayoutData) };
        if layout_data.is_null() {
            return Err("current macOS keyboard layout has no Unicode layout data".into());
        }

        // SAFETY: layout_data is a CFData owned by the retained input source and
        // remains valid until source is dropped after the scan.
        let layout = unsafe { CFDataGetBytePtr(layout_data) };
        if layout.is_null() {
            return Err("current macOS keyboard layout data is empty".into());
        }

        // SAFETY: LMGetKbdType has no arguments and returns the current physical
        // keyboard type used by UCKeyTranslate.
        let keyboard_type = unsafe { LMGetKbdType() } as u32;
        let keycode = find_keycode(|keycode| {
            let mut dead_key_state = 0;
            let mut chars = [0_u16; 4];
            let mut length = 0_usize;

            // SAFETY: layout points to valid UCKeyboardLayout bytes while source
            // is retained. All output pointers reference initialized local
            // storage of the declared sizes.
            let status = unsafe {
                UCKeyTranslate(
                    layout,
                    keycode,
                    UC_KEY_ACTION_DISPLAY,
                    COMMAND_MODIFIER_STATE,
                    keyboard_type,
                    UC_KEY_TRANSLATE_NO_DEAD_KEYS_MASK,
                    &mut dead_key_state,
                    chars.len(),
                    &mut length,
                    chars.as_mut_ptr(),
                )
            };

            status == 0 && length == 1 && chars[0] == u16::from(b'v')
        })
        .ok_or_else(|| "could not map Cmd+V in the current macOS keyboard layout".to_string())?;

        Ok(keycode)
    }

    pub(super) fn command_v_key() -> Key {
        match resolve_command_v_keycode() {
            Ok(keycode) => {
                debug!("Resolved Cmd+V for the active macOS layout to keycode {keycode}");
                Key::Other(u32::from(keycode))
            }
            Err(error) => {
                warn!(
                    "Could not resolve Cmd+V for the active macOS layout ({error}); using ANSI V keycode {ANSI_V_KEYCODE}"
                );
                Key::Other(u32::from(ANSI_V_KEYCODE))
            }
        }
    }
}

/// Wrapper for Enigo to store in Tauri's managed state.
/// Enigo is wrapped in a Mutex since it requires mutable access.
pub struct EnigoState(pub Mutex<Enigo>);

impl EnigoState {
    pub fn new() -> Result<Self, String> {
        let enigo = Enigo::new(&Settings::default())
            .map_err(|e| format!("Failed to initialize Enigo: {}", e))?;
        Ok(Self(Mutex::new(enigo)))
    }
}

/// Get the current mouse cursor position using the managed Enigo instance.
/// Returns None if the state is not available or if getting the location fails.
pub fn get_cursor_position(app_handle: &AppHandle) -> Option<(i32, i32)> {
    let enigo_state = app_handle.try_state::<EnigoState>()?;
    let enigo = enigo_state.0.lock().ok()?;
    enigo.location().ok()
}

/// Sends a Ctrl+V or Cmd+V paste command using platform-specific virtual key codes.
/// This ensures the paste works regardless of keyboard layout (e.g., Russian, AZERTY, DVORAK).
/// Note: On Wayland, this may not work - callers should check for Wayland and use alternative methods.
///
/// `hold_ms` is how long the modifier stays held after the V click before being
/// released. Most applications read the modifier from the V event's flags and
/// need no hold at all, but applications that poll global keyboard state when
/// handling the key need the modifier to still be down — the hold insures
/// against those. Callers that can detect a failed chord (e.g. the
/// receipt-sequenced paste path) may use a much shorter hold.
pub fn send_paste_ctrl_v(enigo: &mut Enigo, hold_ms: u64) -> Result<(), String> {
    // Platform-specific key definitions
    #[cfg(target_os = "macos")]
    let (modifier_key, v_key_code) = (Key::Meta, macos::command_v_key());
    #[cfg(target_os = "windows")]
    let (modifier_key, v_key_code) = (Key::Control, Key::Other(0x56)); // VK_V
    #[cfg(target_os = "linux")]
    let (modifier_key, v_key_code) = (Key::Control, Key::Unicode('v'));

    // Press modifier + V
    enigo
        .key(modifier_key, enigo::Direction::Press)
        .map_err(|e| format!("Failed to press modifier key: {}", e))?;
    enigo
        .key(v_key_code, enigo::Direction::Click)
        .map_err(|e| format!("Failed to click V key: {}", e))?;

    std::thread::sleep(std::time::Duration::from_millis(hold_ms));

    enigo
        .key(modifier_key, enigo::Direction::Release)
        .map_err(|e| format!("Failed to release modifier key: {}", e))?;

    Ok(())
}

/// Sends a Ctrl+Shift+V paste command.
/// This is commonly used in terminal applications on Linux to paste without formatting.
/// Note: On Wayland, this may not work - callers should check for Wayland and use alternative methods.
pub fn send_paste_ctrl_shift_v(enigo: &mut Enigo, hold_ms: u64) -> Result<(), String> {
    // Platform-specific key definitions
    #[cfg(target_os = "macos")]
    let (modifier_key, v_key_code) = (Key::Meta, macos::command_v_key());
    #[cfg(target_os = "windows")]
    let (modifier_key, v_key_code) = (Key::Control, Key::Other(0x56)); // VK_V
    #[cfg(target_os = "linux")]
    let (modifier_key, v_key_code) = (Key::Control, Key::Unicode('v'));

    // Press Ctrl/Cmd + Shift + V
    enigo
        .key(modifier_key, enigo::Direction::Press)
        .map_err(|e| format!("Failed to press modifier key: {}", e))?;
    enigo
        .key(Key::Shift, enigo::Direction::Press)
        .map_err(|e| format!("Failed to press Shift key: {}", e))?;
    enigo
        .key(v_key_code, enigo::Direction::Click)
        .map_err(|e| format!("Failed to click V key: {}", e))?;

    std::thread::sleep(std::time::Duration::from_millis(hold_ms));

    enigo
        .key(Key::Shift, enigo::Direction::Release)
        .map_err(|e| format!("Failed to release Shift key: {}", e))?;
    enigo
        .key(modifier_key, enigo::Direction::Release)
        .map_err(|e| format!("Failed to release modifier key: {}", e))?;

    Ok(())
}

/// Sends a Shift+Insert paste command (Windows and Linux only).
/// This is more universal for terminal applications and legacy software.
/// Note: On Wayland, this may not work - callers should check for Wayland and use alternative methods.
pub fn send_paste_shift_insert(enigo: &mut Enigo, hold_ms: u64) -> Result<(), String> {
    #[cfg(target_os = "windows")]
    let insert_key_code = Key::Other(0x2D); // VK_INSERT
    #[cfg(not(target_os = "windows"))]
    let insert_key_code = Key::Other(0x76); // XK_Insert (keycode 118 / 0x76, also used as fallback)

    // Press Shift + Insert
    enigo
        .key(Key::Shift, enigo::Direction::Press)
        .map_err(|e| format!("Failed to press Shift key: {}", e))?;
    enigo
        .key(insert_key_code, enigo::Direction::Click)
        .map_err(|e| format!("Failed to click Insert key: {}", e))?;

    std::thread::sleep(std::time::Duration::from_millis(hold_ms));

    enigo
        .key(Key::Shift, enigo::Direction::Release)
        .map_err(|e| format!("Failed to release Shift key: {}", e))?;

    Ok(())
}

/// Types text at the cursor with explicit handling for the characters that
/// keystroke-level injection cannot carry reliably on macOS.
///
/// enigo's macOS `text` posts one CGEvent per 20 *characters* with no pacing
/// (its inter-event accounting only runs on drop). Apps with an active input
/// method drop characters when those unicode-string events arrive back to
/// back; a chunk boundary can land inside a surrogate pair; and enigo works
/// around a macOS quirk (an event whose string *starts* with a newline is
/// silently dropped) by prefixing a zero-width space — which then lands in
/// the text. Here the text is pre-split into runs of at most 20 UTF-16
/// units (so enigo never splits again, and never sees a run starting with a
/// newline), newlines and tabs become real Return/Tab keystrokes, invisible
/// characters are dropped, and a small gap between events lets slow apps
/// keep up.
#[cfg(target_os = "macos")]
pub fn paste_text_direct(enigo: &mut Enigo, text: &str) -> Result<(), String> {
    for segment in split_typed_segments(text) {
        match segment {
            TypedSegment::Text(run) => {
                enigo
                    .text(&run)
                    .map_err(|e| format!("Failed to send text directly: {e}"))?;
            }
            TypedSegment::Return => {
                enigo
                    .key(Key::Return, enigo::Direction::Click)
                    .map_err(|e| format!("Failed to type Return: {e}"))?;
            }
            TypedSegment::Tab => {
                enigo
                    .key(Key::Tab, enigo::Direction::Click)
                    .map_err(|e| format!("Failed to type Tab: {e}"))?;
            }
        }
        std::thread::sleep(MACOS_TYPED_EVENT_GAP);
    }
    Ok(())
}

/// Gap between injected events. enigo assumes the OS needs ~20ms per event
/// but never sleeps between them; 10ms keeps a 48-char streamed chunk well
/// under 50ms of main-thread time while letting IME-heavy apps keep up.
#[cfg(target_os = "macos")]
const MACOS_TYPED_EVENT_GAP: std::time::Duration = std::time::Duration::from_millis(10);

/// One piece of typed output after splitting for reliable injection.
#[cfg(target_os = "macos")]
#[derive(Debug, PartialEq, Eq)]
enum TypedSegment {
    /// A run of visible text: at most 20 UTF-16 units, no control characters,
    /// and never starting with a newline or tab.
    Text(String),
    /// A line break, typed as a real Return keystroke.
    Return,
    /// A horizontal tab, typed as a real Tab keystroke.
    Tab,
}

/// Splits text into [`TypedSegment`]s (see the `paste_text_direct` docs for
/// why each guarantee exists). Pure, so the boundary rules are testable
/// without posting events.
#[cfg(target_os = "macos")]
fn split_typed_segments(text: &str) -> Vec<TypedSegment> {
    /// CGEventKeyboardSetUnicodeString truncates beyond 20 UTF-16 units.
    const MAX_UTF16_PER_EVENT: usize = 20;

    fn push_run(segments: &mut Vec<TypedSegment>, run: &mut String, units: &mut usize) {
        if !run.is_empty() {
            segments.push(TypedSegment::Text(std::mem::take(run)));
        }
        *units = 0;
    }

    let mut segments = Vec::new();
    let mut run = String::new();
    let mut units = 0usize;

    let mut chars = text.chars().peekable();
    while let Some(ch) = chars.next() {
        match ch {
            '\r' => {
                // A CRLF pair is one line break, not two.
                if chars.peek() == Some(&'\n') {
                    chars.next();
                }
                push_run(&mut segments, &mut run, &mut units);
                segments.push(TypedSegment::Return);
            }
            '\n' => {
                push_run(&mut segments, &mut run, &mut units);
                segments.push(TypedSegment::Return);
            }
            '\t' => {
                push_run(&mut segments, &mut run, &mut units);
                segments.push(TypedSegment::Tab);
            }
            // Zero-width characters and BOMs never belong in typed output
            // (enigo itself injects U+200B as a newline workaround), and a
            // null byte is rejected outright by the Windows backend.
            '\u{200B}' | '\u{200C}' | '\u{200D}' | '\u{FEFF}' | '\0' => {}
            _ => {
                if units + ch.len_utf16() > MAX_UTF16_PER_EVENT {
                    push_run(&mut segments, &mut run, &mut units);
                }
                run.push(ch);
                units += ch.len_utf16();
            }
        }
    }
    push_run(&mut segments, &mut run, &mut units);
    segments
}

/// Pastes text directly using the enigo text method.
/// This tries to use system input methods if possible, otherwise simulates keystrokes one by one.
#[cfg(not(target_os = "macos"))]
pub fn paste_text_direct(enigo: &mut Enigo, text: &str) -> Result<(), String> {
    enigo
        .text(text)
        .map_err(|e| format!("Failed to send text directly: {e}"))?;

    Ok(())
}

#[cfg(all(test, target_os = "macos"))]
mod tests {
    use super::{split_typed_segments, TypedSegment};

    fn utf16_len(s: &str) -> usize {
        s.chars().map(char::len_utf16).sum()
    }

    fn text_runs(segments: &[TypedSegment]) -> Vec<&str> {
        segments
            .iter()
            .filter_map(|s| match s {
                TypedSegment::Text(t) => Some(t.as_str()),
                _ => None,
            })
            .collect()
    }

    #[test]
    fn plain_text_is_a_single_run() {
        assert_eq!(
            split_typed_segments("hello 世界"),
            vec![TypedSegment::Text("hello 世界".to_string())]
        );
        assert_eq!(split_typed_segments(""), Vec::<TypedSegment>::new());
    }

    #[test]
    fn cjk_runs_respect_the_utf16_event_limit() {
        let text = "字".repeat(45);
        let segments = split_typed_segments(&text);
        let runs = text_runs(&segments);
        assert_eq!(runs.len(), 3, "45 CJK chars split at 20/20/5");
        assert!(runs.iter().all(|t| utf16_len(t) <= 20));
        assert_eq!(runs.concat(), text);
    }

    #[test]
    fn surrogate_pairs_are_never_split() {
        // 11 emoji = 22 UTF-16 units: 10 in the first event, 1 in the next,
        // each pair intact.
        let text = "😀".repeat(11);
        let segments = split_typed_segments(&text);
        let runs = text_runs(&segments);
        assert_eq!(runs.len(), 2);
        assert!(runs.iter().all(|t| utf16_len(t) <= 20));
        assert_eq!(runs.concat(), text);
    }

    #[test]
    fn newlines_and_tabs_become_keystrokes() {
        assert_eq!(
            split_typed_segments("one\ntwo\r\nthree\tend"),
            vec![
                TypedSegment::Text("one".to_string()),
                TypedSegment::Return,
                TypedSegment::Text("two".to_string()),
                TypedSegment::Return,
                TypedSegment::Text("three".to_string()),
                TypedSegment::Tab,
                TypedSegment::Text("end".to_string()),
            ]
        );
    }

    #[test]
    fn a_leading_newline_types_before_any_text() {
        // enigo drops events whose string starts with a newline, so the
        // very first segment must be the keystroke, never text.
        assert_eq!(
            split_typed_segments("\npara"),
            vec![TypedSegment::Return, TypedSegment::Text("para".to_string())]
        );
    }

    #[test]
    fn invisible_characters_are_dropped() {
        assert_eq!(
            split_typed_segments("a\u{200B}\u{FEFF}b\u{200D}c\u{0}"),
            vec![TypedSegment::Text("abc".to_string())]
        );
    }
}
