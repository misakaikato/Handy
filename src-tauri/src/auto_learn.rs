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
//! only the AX glue below is macOS-specific.

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
    let Some(field_value) = ax::focused_field_value() else {
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

// ------------------------------------------------------- macOS AX glue

/// Minimal Accessibility API reader for the focused element's value.
///
/// Hand-written FFI (ApplicationServices + CoreFoundation) rather than a new
/// dependency: only two attribute reads are needed. Every failure path
/// (missing accessibility permission, element without a value, non-string
/// value) maps to `None`, which callers treat as a silent skip.
#[cfg(target_os = "macos")]
mod ax {
    use std::ffi::{c_char, c_void, CString};

    type CFTypeRef = *const c_void;
    type CFStringRef = *const c_void;
    type AXUIElementRef = *const c_void;
    type AXValueRef = *const c_void;
    type CFIndex = i64;
    type Boolean = u8;
    /// AXError; 0 is kAXErrorSuccess.
    type AXError = i32;

    const K_AX_ERROR_SUCCESS: AXError = 0;
    /// kAXValueCFTypeType from the AXValueType enum.
    const K_AX_VALUE_CF_TYPE_TYPE: i32 = 8;
    /// kCFStringEncodingUTF8.
    const K_CF_STRING_ENCODING_UTF8: u32 = 0x0800_0100;
    /// Cap the value we are willing to diff (guards against huge documents).
    const MAX_VALUE_BYTES: usize = 2_000_000;
    /// CFStringGetLength counts UTF-16 units; this bounds the intermediate
    /// buffer before it is ever allocated.
    const MAX_VALUE_UTF16_UNITS: CFIndex = 600_000;

    #[link(name = "ApplicationServices", kind = "framework")]
    extern "C" {
        fn AXUIElementCreateSystemWide() -> AXUIElementRef;
        fn AXUIElementCopyAttributeValue(
            element: AXUIElementRef,
            attribute: CFStringRef,
            value: *mut CFTypeRef,
        ) -> AXError;
        /// Best-effort cap on how long a single AX message may block.
        fn AXUIElementSetMessagingTimeout(element: AXUIElementRef, timeout: f64) -> AXError;
        fn AXValueGetValue(value: AXValueRef, the_type: i32, value_ptr: *mut c_void) -> Boolean;
    }

    #[link(name = "CoreFoundation", kind = "framework")]
    extern "C" {
        fn CFRetain(cf: CFTypeRef);
        fn CFRelease(cf: CFTypeRef);
        fn CFGetTypeID(cf: CFTypeRef) -> usize;
        fn CFStringGetTypeID() -> usize;
        fn CFStringCreateWithCString(
            alloc: CFTypeRef,
            c_str: *const c_char,
            encoding: u32,
        ) -> CFStringRef;
        fn CFStringGetLength(the_string: CFStringRef) -> CFIndex;
        fn CFStringGetMaximumSizeForEncoding(length: CFIndex, encoding: u32) -> CFIndex;
        fn CFStringGetCString(
            the_string: CFStringRef,
            buffer: *mut c_char,
            buffer_size: CFIndex,
            encoding: u32,
        ) -> Boolean;
    }

    /// Creates a CFString attribute name from a UTF-8 literal.
    fn attr_string(name: &str) -> Option<CFStringRef> {
        let c_string = CString::new(name).ok()?;
        // SAFETY: `c_string` outlives the call and the allocator argument is
        // null (kCFAllocatorDefault). The returned string is owned (+1).
        let cf = unsafe {
            CFStringCreateWithCString(
                std::ptr::null(),
                c_string.as_ptr(),
                K_CF_STRING_ENCODING_UTF8,
            )
        };
        (!cf.is_null()).then_some(cf)
    }

    /// Reads the focused UI element's value attribute as a Rust string.
    pub fn focused_field_value() -> Option<String> {
        // SAFETY: each block releases what it acquired; pointers are only
        // dereferenced through CF/AX calls, which validate their inputs.
        unsafe {
            let system_wide = AXUIElementCreateSystemWide();
            if system_wide.is_null() {
                return None;
            }
            let _ = AXUIElementSetMessagingTimeout(system_wide, 1.0);

            let result = read_focused_value(system_wide);
            CFRelease(system_wide);
            result
        }
    }

    unsafe fn read_focused_value(system_wide: AXUIElementRef) -> Option<String> {
        let focused_attr = attr_string("AXFocusedUIElement")?;
        let mut focused: CFTypeRef = std::ptr::null();
        let err = AXUIElementCopyAttributeValue(system_wide, focused_attr, &mut focused);
        CFRelease(focused_attr);
        if err != K_AX_ERROR_SUCCESS || focused.is_null() {
            return None; // Includes kAXErrorAPIDisabled without accessibility permission.
        }

        let value_attr = match attr_string("AXValue") {
            Some(attr) => attr,
            None => {
                CFRelease(focused);
                return None;
            }
        };
        let mut value: CFTypeRef = std::ptr::null();
        let err = AXUIElementCopyAttributeValue(focused as AXUIElementRef, value_attr, &mut value);
        CFRelease(focused);
        let result = if err == K_AX_ERROR_SUCCESS && !value.is_null() {
            coerce_string(value)
        } else {
            None
        };
        if !value.is_null() {
            CFRelease(value);
        }
        CFRelease(value_attr);
        result
    }

    /// Converts a CFTypeRef from AXValue into a String: either a direct
    /// CFString or a CFType-wrapped one inside an AXValue.
    unsafe fn coerce_string(value: CFTypeRef) -> Option<String> {
        if CFGetTypeID(value) == CFStringGetTypeID() {
            return cf_string_to_string(value as CFStringRef);
        }

        let mut inner: CFTypeRef = std::ptr::null();
        let ok = AXValueGetValue(
            value as AXValueRef,
            K_AX_VALUE_CF_TYPE_TYPE,
            &mut inner as *mut CFTypeRef as *mut c_void,
        );
        if ok == 0 || inner.is_null() {
            return None;
        }
        // The AXValue still owns `inner`; retain before the caller releases
        // the wrapper.
        CFRetain(inner);
        let result = (CFGetTypeID(inner) == CFStringGetTypeID())
            .then(|| cf_string_to_string(inner as CFStringRef))
            .flatten();
        CFRelease(inner);
        result
    }

    unsafe fn cf_string_to_string(cf: CFStringRef) -> Option<String> {
        let length = CFStringGetLength(cf);
        if length == 0 {
            return Some(String::new());
        }
        if length > MAX_VALUE_UTF16_UNITS {
            return None;
        }
        let max_bytes = CFStringGetMaximumSizeForEncoding(length, K_CF_STRING_ENCODING_UTF8) + 1;
        let mut buffer: Vec<u8> = vec![0; max_bytes as usize];
        let ok = CFStringGetCString(
            cf,
            buffer.as_mut_ptr() as *mut c_char,
            max_bytes,
            K_CF_STRING_ENCODING_UTF8,
        );
        if ok == 0 {
            return None;
        }
        let end = buffer
            .iter()
            .position(|&byte| byte == 0)
            .unwrap_or(buffer.len());
        buffer.truncate(end);
        let string = String::from_utf8(buffer).ok()?;
        (string.len() <= MAX_VALUE_BYTES).then_some(string)
    }
}
