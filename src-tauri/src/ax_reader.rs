//! Shared, hand-rolled macOS Accessibility (AX) reader.
//!
//! Several features inspect the focused UI element of the frontmost app:
//! auto-learning of custom words reads the focused field's value, cursor
//! context for post-processing needs the selection range alongside it, and
//! voice editing needs the selected text. They all go through this one
//! minimal ApplicationServices + CoreFoundation FFI — a handful of attribute
//! reads is not worth a new dependency.
//!
//! Every failure path (missing accessibility permission, element without the
//! attribute, non-string value, messaging timeout) maps to `None`, which
//! callers treat as a silent skip: none of these features may disturb the
//! user when the AX read fails.

#[cfg(target_os = "macos")]
mod ffi {
    use std::ffi::{c_char, c_void};

    pub(super) type CFTypeRef = *const c_void;
    pub(super) type CFStringRef = *const c_void;
    pub(super) type AXUIElementRef = *const c_void;
    pub(super) type AXValueRef = *const c_void;
    pub(super) type CFIndex = i64;
    pub(super) type Boolean = u8;
    /// AXError; 0 is kAXErrorSuccess.
    pub(super) type AXError = i32;

    pub(super) const K_AX_ERROR_SUCCESS: AXError = 0;
    /// kAXValueCFRangeType from the AXValueType enum.
    pub(super) const K_AX_VALUE_CF_RANGE_TYPE: i32 = 4;
    /// kAXValueCFTypeType from the AXValueType enum.
    pub(super) const K_AX_VALUE_CF_TYPE_TYPE: i32 = 8;
    /// kCFStringEncodingUTF8.
    pub(super) const K_CF_STRING_ENCODING_UTF8: u32 = 0x0800_0100;
    /// Cap the value we are willing to read (guards against huge documents).
    pub(super) const MAX_VALUE_BYTES: usize = 2_000_000;
    /// CFStringGetLength counts UTF-16 units; this bounds the intermediate
    /// buffer before it is ever allocated.
    pub(super) const MAX_VALUE_UTF16_UNITS: CFIndex = 600_000;

    /// CFRange { location, length } in CFIndex units.
    #[repr(C)]
    #[derive(Debug, Clone, Copy)]
    pub(super) struct CFRange {
        pub location: CFIndex,
        pub length: CFIndex,
    }

    #[link(name = "ApplicationServices", kind = "framework")]
    extern "C" {
        pub(super) fn AXUIElementCreateSystemWide() -> AXUIElementRef;
        pub(super) fn AXUIElementCopyAttributeValue(
            element: AXUIElementRef,
            attribute: CFStringRef,
            value: *mut CFTypeRef,
        ) -> AXError;
        /// Best-effort cap on how long a single AX message may block.
        pub(super) fn AXUIElementSetMessagingTimeout(
            element: AXUIElementRef,
            timeout: f64,
        ) -> AXError;
        pub(super) fn AXValueGetValue(
            value: AXValueRef,
            the_type: i32,
            value_ptr: *mut c_void,
        ) -> Boolean;
    }

    #[link(name = "CoreFoundation", kind = "framework")]
    extern "C" {
        pub(super) fn CFRetain(cf: CFTypeRef);
        pub(super) fn CFRelease(cf: CFTypeRef);
        pub(super) fn CFGetTypeID(cf: CFTypeRef) -> usize;
        pub(super) fn CFStringGetTypeID() -> usize;
        pub(super) fn CFStringCreateWithCString(
            alloc: CFTypeRef,
            c_str: *const c_char,
            encoding: u32,
        ) -> CFStringRef;
        pub(super) fn CFStringGetLength(the_string: CFStringRef) -> CFIndex;
        pub(super) fn CFStringGetMaximumSizeForEncoding(length: CFIndex, encoding: u32) -> CFIndex;
        pub(super) fn CFStringGetCString(
            the_string: CFStringRef,
            buffer: *mut c_char,
            buffer_size: CFIndex,
            encoding: u32,
        ) -> Boolean;
    }
}

/// Text-related attributes of the focused UI element, read in one pass.
#[derive(Debug, Default, Clone)]
pub(crate) struct FocusedElementSnapshot {
    /// The element's full text content (AXValue), when it has one.
    pub value: Option<String>,
    /// The element's role (AXRole), e.g. "AXTextField".
    pub role: Option<String>,
    /// The element's subrole (AXSubrole).
    pub subrole: Option<String>,
    /// The currently selected text (AXSelectedText), when exposed.
    pub selected_text: Option<String>,
    /// (location, length) of the selection in UTF-16 units into `value`.
    pub selected_range: Option<(usize, usize)>,
}

/// Reads the focused UI element's value attribute as a Rust string.
#[cfg(target_os = "macos")]
pub(crate) fn focused_field_value() -> Option<String> {
    focused_element_snapshot().and_then(|snapshot| snapshot.value)
}

/// Reads the text-related attributes of the system-wide focused UI element.
/// Returns `None` when the focused element cannot be queried at all (which
/// includes a missing accessibility permission).
#[cfg(target_os = "macos")]
pub(crate) fn focused_element_snapshot() -> Option<FocusedElementSnapshot> {
    // SAFETY: each block releases what it acquired; pointers are only
    // dereferenced through CF/AX calls, which validate their inputs.
    unsafe {
        let system_wide = ffi::AXUIElementCreateSystemWide();
        if system_wide.is_null() {
            return None;
        }
        let _ = ffi::AXUIElementSetMessagingTimeout(system_wide, 1.0);

        let focused = copy_attribute(system_wide, "AXFocusedUIElement")?;
        let snapshot = read_focused_attributes(focused as ffi::AXUIElementRef);
        ffi::CFRelease(focused);
        ffi::CFRelease(system_wide);
        snapshot
    }
}

/// No-op outside macOS: the AX reader only exists there. Callers gate their
/// features on the platform themselves; this keeps non-macOS builds of
/// shared call sites compiling.
#[cfg(not(target_os = "macos"))]
pub(crate) fn focused_element_snapshot() -> Option<FocusedElementSnapshot> {
    None
}

// ------------------------------------------------------------- macOS glue

/// Reads every attribute the snapshot describes from one focused element.
#[cfg(target_os = "macos")]
unsafe fn read_focused_attributes(focused: ffi::AXUIElementRef) -> Option<FocusedElementSnapshot> {
    let mut snapshot = FocusedElementSnapshot::default();

    if let Some(value) = copy_attribute(focused, "AXValue") {
        snapshot.value = coerce_string(value);
        ffi::CFRelease(value);
    }
    if let Some(role) = copy_attribute(focused, "AXRole") {
        snapshot.role = coerce_string(role);
        ffi::CFRelease(role);
    }
    if let Some(subrole) = copy_attribute(focused, "AXSubrole") {
        snapshot.subrole = coerce_string(subrole);
        ffi::CFRelease(subrole);
    }
    if let Some(selected) = copy_attribute(focused, "AXSelectedText") {
        snapshot.selected_text = coerce_string(selected);
        ffi::CFRelease(selected);
    }
    snapshot.selected_range = copy_range_attribute(focused, "AXSelectedTextRange");

    Some(snapshot)
}

/// Copies an attribute value the caller must release, or `None` when the
/// element does not expose it (or the read failed).
#[cfg(target_os = "macos")]
unsafe fn copy_attribute(element: ffi::AXUIElementRef, name: &str) -> Option<ffi::CFTypeRef> {
    let attr = attr_string(name)?;
    let mut value: ffi::CFTypeRef = std::ptr::null();
    let err = ffi::AXUIElementCopyAttributeValue(element, attr, &mut value);
    ffi::CFRelease(attr);
    (err == ffi::K_AX_ERROR_SUCCESS && !value.is_null()).then_some(value)
}

/// Reads a CFRange-valued attribute (AXSelectedTextRange) as
/// (location, length), clamped to non-negative values.
#[cfg(target_os = "macos")]
unsafe fn copy_range_attribute(element: ffi::AXUIElementRef, name: &str) -> Option<(usize, usize)> {
    let value = copy_attribute(element, name)?;
    let mut range = ffi::CFRange {
        location: 0,
        length: 0,
    };
    let ok = ffi::AXValueGetValue(
        value as ffi::AXValueRef,
        ffi::K_AX_VALUE_CF_RANGE_TYPE,
        &mut range as *mut ffi::CFRange as *mut std::ffi::c_void,
    );
    ffi::CFRelease(value);
    if ok == 0 {
        return None;
    }
    Some((range.location.max(0) as usize, range.length.max(0) as usize))
}

/// Creates a CFString attribute name from a UTF-8 literal.
#[cfg(target_os = "macos")]
unsafe fn attr_string(name: &str) -> Option<ffi::CFStringRef> {
    let c_string = std::ffi::CString::new(name).ok()?;
    // SAFETY: `c_string` outlives the call and the allocator argument is
    // null (kCFAllocatorDefault). The returned string is owned (+1).
    let cf = ffi::CFStringCreateWithCString(
        std::ptr::null(),
        c_string.as_ptr(),
        ffi::K_CF_STRING_ENCODING_UTF8,
    );
    (!cf.is_null()).then_some(cf)
}

/// Converts a CFTypeRef from AXValue into a String: either a direct
/// CFString or a CFType-wrapped one inside an AXValue.
#[cfg(target_os = "macos")]
unsafe fn coerce_string(value: ffi::CFTypeRef) -> Option<String> {
    if ffi::CFGetTypeID(value) == ffi::CFStringGetTypeID() {
        return cf_string_to_string(value as ffi::CFStringRef);
    }

    let mut inner: ffi::CFTypeRef = std::ptr::null();
    let ok = ffi::AXValueGetValue(
        value as ffi::AXValueRef,
        ffi::K_AX_VALUE_CF_TYPE_TYPE,
        &mut inner as *mut ffi::CFTypeRef as *mut std::ffi::c_void,
    );
    if ok == 0 || inner.is_null() {
        return None;
    }
    // The AXValue still owns `inner`; retain before the caller releases
    // the wrapper.
    ffi::CFRetain(inner);
    let result = (ffi::CFGetTypeID(inner) == ffi::CFStringGetTypeID())
        .then(|| unsafe { cf_string_to_string(inner as ffi::CFStringRef) })
        .flatten();
    ffi::CFRelease(inner);
    result
}

#[cfg(target_os = "macos")]
unsafe fn cf_string_to_string(cf: ffi::CFStringRef) -> Option<String> {
    let length = ffi::CFStringGetLength(cf);
    if length == 0 {
        return Some(String::new());
    }
    if length > ffi::MAX_VALUE_UTF16_UNITS {
        return None;
    }
    let max_bytes =
        ffi::CFStringGetMaximumSizeForEncoding(length, ffi::K_CF_STRING_ENCODING_UTF8) + 1;
    let mut buffer: Vec<u8> = vec![0; max_bytes as usize];
    let ok = ffi::CFStringGetCString(
        cf,
        buffer.as_mut_ptr() as *mut std::ffi::c_char,
        max_bytes,
        ffi::K_CF_STRING_ENCODING_UTF8,
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
    (string.len() <= ffi::MAX_VALUE_BYTES).then_some(string)
}
