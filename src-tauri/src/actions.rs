#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
use crate::apple_intelligence;
use crate::audio_feedback::{play_feedback_sound, play_feedback_sound_blocking, SoundType};
use crate::audio_toolkit::VadPolicy;
use crate::managers::audio::{recording_error_type, AudioRecordingManager};
use crate::managers::history::HistoryManager;
use crate::managers::model::ModelManager;
use crate::managers::transcription::policy::{live_decode, LiveDecode, ModelTraits};
use crate::managers::transcription::StreamWorkKind;
use crate::managers::transcription::TranscriptionManager;
use crate::settings::{
    get_settings, AppSettings, ClipboardHandling, OverlayStyle, PasteMethod, PostProcessApiFormat,
    APPLE_INTELLIGENCE_PROVIDER_ID,
};
use crate::shortcut;
use crate::tray::{set_tray_state, TrayIconState};
use crate::utils::{
    self, show_processing_overlay, show_recording_overlay, show_transcribing_overlay,
};
use crate::TranscriptionCoordinator;
use log::{debug, error, info, warn};
use once_cell::sync::Lazy;
use std::collections::HashMap;
use std::future::Future;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tauri::Manager;
use tauri::{AppHandle, Emitter};

const CANCELLATION_POLL_INTERVAL: Duration = Duration::from_millis(25);

#[derive(Clone, serde::Serialize)]
struct RecordingErrorEvent {
    error_type: String,
    detail: Option<String>,
}

/// Drop guard that finishes the transcription pipeline, including immediate
/// model unloading on early exits.
struct FinishGuard(AppHandle, Arc<TranscriptionManager>);
impl Drop for FinishGuard {
    fn drop(&mut self) {
        self.1.maybe_unload_immediately("transcription session");
        if let Some(c) = self.0.try_state::<TranscriptionCoordinator>() {
            c.notify_processing_finished();
        }
        // The pipeline just freed its large transient buffers (captured PCM,
        // WAV copy, engine scratch); hand the cached pages back to the OS so
        // they don't sit in malloc arenas until they get swapped out (#1792).
        crate::memory::trim_freed_memory();
    }
}

/// Per-stage latency of one transcription pipeline run, logged as a single
/// info line once the output attempt finishes, e.g.
/// `timing: transcribe=812ms post_process=1430ms paste=35ms total=2277ms`.
///
/// `transcribe` covers the model run (live finalize or batch decode,
/// including any model-load wait), `post_process` the LLM post-processing
/// call (near zero when disabled), and the output stage the paste — or, for
/// streamed rewrites, the incremental typing window (`type=...`), with
/// `first_token` calling out how long the stream took to start producing
/// visible text. `total` spans the whole stop-to-output pipeline, so it also
/// includes what sits between the stages: recording teardown, WAV
/// save/verify, history persistence, and main-thread dispatch.
struct PipelineTimings {
    started: Instant,
    transcribe: Duration,
    post_process: Duration,
    /// Streamed rewrite: time to the first visible token, when one arrived.
    first_token: Option<Duration>,
    /// Whether the output was typed as it streamed (vs. pasted at the end).
    streamed: bool,
}

impl PipelineTimings {
    fn log(&self, output: Duration) {
        let transcribe = self.transcribe.as_millis();
        let post_process = self.post_process.as_millis();
        let total = self.started.elapsed().as_millis();
        if self.streamed {
            let first_token = self.first_token.map(|d| d.as_millis()).unwrap_or(0);
            info!(
                "timing: transcribe={transcribe}ms post_process={post_process}ms first_token={first_token}ms type={}ms total={total}ms",
                output.as_millis()
            );
        } else {
            info!(
                "timing: transcribe={transcribe}ms post_process={post_process}ms paste={}ms total={total}ms",
                output.as_millis()
            );
        }
    }

    /// Persists this run as a dictation metric for the Stats page. Runs on a
    /// background thread: the frontmost-app lookup talks to the AX API
    /// (macOS) and the insert hits SQLite, and neither belongs on the paste
    /// path. Failures are logged and forgotten - statistics must never
    /// disturb the pipeline.
    fn record(&self, app: &AppHandle, char_count: u64, post_processed: bool, output: Duration) {
        let metric = crate::stats::DictationMetric {
            timestamp: chrono::Utc::now().timestamp(),
            app_name: None,
            char_count,
            transcribe_ms: self.transcribe.as_millis() as u64,
            post_process_ms: self.post_process.as_millis() as u64,
            first_token_ms: self.first_token.map(|d| d.as_millis() as u64),
            output_ms: output.as_millis() as u64,
            total_ms: self.started.elapsed().as_millis() as u64,
            streamed: self.streamed,
            post_processed,
        };
        let app = app.clone();
        std::thread::spawn(move || {
            let mut metric = metric;
            metric.app_name = crate::ax_reader::focused_application_title();
            let Some(hm) = app.try_state::<Arc<HistoryManager>>() else {
                debug!("Stats: history manager unavailable; metric dropped");
                return;
            };
            if let Err(err) = hm.record_dictation_metric(&metric) {
                debug!("Stats: failed to record dictation metric: {err}");
            }
        });
    }
}

// Shortcut Action Trait
pub trait ShortcutAction: Send + Sync {
    fn start(&self, app: &AppHandle, binding_id: &str, shortcut_str: &str);
    fn stop(&self, app: &AppHandle, binding_id: &str, shortcut_str: &str);
}

// Transcribe Action
struct TranscribeAction {
    post_process: bool,
}

/// Field name for structured output JSON schema
const TRANSCRIPTION_FIELD: &str = "transcription";

/// Clean an LLM's final text output for use as paste/typed content: strip
/// every complete `<think>` block, then invisible characters some models
/// insert.
pub(crate) fn clean_llm_output(content: &str) -> String {
    strip_invisible_chars(&strip_think_blocks(content))
}

/// Strip invisible Unicode characters that some LLMs may insert
fn strip_invisible_chars(s: &str) -> String {
    s.replace(['\u{200B}', '\u{200C}', '\u{200D}', '\u{FEFF}'], "")
}

/// Strip every complete `<think>...</think>` block. Some endpoints can't
/// disable reasoning, and some local servers put the reasoning text into
/// `content` instead of a separate field — without this the user would get the
/// model's chain of thought pasted along with the cleaned transcription.
/// Blocks may also appear mid-text (models that think again after starting to
/// answer), so all of them go, wherever they sit. An unterminated block is
/// left verbatim rather than guessed at.
fn strip_think_blocks(s: &str) -> String {
    if !s.contains("<think>") {
        return s.to_string();
    }
    let mut visible = String::with_capacity(s.len());
    let mut rest = s;
    let mut stripped_any = false;
    while let Some(start) = rest.find("<think>") {
        visible.push_str(&rest[..start]);
        match rest[start..].find("</think>") {
            Some(end) => {
                rest = &rest[start + end + "</think>".len()..];
                stripped_any = true;
            }
            None => {
                // Unclosed block: leave the remainder untouched rather than guess.
                visible.push_str(&rest[start..]);
                rest = "";
                break;
            }
        }
    }
    visible.push_str(rest);
    if stripped_any {
        visible.trim_start().to_string()
    } else {
        visible
    }
}

/// Build a system prompt from the user's prompt template.
/// Removes `${output}` placeholder since the transcription is sent as the user message.
fn build_system_prompt(prompt_template: &str) -> String {
    prompt_template.replace("${output}", "").trim().to_string()
}

/// Returns `true` when a transcription has no meaningful content to
/// post-process (empty or whitespace-only). Used to skip the post-processing
/// LLM call when nothing was actually transcribed, which would otherwise make
/// the model reply with an error message such as "you need to provide the
/// transcription".
fn is_blank_transcription(transcription: &str) -> bool {
    transcription.trim().is_empty()
}

async fn complete_unless_cancelled<F, C>(operation: F, is_cancelled: C) -> Option<F::Output>
where
    F: Future,
    C: Fn() -> bool,
{
    tokio::pin!(operation);

    loop {
        if is_cancelled() {
            return None;
        }

        if let Ok(result) =
            tokio::time::timeout(CANCELLATION_POLL_INTERVAL, operation.as_mut()).await
        {
            return Some(result);
        }
    }
}

fn should_use_streaming_overlay(style: OverlayStyle, is_streaming: bool) -> bool {
    style == OverlayStyle::Live && is_streaming
}

/// Suffix appended to post-processing prompts that lists the user's custom
/// words as a glossary, so the rewrite model keeps their preferred
/// spellings. Empty when there is nothing to inject.
fn custom_words_glossary(custom_words: &[String]) -> String {
    const MAX_TERMS: usize = 50;
    const MAX_CHARS: usize = 1_000;

    let mut joined = String::new();
    let mut terms = 0;
    for word in custom_words {
        let term = word.trim();
        if term.is_empty() {
            continue;
        }
        if terms >= MAX_TERMS || joined.len() + term.len() + 2 > MAX_CHARS {
            break;
        }
        if !joined.is_empty() {
            joined.push_str(", ");
        }
        joined.push_str(term);
        terms += 1;
    }
    if joined.is_empty() {
        String::new()
    } else {
        format!("\n\nPreferred terms (use these exact spellings where they apply): {joined}")
    }
}

/// Everything a post-processing LLM call needs, resolved and validated from
/// the current settings. Each `None` carries a debug log explaining the skip.
struct PostProcessRequest {
    provider: crate::settings::PostProcessProvider,
    model: String,
    prompt: String,
    api_key: String,
    thinking: crate::settings::PostProcessThinking,
}

fn resolve_post_process_request(settings: &AppSettings) -> Option<PostProcessRequest> {
    let provider = match settings.active_post_process_provider().cloned() {
        Some(provider) => provider,
        None => {
            debug!("Post-processing enabled but no provider is selected");
            return None;
        }
    };

    let model = settings
        .post_process_models
        .get(&provider.id)
        .cloned()
        .unwrap_or_default();

    if model.trim().is_empty() {
        debug!(
            "Post-processing skipped because provider '{}' has no model configured",
            provider.id
        );
        return None;
    }

    let selected_prompt_id = match &settings.post_process_selected_prompt_id {
        Some(id) => id.clone(),
        None => {
            debug!("Post-processing skipped because no prompt is selected");
            return None;
        }
    };

    let prompt = match settings
        .post_process_prompts
        .iter()
        .find(|prompt| prompt.id == selected_prompt_id)
    {
        Some(prompt) => prompt.prompt.clone(),
        None => {
            debug!(
                "Post-processing skipped because prompt '{}' was not found",
                selected_prompt_id
            );
            return None;
        }
    };

    if prompt.trim().is_empty() {
        debug!("Post-processing skipped because the selected prompt is empty");
        return None;
    }

    let api_key = settings
        .post_process_api_keys
        .get(&provider.id)
        .cloned()
        .unwrap_or_default();

    Some(PostProcessRequest {
        provider,
        model,
        prompt,
        api_key,
        thinking: settings.post_process_thinking,
    })
}

/// The currently selected post-processing prompt template, for history
/// records.
fn selected_post_process_prompt(settings: &AppSettings) -> Option<String> {
    let prompt_id = settings.post_process_selected_prompt_id.as_ref()?;
    settings
        .post_process_prompts
        .iter()
        .find(|prompt| &prompt.id == prompt_id)
        .map(|prompt| prompt.prompt.clone())
}

async fn post_process_transcription(settings: &AppSettings, transcription: &str) -> Option<String> {
    if is_blank_transcription(transcription) {
        debug!("Post-processing skipped because the transcription is empty");
        return None;
    }

    let PostProcessRequest {
        provider,
        model,
        prompt,
        api_key,
        thinking,
    } = resolve_post_process_request(settings)?;

    debug!(
        "Starting LLM post-processing with provider '{}' (model: {})",
        provider.id, model
    );

    // Reference-only text around the cursor in the focused field (macOS;
    // silent skip on any failure or when disabled).
    let cursor_context = if settings.use_cursor_context {
        crate::cursor_context::read_cursor_context(settings.cursor_context_chars).await
    } else {
        None
    };

    if provider.supports_structured_output {
        debug!("Using structured outputs for provider '{}'", provider.id);

        let glossary = custom_words_glossary(&settings.custom_words);
        let system_prompt = format!("{}{glossary}", build_system_prompt(&prompt));
        let system_prompt = match &cursor_context {
            Some(context) => format!(
                "{system_prompt}{}",
                crate::cursor_context::context_prompt_block(context)
            ),
            None => system_prompt,
        };
        let user_content = transcription.to_string();

        // Handle Apple Intelligence separately since it uses native Swift APIs
        if provider.id == APPLE_INTELLIGENCE_PROVIDER_ID {
            #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
            {
                if !apple_intelligence::check_apple_intelligence_availability() {
                    debug!(
                        "Apple Intelligence selected but not currently available on this device"
                    );
                    return None;
                }

                let token_limit = model.trim().parse::<i32>().unwrap_or(0);
                return match apple_intelligence::process_text_with_system_prompt(
                    &system_prompt,
                    &user_content,
                    token_limit,
                ) {
                    Ok(result) => {
                        if result.trim().is_empty() {
                            debug!("Apple Intelligence returned an empty response");
                            None
                        } else {
                            let result = strip_invisible_chars(&result);
                            debug!(
                                "Apple Intelligence post-processing succeeded. Output length: {} chars",
                                result.len()
                            );
                            Some(result)
                        }
                    }
                    Err(err) => {
                        error!("Apple Intelligence post-processing failed: {}", err);
                        None
                    }
                };
            }

            #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
            {
                debug!("Apple Intelligence provider selected on unsupported platform");
                return None;
            }
        }

        // Define JSON schema for transcription output
        let json_schema = serde_json::json!({
            "type": "object",
            "properties": {
                (TRANSCRIPTION_FIELD): {
                    "type": "string",
                    "description": "The cleaned and processed transcription text"
                }
            },
            "required": [TRANSCRIPTION_FIELD],
            "additionalProperties": false
        });

        match crate::llm_client::send_chat_completion_with_schema(
            &provider,
            api_key.clone(),
            &model,
            user_content,
            Some(system_prompt),
            Some(json_schema),
            thinking,
        )
        .await
        {
            Ok(Some(content)) => {
                // Parse the JSON response to extract the transcription field
                let content = &strip_think_blocks(&content);
                match serde_json::from_str::<serde_json::Value>(content) {
                    Ok(json) => {
                        if let Some(transcription_value) =
                            json.get(TRANSCRIPTION_FIELD).and_then(|t| t.as_str())
                        {
                            let result = strip_invisible_chars(transcription_value);
                            debug!(
                                "Structured output post-processing succeeded for provider '{}'. Output length: {} chars",
                                provider.id,
                                result.len()
                            );
                            return Some(result);
                        } else {
                            error!("Structured output response missing 'transcription' field");
                            return Some(strip_invisible_chars(content));
                        }
                    }
                    Err(e) => {
                        error!(
                            "Failed to parse structured output JSON: {}. Returning raw content.",
                            e
                        );
                        return Some(strip_invisible_chars(content));
                    }
                }
            }
            Ok(None) => {
                error!("LLM API response has no content");
                return None;
            }
            Err(e) => {
                warn!(
                    "Structured output failed for provider '{}': {}. Falling back to legacy mode.",
                    provider.id, e
                );
                // Fall through to legacy mode below
            }
        }
    }

    // Legacy mode: Replace ${output} variable in the prompt with the actual text
    let glossary = custom_words_glossary(&settings.custom_words);
    let mut processed_prompt = format!("{}{glossary}", prompt.replace("${output}", transcription));
    if let Some(context) = &cursor_context {
        processed_prompt.push_str(&crate::cursor_context::context_prompt_block(context));
    }
    debug!("Processed prompt length: {} chars", processed_prompt.len());

    match crate::llm_client::send_chat_completion(
        &provider,
        api_key,
        &model,
        processed_prompt,
        thinking,
    )
    .await
    {
        Ok(Some(content)) => {
            let content = strip_invisible_chars(&strip_think_blocks(&content));
            debug!(
                "LLM post-processing succeeded for provider '{}'. Output length: {} chars",
                provider.id,
                content.len()
            );
            Some(content)
        }
        Ok(None) => {
            error!("LLM API response has no content");
            None
        }
        Err(e) => {
            error!(
                "LLM post-processing failed for provider '{}': {}. Falling back to original transcription.",
                provider.id,
                e
            );
            None
        }
    }
}

pub(crate) struct ProcessedTranscription {
    pub final_text: String,
    pub post_processed_text: Option<String>,
    pub post_process_prompt: Option<String>,
}
pub(crate) async fn process_transcription_output(
    app: &AppHandle,
    transcription: &str,
    post_process: bool,
) -> ProcessedTranscription {
    let settings = get_settings(app);
    let mut final_text = transcription.to_string();
    let mut post_processed_text: Option<String> = None;
    let mut post_process_prompt: Option<String> = None;

    if post_process {
        if let Some(processed_text) = post_process_transcription(&settings, &final_text).await {
            post_processed_text = Some(processed_text.clone());
            final_text = processed_text;
            post_process_prompt = selected_post_process_prompt(&settings);
        }
    }

    ProcessedTranscription {
        final_text,
        post_processed_text,
        post_process_prompt,
    }
}

// ------------------------------------------------- streamed rewrite output

/// Timing facts of a streamed rewrite, for the pipeline timing log.
struct StreamedTimings {
    first_token: Option<Duration>,
    typing: Duration,
}

/// Decides when buffered streamed text is worth a typing dispatch: after a
/// sentence or clause boundary once a minimum has accumulated, or at a hard
/// cap so a steady stream never stalls waiting for punctuation.
fn should_flush_stream_buffer(pending: &str) -> bool {
    const MIN_CHARS: usize = 12;
    const MAX_CHARS: usize = 48;
    let count = pending.chars().count();
    if count >= MAX_CHARS {
        return true;
    }
    if count < MIN_CHARS {
        return false;
    }
    pending.chars().last().is_some_and(|c| {
        matches!(
            c,
            '。' | '！' | '？' | '；' | '\n' | '.' | '!' | '?' | ';' | '，' | ','
        )
    })
}

/// Types streamed rewrite text at the cursor as it arrives, in small chunks.
///
/// Dispatches go through `run_on_main_thread` (the paste path's convention —
/// enigo's macOS layout resolution must run there), which preserves order, so
/// typed output always matches `received` up to `typed_chars`.
struct StreamTyper {
    app: AppHandle,
    /// All visible text received so far (invisible chars already stripped).
    received: String,
    /// How many chars of `received` have been dispatched for typing.
    typed_chars: usize,
    /// Received but not yet dispatched.
    pending: String,
    /// When the first dispatch happened, to measure the typing window.
    typing_started: Option<Instant>,
}

impl StreamTyper {
    fn new(app: AppHandle) -> Self {
        Self {
            app,
            received: String::new(),
            typed_chars: 0,
            pending: String::new(),
            typing_started: None,
        }
    }

    fn receive(&mut self, delta: &str) {
        let delta = strip_invisible_chars(delta);
        if delta.is_empty() {
            return;
        }
        self.received.push_str(&delta);
        self.pending.push_str(&delta);
        if should_flush_stream_buffer(&self.pending) {
            self.flush();
        }
    }

    /// Dispatches buffered text to the main thread for typing. A failed
    /// dispatch keeps the chunk buffered so a later fallback paste still
    /// includes it.
    fn flush(&mut self) {
        if self.pending.is_empty() {
            return;
        }
        let chunk = std::mem::take(&mut self.pending);
        let app = self.app.clone();
        let app_for_closure = app.clone();
        let chunk_for_closure = chunk.clone();
        let dispatched = app
            .run_on_main_thread(move || {
                if let Err(e) =
                    crate::clipboard::type_text_at_cursor(&app_for_closure, &chunk_for_closure)
                {
                    debug!("Streamed typing failed: {e}");
                }
            })
            .is_ok();
        if dispatched {
            if self.typing_started.is_none() {
                self.typing_started = Some(Instant::now());
            }
            self.typed_chars += chunk.chars().count();
        } else {
            self.pending = chunk;
        }
    }

    fn typed_anything(&self) -> bool {
        self.typed_chars > 0
    }

    fn received_text(&self) -> &str {
        &self.received
    }

    /// Everything received but not typed, for the mid-stream error fallback.
    fn untyped_text(&self) -> String {
        self.received.chars().skip(self.typed_chars).collect()
    }

    /// Elapsed time of the typing window (first dispatch to now).
    fn typing_duration(&self) -> Duration {
        self.typing_started
            .map(|start| start.elapsed())
            .unwrap_or(Duration::ZERO)
    }

    /// Applies the output-side settings the paste path applies — trailing
    /// space, clipboard copy, auto-submit — and kicks the auto-learn hook with
    /// the fully typed text.
    fn finalize(&self, settings: &AppSettings) {
        let text = self.received.clone();
        if settings.append_trailing_space {
            self.dispatch_typing(" ");
        }
        if settings.auto_submit {
            let app = self.app.clone();
            let app_for_closure = app.clone();
            let _ = app.run_on_main_thread(move || {
                if let Err(e) = crate::clipboard::send_auto_submit_key(&app_for_closure) {
                    debug!("Auto-submit after streamed typing failed: {e}");
                }
            });
        }
        if settings.clipboard_handling == ClipboardHandling::CopyToClipboard {
            if let Err(e) = crate::clipboard::write_text_to_clipboard(&self.app, &text) {
                debug!("Copying streamed text to clipboard failed: {e}");
            }
        }
        crate::auto_learn::schedule_after_paste(&self.app, text);
    }

    fn dispatch_typing(&self, text: &str) {
        let app = self.app.clone();
        let app_for_closure = app.clone();
        let chunk = text.to_string();
        let _ = app.run_on_main_thread(move || {
            if let Err(e) = crate::clipboard::type_text_at_cursor(&app_for_closure, &chunk) {
                debug!("Streamed typing failed: {e}");
            }
        });
    }
}

/// Result of attempting the streamed rewrite.
enum StreamedPipeline {
    /// The rewrite streamed and was fully typed at the cursor. `text` is what
    /// the user now sees; no further paste is needed.
    Done {
        text: String,
        first_token: Option<Duration>,
        typing: Duration,
    },
    /// Streaming was disabled, not applicable, or failed before typing
    /// anything — the caller falls back to the non-streaming path (which in
    /// turn falls back to pasting the raw transcript).
    Unavailable,
    /// Cancelled mid-stream: whatever was typed stays, nothing more happens.
    Cancelled,
}

/// Streams the post-processing rewrite and types it at the cursor as chunks
/// arrive, instead of waiting for the full response and pasting.
async fn stream_post_process_and_type(
    app: &AppHandle,
    transcription: &str,
    post_process: bool,
    is_cancelled: impl Fn() -> bool,
) -> StreamedPipeline {
    let settings = get_settings(app);
    if !post_process || !settings.stream_post_process {
        return StreamedPipeline::Unavailable;
    }
    if is_blank_transcription(transcription) {
        return StreamedPipeline::Unavailable;
    }
    if settings.paste_method == PasteMethod::None {
        // The user asked for no output injection; typing would violate that.
        debug!("Streamed rewrite skipped: paste method is None");
        return StreamedPipeline::Unavailable;
    }

    let Some(request) = resolve_post_process_request(&settings) else {
        return StreamedPipeline::Unavailable;
    };
    // Only the OpenAI-compatible Chat Completions endpoint streams; the
    // Anthropic Messages format and Apple Intelligence keep the plain path.
    if request.provider.api_format != PostProcessApiFormat::OpenAi
        || request.provider.id == APPLE_INTELLIGENCE_PROVIDER_ID
    {
        debug!(
            "Streamed rewrite unsupported for provider '{}'; using the normal path",
            request.provider.id
        );
        return StreamedPipeline::Unavailable;
    }

    // Same prompt shape as the non-streaming structured path: system prompt,
    // glossary, and reference-only cursor context.
    let glossary = custom_words_glossary(&settings.custom_words);
    let mut system_prompt = format!("{}{glossary}", build_system_prompt(&request.prompt));
    if settings.use_cursor_context {
        if let Some(context) =
            crate::cursor_context::read_cursor_context(settings.cursor_context_chars).await
        {
            system_prompt = format!(
                "{system_prompt}{}",
                crate::cursor_context::context_prompt_block(&context)
            );
        }
    }

    let mut typer = StreamTyper::new(app.clone());
    let result = crate::llm_client::send_chat_completion_streaming(
        &request.provider,
        request.api_key,
        &request.model,
        Some(system_prompt),
        transcription.to_string(),
        request.thinking,
        |delta| typer.receive(delta),
        is_cancelled,
    )
    .await;

    match result {
        Ok(Some(_outcome)) => {
            // The canonical text is what the typer received (already
            // invisible-char-stripped, think-stripped by the client).
            typer.flush();
            let text = typer.received_text().to_string();
            if text.is_empty() {
                // The endpoint answered without streaming any content (e.g. it
                // ignored stream:true); fall back rather than paste nothing —
                // and without applying the finalize niceties to empty output.
                debug!("Streamed rewrite produced no text; falling back to the normal path");
                return StreamedPipeline::Unavailable;
            }
            let typing = typer.typing_duration();
            typer.finalize(&settings);
            StreamedPipeline::Done {
                text,
                first_token: _outcome.first_token,
                typing,
            }
        }
        Ok(None) => StreamedPipeline::Cancelled,
        Err(e) => {
            if !typer.typed_anything() {
                warn!("Streamed rewrite failed before any output: {e}; falling back to the normal path");
                StreamedPipeline::Unavailable
            } else {
                // Keep the partial rewrite: paste whatever was received but
                // not yet typed, so the user ends up with the full partial.
                warn!("Streamed rewrite failed mid-output: {e}; pasting the remainder");
                let text = typer.received_text().to_string();
                let typing = typer.typing_duration();
                let remainder = typer.untyped_text();
                let paste_app = app.clone();
                let _ = app.run_on_main_thread(move || {
                    if let Err(paste_err) = utils::paste(remainder, paste_app.clone()) {
                        error!("Failed to paste streamed remainder: {paste_err}");
                    }
                });
                StreamedPipeline::Done {
                    text,
                    first_token: None,
                    typing,
                }
            }
        }
    }
}

impl ShortcutAction for TranscribeAction {
    fn start(&self, app: &AppHandle, binding_id: &str, _shortcut_str: &str) {
        let start_time = Instant::now();
        debug!("TranscribeAction::start called for binding: {}", binding_id);

        start_recording_session(app, binding_id);

        debug!(
            "TranscribeAction::start completed in {:?}",
            start_time.elapsed()
        );
    }

    fn stop(&self, app: &AppHandle, binding_id: &str, _shortcut_str: &str) {
        stop_and_process_transcription(app, binding_id, self.post_process);
    }
}

/// Shared recording start for the transcribe and voice-edit actions: model
/// and VAD preload, tray/overlay state, microphone open, cancel-shortcut
/// registration, and error rollback.
fn start_recording_session(app: &AppHandle, binding_id: &str) {
    {
        // Load model in the background
        let tm = app.state::<Arc<TranscriptionManager>>();
        let rm = app.state::<Arc<AudioRecordingManager>>();

        // Load ASR model and VAD model in parallel
        let kickoff_started = Instant::now();
        tm.initiate_model_load();
        let rm_clone = Arc::clone(&rm);
        std::thread::spawn(move || {
            if let Err(e) = rm_clone.preload_vad() {
                debug!("VAD pre-load failed: {}", e);
            }
        });
        let kickoff_elapsed = kickoff_started.elapsed();

        // Don't open the mic if nothing can transcribe the recording; the load
        // kicked off above fails and reports why.
        if !tm.is_model_loaded() {
            let selected_model = get_settings(app).selected_model;
            if let Err(e) = app
                .state::<Arc<ModelManager>>()
                .get_model_path(&selected_model)
            {
                warn!("Not starting recording: no model can transcribe it ({})", e);
                return;
            }
        }

        let binding_id = binding_id.to_string();
        let tray_started = Instant::now();
        set_tray_state(app, TrayIconState::Recording);
        let tray_elapsed = tray_started.elapsed();

        // Get the microphone mode to determine audio feedback timing
        let plan_started = Instant::now();
        let settings = get_settings(app);
        let is_always_on = settings.always_on_microphone;

        let selected_model_info = app
            .state::<Arc<ModelManager>>()
            .get_model_info(&settings.selected_model);

        // Use the app-facing model capability as the single pre-recording source
        // for live streaming decisions. Unknown support is represented as false
        // until the model registry is updated by discovery or runtime load.
        let model_supports_streaming = selected_model_info
            .as_ref()
            .map(|m| m.supports_streaming)
            .unwrap_or(false);
        let vad_policy = if !settings.vad_enabled {
            VadPolicy::Disabled
        } else if model_supports_streaming {
            VadPolicy::Streaming
        } else {
            VadPolicy::Offline
        };
        // Before the model loads, whether it keeps its accuracy on long audio
        // is unknown, so assume not: the worker decides again with the loaded
        // model, and a wrong guess costs only an idle worker.
        let expected_model = ModelTraits {
            streaming: model_supports_streaming,
            long_form: false,
        };
        if live_decode(expected_model, settings.long_form_chunking) != LiveDecode::None {
            tm.start_stream();
        }
        let plan_elapsed = plan_started.elapsed();

        // Sizing the overlay follows the same advertised capability. A model that
        // doesn't stream (or whose capability is not known yet) gets the compact
        // pill instead of an oversized transparent live window.
        let overlay_started = Instant::now();
        match settings.overlay_style {
            OverlayStyle::Live if model_supports_streaming => utils::show_streaming_overlay(app),
            OverlayStyle::Live | OverlayStyle::Minimal => show_recording_overlay(app),
            OverlayStyle::None => {} // show_overlay_state no-ops on None anyway
        }
        // Everything above runs before capture can begin, so each span here is
        // added keypress->capture latency.
        debug!(
            "start-path pre-recording steps: model_kickoff={:?} tray={:?} settings+stream_plan={:?} overlay={:?}",
            kickoff_elapsed,
            tray_elapsed,
            plan_elapsed,
            overlay_started.elapsed()
        );
        debug!("Microphone mode - always_on: {}", is_always_on);

        let mut recording_error: Option<anyhow::Error> = None;
        let recording_start_time = Instant::now();
        match rm.try_start_recording(&binding_id, vad_policy) {
            Ok(readiness) => {
                debug!(
                    "Recording request accepted in {:?}; waiting for first microphone samples",
                    recording_start_time.elapsed()
                );
                let generation = readiness.generation();
                let app_clone = app.clone();
                let rm_clone = Arc::clone(&rm);
                std::thread::spawn(move || {
                    if !readiness.wait() {
                        debug!("Microphone readiness wait ended without receiving samples");
                        return;
                    }

                    // Development-only preview hook for evaluating the brief
                    // arming animation on hardware that normally starts too fast
                    // to make it visible.
                    #[cfg(debug_assertions)]
                    if let Ok(delay_ms) = std::env::var("HANDY_DEBUG_MIC_READY_DELAY_MS")
                        .unwrap_or_default()
                        .parse::<u64>()
                    {
                        let delay_ms = delay_ms.min(10_000);
                        if delay_ms > 0 {
                            debug!("Delaying microphone-ready cue by {delay_ms}ms for UI preview");
                            std::thread::sleep(Duration::from_millis(delay_ms));
                        }
                    }

                    if !rm_clone.is_recording_readiness_current(generation) {
                        debug!("Microphone became ready for an inactive recording");
                        return;
                    }

                    debug!("Microphone is receiving samples; recording is ready");
                    utils::emit_recording_ready(&app_clone);

                    // The start chime is a readiness cue, so it must follow the
                    // first real input callback rather than Stream::play() or a
                    // fixed delay. The helper returns immediately when feedback
                    // is disabled; mute still follows the same readiness point.
                    if rm_clone.is_recording_readiness_current(generation) {
                        play_feedback_sound_blocking(&app_clone, SoundType::Start);
                    }
                    if rm_clone.is_recording_readiness_current(generation) {
                        rm_clone.apply_mute();
                    }
                });
            }
            Err(e) => {
                debug!("Failed to start recording: {}", e);
                recording_error = Some(e);
            }
        }

        if recording_error.is_none() {
            // Dynamically register the cancel shortcut in a separate task to avoid deadlock
            shortcut::register_cancel_shortcut(app);
        } else {
            // Starting failed (for example due to blocked microphone permissions).
            // Revert UI state so we don't stay stuck in the recording overlay.
            tm.cancel_stream();
            utils::hide_recording_overlay(app);
            set_tray_state(app, TrayIconState::Idle);
            if let Some(err) = recording_error {
                let _ = app.emit(
                    "recording-error",
                    RecordingErrorEvent {
                        error_type: recording_error_type(&err).to_string(),
                        detail: Some(format!("{err:#}")),
                    },
                );
            }
        }
    }
}

/// Shared recording stop and transcription pipeline for the transcribe and
/// voice-edit actions.
fn stop_and_process_transcription(app: &AppHandle, binding_id: &str, post_process: bool) {
    // Prevent a slow microphone from emitting a ready event or start chime
    // after the user has already requested stop.
    app.state::<Arc<AudioRecordingManager>>()
        .invalidate_recording_readiness();

    // Unregister the cancel shortcut when transcription stops
    shortcut::unregister_cancel_shortcut(app);

    debug!("Recording stop requested for binding: {}", binding_id);

    let ah = app.clone();
    let rm = Arc::clone(&app.state::<Arc<AudioRecordingManager>>());
    let tm = Arc::clone(&app.state::<Arc<TranscriptionManager>>());
    let hm = Arc::clone(&app.state::<Arc<HistoryManager>>());

    set_tray_state(app, TrayIconState::Transcribing);
    // Stop should give immediate visual feedback. Live streaming can keep
    // the larger panel, but it still switches from listening to a working
    // spinner while the stream finalizes. Non-streaming paths use the
    // compact transcribing pill (None no-ops in show_*).
    let style = get_settings(app).overlay_style;
    // Capture this before finalizing the stream so every later working state
    // targets the same overlay that was shown for this transcription.
    let use_streaming_overlay = should_use_streaming_overlay(style, tm.is_streaming());
    if use_streaming_overlay {
        tm.emit_stream_working(StreamWorkKind::Transcribing);
    } else {
        show_transcribing_overlay(app);
    }

    // Unmute before playing audio feedback so the stop sound is audible
    rm.remove_mute();

    // Play audio feedback for recording stop
    play_feedback_sound(app, SoundType::Stop);

    let binding_id = binding_id.to_string(); // Clone binding_id for the async task
    let cancel_generation = rm.cancel_generation();

    tauri::async_runtime::spawn(async move {
        let _guard = FinishGuard(ah.clone(), Arc::clone(&tm));
        let pipeline_started = Instant::now();
        debug!(
            "Starting async transcription task for binding: {}",
            binding_id
        );

        let stop_recording_time = Instant::now();
        if let Some(samples) = rm.stop_recording(&binding_id, cancel_generation) {
            debug!(
                "Recording stopped and samples retrieved in {:?}, sample count: {}",
                stop_recording_time.elapsed(),
                samples.len()
            );

            if rm.was_cancelled_since(cancel_generation) {
                debug!("Transcription operation cancelled after recording stop");
                tm.cancel_stream();
                utils::hide_recording_overlay(&ah);
                set_tray_state(&ah, TrayIconState::Idle);
                return;
            }

            if samples.is_empty() {
                debug!("Recording produced no audio samples; skipping persistence");
                // Tear down any streaming worker so its channel doesn't leak
                // and block the next start_stream.
                tm.cancel_stream();
                utils::hide_recording_overlay(&ah);
                set_tray_state(&ah, TrayIconState::Idle);
            } else {
                // Save WAV concurrently with transcription
                let sample_count = samples.len();
                let file_name = format!("handy-{}.wav", chrono::Utc::now().timestamp());
                let wav_path = hm.recordings_dir().join(&file_name);
                let wav_path_for_verify = wav_path.clone();
                let samples_for_wav = samples.clone();
                let wav_handle = tauri::async_runtime::spawn_blocking(move || {
                    crate::audio_toolkit::save_wav_file(&wav_path, &samples_for_wav)
                });

                // Transcribe concurrently with WAV save: the live decode's
                // result when it has one (all audio was already fed to it),
                // otherwise a batch run of the samples. A cancel is surfaced
                // as an error, so it never starts a batch run.
                let transcription_time = Instant::now();
                let transcription_result = tm.transcribe_recording(samples);
                let transcribe_elapsed = transcription_time.elapsed();

                // Await WAV save and verify
                let wav_saved = match wav_handle.await {
                    Ok(Ok(())) => {
                        match crate::audio_toolkit::verify_wav_file(
                            &wav_path_for_verify,
                            sample_count,
                        ) {
                            Ok(()) => true,
                            Err(e) => {
                                error!("WAV verification failed: {}", e);
                                false
                            }
                        }
                    }
                    Ok(Err(e)) => {
                        error!("Failed to save WAV file: {}", e);
                        false
                    }
                    Err(e) => {
                        error!("WAV save task panicked: {}", e);
                        false
                    }
                };

                if rm.was_cancelled_since(cancel_generation) {
                    debug!("Transcription operation cancelled before output handling");
                    utils::hide_recording_overlay(&ah);
                    set_tray_state(&ah, TrayIconState::Idle);
                    return;
                }

                match transcription_result {
                    Ok(transcription) => {
                        debug!(
                            "Transcription completed in {:?}: '{}'",
                            transcription_time.elapsed(),
                            utils::redact_text(&transcription)
                        );

                        if post_process {
                            if use_streaming_overlay {
                                tm.emit_stream_working(StreamWorkKind::Polishing);
                            } else {
                                show_processing_overlay(&ah);
                            }
                        }
                        let settings_at_stop = get_settings(&ah);
                        let mut streamed_timings: Option<StreamedTimings> = None;
                        let post_process_started = Instant::now();
                        // Streamed rewrite: type the LLM output at the
                        // cursor as chunks arrive instead of pasting at
                        // the end. Falls back to the normal path when
                        // disabled, unsupported, or failed before typing.
                        let streamed =
                            stream_post_process_and_type(&ah, &transcription, post_process, || {
                                rm.was_cancelled_since(cancel_generation)
                            })
                            .await;

                        let processed = match streamed {
                            StreamedPipeline::Cancelled => {
                                debug!("Transcription operation cancelled during streamed output");
                                utils::hide_recording_overlay(&ah);
                                set_tray_state(&ah, TrayIconState::Idle);
                                return;
                            }
                            StreamedPipeline::Done {
                                text,
                                first_token,
                                typing,
                            } => {
                                streamed_timings = Some(StreamedTimings {
                                    first_token,
                                    typing,
                                });
                                ProcessedTranscription {
                                    final_text: text.clone(),
                                    post_processed_text: Some(text),
                                    post_process_prompt: selected_post_process_prompt(
                                        &settings_at_stop,
                                    ),
                                }
                            }
                            StreamedPipeline::Unavailable => {
                                let Some(processed) = complete_unless_cancelled(
                                    process_transcription_output(&ah, &transcription, post_process),
                                    || rm.was_cancelled_since(cancel_generation),
                                )
                                .await
                                else {
                                    debug!(
                                        "Transcription operation cancelled during output handling"
                                    );
                                    utils::hide_recording_overlay(&ah);
                                    set_tray_state(&ah, TrayIconState::Idle);
                                    return;
                                };
                                processed
                            }
                        };
                        let timings = PipelineTimings {
                            started: pipeline_started,
                            transcribe: transcribe_elapsed,
                            post_process: post_process_started.elapsed(),
                            first_token: streamed_timings.as_ref().and_then(|t| t.first_token),
                            streamed: streamed_timings.is_some(),
                        };

                        if rm.was_cancelled_since(cancel_generation) {
                            debug!("Transcription operation cancelled before paste");
                            utils::hide_recording_overlay(&ah);
                            set_tray_state(&ah, TrayIconState::Idle);
                            return;
                        }

                        // Save to history if WAV was saved
                        if wav_saved {
                            if let Err(err) = hm.save_entry(
                                file_name,
                                transcription,
                                post_process,
                                processed.post_processed_text.clone(),
                                processed.post_process_prompt.clone(),
                            ) {
                                error!("Failed to save history entry: {}", err);
                            }
                        }

                        if processed.final_text.is_empty() {
                            utils::hide_recording_overlay(&ah);
                            set_tray_state(&ah, TrayIconState::Idle);
                        } else if let Some(streamed) = streamed_timings {
                            // Everything was already typed at the cursor
                            // while the rewrite streamed (or pasted as a
                            // mid-stream fallback); nothing left to emit.
                            timings.log(streamed.typing);
                            timings.record(
                                &ah,
                                processed.final_text.chars().count() as u64,
                                post_process,
                                streamed.typing,
                            );
                            utils::hide_recording_overlay(&ah);
                            set_tray_state(&ah, TrayIconState::Idle);
                        } else {
                            let ah_clone = ah.clone();
                            let final_text = processed.final_text;
                            let rm_for_paste = Arc::clone(&rm);
                            ah.run_on_main_thread(move || {
                                if rm_for_paste.was_cancelled_since(cancel_generation) {
                                    debug!("Transcription operation cancelled before paste");
                                    utils::hide_recording_overlay(&ah_clone);
                                    set_tray_state(&ah_clone, TrayIconState::Idle);
                                    return;
                                }

                                let paste_started = Instant::now();
                                let pasted_text = final_text.clone();
                                let pasted_chars = final_text.chars().count() as u64;
                                match utils::paste(final_text, ah_clone.clone()) {
                                    Ok(()) => {
                                        debug!(
                                            "Text pasted successfully in {:?}",
                                            paste_started.elapsed()
                                        );
                                        // Learn custom words from the user's
                                        // corrections of the pasted text (no-op
                                        // unless enabled, macOS only).
                                        crate::auto_learn::schedule_after_paste(
                                            &ah_clone,
                                            pasted_text,
                                        );
                                    }
                                    Err(e) => {
                                        error!("Failed to paste transcription: {}", e);
                                        let _ = ah_clone.emit("paste-error", ());
                                    }
                                }
                                let output_elapsed = paste_started.elapsed();
                                timings.log(output_elapsed);
                                timings.record(
                                    &ah_clone,
                                    pasted_chars,
                                    post_process,
                                    output_elapsed,
                                );
                                utils::hide_recording_overlay(&ah_clone);
                                set_tray_state(&ah_clone, TrayIconState::Idle);
                            })
                            .unwrap_or_else(|e| {
                                error!("Failed to run paste on main thread: {:?}", e);
                                utils::hide_recording_overlay(&ah);
                                set_tray_state(&ah, TrayIconState::Idle);
                            });
                        }
                    }
                    Err(err) => {
                        if rm.was_cancelled_since(cancel_generation) {
                            debug!("Transcription operation cancelled after transcription error");
                            utils::hide_recording_overlay(&ah);
                            set_tray_state(&ah, TrayIconState::Idle);
                            return;
                        }

                        error!("Transcription failed: {}", err);
                        // Surface the failure to the UI (toast). The full
                        // message is also in handy.log via the line above.
                        let _ = ah.emit("transcription-error", err.to_string());
                        // Save entry with empty text so user can retry
                        if wav_saved {
                            if let Err(save_err) =
                                hm.save_entry(file_name, String::new(), post_process, None, None)
                            {
                                error!("Failed to save failed history entry: {}", save_err);
                            }
                        }
                        utils::hide_recording_overlay(&ah);
                        set_tray_state(&ah, TrayIconState::Idle);
                    }
                }
            }
        } else {
            debug!("No samples retrieved from recording stop");
            // Tear down any streaming worker so its channel doesn't leak.
            tm.cancel_stream();
            utils::hide_recording_overlay(&ah);
            set_tray_state(&ah, TrayIconState::Idle);
        }
    });
}

// Voice Edit Action: capture the selection, record an instruction, rewrite.
struct VoiceEditAction;

impl ShortcutAction for VoiceEditAction {
    fn start(&self, app: &AppHandle, binding_id: &str, _shortcut_str: &str) {
        let start_time = Instant::now();
        debug!("VoiceEditAction::start called for binding: {}", binding_id);

        // The selection must be read before anything else shifts focus
        // (the overlay never steals focus, but capture first regardless).
        if !crate::voice_edit::capture_selection_for_recording(app) {
            return; // No selection (or a secure field): silently ignore.
        }

        start_recording_session(app, binding_id);

        debug!(
            "VoiceEditAction::start completed in {:?}",
            start_time.elapsed()
        );
    }

    fn stop(&self, app: &AppHandle, binding_id: &str, _shortcut_str: &str) {
        // Prevent a slow microphone from emitting a ready event or start chime
        // after the user has already requested stop.
        app.state::<Arc<AudioRecordingManager>>()
            .invalidate_recording_readiness();

        // Unregister the cancel shortcut when the recording stops.
        shortcut::unregister_cancel_shortcut(app);

        debug!("VoiceEditAction::stop called for binding: {}", binding_id);

        let ah = app.clone();
        let rm = Arc::clone(&app.state::<Arc<AudioRecordingManager>>());
        let tm = Arc::clone(&app.state::<Arc<TranscriptionManager>>());

        set_tray_state(app, TrayIconState::Transcribing);
        let style = get_settings(app).overlay_style;
        let use_streaming_overlay = should_use_streaming_overlay(style, tm.is_streaming());
        if use_streaming_overlay {
            tm.emit_stream_working(StreamWorkKind::Transcribing);
        } else {
            show_transcribing_overlay(app);
        }

        // Unmute before playing audio feedback so the stop sound is audible.
        rm.remove_mute();
        play_feedback_sound(app, SoundType::Stop);

        let binding_id = binding_id.to_string();
        let cancel_generation = rm.cancel_generation();

        tauri::async_runtime::spawn(async move {
            let _guard = FinishGuard(ah.clone(), Arc::clone(&tm));
            debug!("Starting async voice-edit task for binding: {}", binding_id);

            let Some(samples) = rm.stop_recording(&binding_id, cancel_generation) else {
                debug!("No samples retrieved from voice-edit recording stop");
                tm.cancel_stream();
                utils::hide_recording_overlay(&ah);
                set_tray_state(&ah, TrayIconState::Idle);
                return;
            };

            if rm.was_cancelled_since(cancel_generation) {
                debug!("Voice edit cancelled after recording stop");
                tm.cancel_stream();
                utils::hide_recording_overlay(&ah);
                set_tray_state(&ah, TrayIconState::Idle);
                return;
            }

            if samples.is_empty() {
                debug!("Voice-edit recording produced no audio samples; skipping");
                tm.cancel_stream();
                utils::hide_recording_overlay(&ah);
                set_tray_state(&ah, TrayIconState::Idle);
                return;
            }

            let transcription_result = tm.transcribe_recording(samples);

            if rm.was_cancelled_since(cancel_generation) {
                debug!("Voice edit cancelled before instruction handling");
                utils::hide_recording_overlay(&ah);
                set_tray_state(&ah, TrayIconState::Idle);
                return;
            }

            match transcription_result {
                Ok(instruction) => {
                    debug!(
                        "Voice-edit instruction transcribed: '{}'",
                        utils::redact_text(&instruction)
                    );
                    if instruction.trim().is_empty() {
                        debug!("Voice edit skipped: empty instruction");
                        utils::hide_recording_overlay(&ah);
                        set_tray_state(&ah, TrayIconState::Idle);
                        return;
                    }

                    show_processing_overlay(&ah);
                    let Some(edited) = complete_unless_cancelled(
                        crate::voice_edit::apply_voice_edit(&ah, &instruction),
                        || rm.was_cancelled_since(cancel_generation),
                    )
                    .await
                    else {
                        debug!("Voice edit cancelled during instruction handling");
                        utils::hide_recording_overlay(&ah);
                        set_tray_state(&ah, TrayIconState::Idle);
                        return;
                    };

                    let Some(edited) = edited.filter(|text| !text.trim().is_empty()) else {
                        error!("Voice edit produced no text; leaving the selection untouched");
                        utils::hide_recording_overlay(&ah);
                        set_tray_state(&ah, TrayIconState::Idle);
                        return;
                    };

                    // Replace the selection by pasting over it.
                    let ah_for_paste = ah.clone();
                    let rm_for_paste = Arc::clone(&rm);
                    ah.run_on_main_thread(move || {
                        if rm_for_paste.was_cancelled_since(cancel_generation) {
                            debug!("Voice edit cancelled before paste");
                            utils::hide_recording_overlay(&ah_for_paste);
                            set_tray_state(&ah_for_paste, TrayIconState::Idle);
                            return;
                        }

                        let paste_started = Instant::now();
                        match utils::paste(edited, ah_for_paste.clone()) {
                            Ok(()) => debug!(
                                "Voice-edit result pasted over the selection in {:?}",
                                paste_started.elapsed()
                            ),
                            Err(e) => {
                                error!("Failed to paste voice-edit result: {e}");
                                let _ = ah_for_paste.emit("paste-error", ());
                            }
                        }
                        utils::hide_recording_overlay(&ah_for_paste);
                        set_tray_state(&ah_for_paste, TrayIconState::Idle);
                    })
                    .unwrap_or_else(|e| {
                        error!("Failed to run voice-edit paste on main thread: {e:?}");
                        utils::hide_recording_overlay(&ah);
                        set_tray_state(&ah, TrayIconState::Idle);
                    });
                }
                Err(err) => {
                    if rm.was_cancelled_since(cancel_generation) {
                        debug!("Voice edit cancelled after transcription error");
                        utils::hide_recording_overlay(&ah);
                        set_tray_state(&ah, TrayIconState::Idle);
                        return;
                    }

                    error!("Voice-edit transcription failed: {err}");
                    let _ = ah.emit("transcription-error", err.to_string());
                    utils::hide_recording_overlay(&ah);
                    set_tray_state(&ah, TrayIconState::Idle);
                }
            }
        });
    }
}

// Cancel Action
struct CancelAction;

impl ShortcutAction for CancelAction {
    fn start(&self, app: &AppHandle, _binding_id: &str, _shortcut_str: &str) {
        utils::cancel_current_operation(app);
    }

    fn stop(&self, _app: &AppHandle, _binding_id: &str, _shortcut_str: &str) {
        // Nothing to do on stop for cancel
    }
}

// Test Action
struct TestAction;

impl ShortcutAction for TestAction {
    fn start(&self, app: &AppHandle, binding_id: &str, shortcut_str: &str) {
        log::info!(
            "Shortcut ID '{}': Started - {} (App: {})", // Changed "Pressed" to "Started" for consistency
            binding_id,
            shortcut_str,
            app.package_info().name
        );
    }

    fn stop(&self, app: &AppHandle, binding_id: &str, shortcut_str: &str) {
        log::info!(
            "Shortcut ID '{}': Stopped - {} (App: {})", // Changed "Released" to "Stopped" for consistency
            binding_id,
            shortcut_str,
            app.package_info().name
        );
    }
}

// Static Action Map
pub static ACTION_MAP: Lazy<HashMap<String, Arc<dyn ShortcutAction>>> = Lazy::new(|| {
    let mut map = HashMap::new();
    map.insert(
        "transcribe".to_string(),
        Arc::new(TranscribeAction {
            post_process: false,
        }) as Arc<dyn ShortcutAction>,
    );
    map.insert(
        "transcribe_with_post_process".to_string(),
        Arc::new(TranscribeAction { post_process: true }) as Arc<dyn ShortcutAction>,
    );
    map.insert(
        "voice_edit".to_string(),
        Arc::new(VoiceEditAction) as Arc<dyn ShortcutAction>,
    );
    map.insert(
        "cancel".to_string(),
        Arc::new(CancelAction) as Arc<dyn ShortcutAction>,
    );
    map.insert(
        "test".to_string(),
        Arc::new(TestAction) as Arc<dyn ShortcutAction>,
    );
    map
});

#[cfg(test)]
mod tests {
    use super::{
        complete_unless_cancelled, custom_words_glossary, is_blank_transcription,
        should_flush_stream_buffer, should_use_streaming_overlay, strip_think_blocks,
    };
    use crate::settings::OverlayStyle;
    use std::future;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Arc;
    use std::thread;
    use std::time::Duration;

    #[test]
    fn blank_transcription_is_detected() {
        assert!(is_blank_transcription(""));
        assert!(is_blank_transcription("   "));
        assert!(is_blank_transcription("\t\n  \r\n"));
    }

    #[test]
    fn glossary_is_empty_without_custom_words() {
        assert_eq!(custom_words_glossary(&[]), "");
        assert_eq!(custom_words_glossary(&["  ".to_string()]), "");
    }

    #[test]
    fn glossary_lists_custom_words() {
        let glossary = custom_words_glossary(&["Handy".to_string(), "微信".to_string()]);
        assert_eq!(
            glossary,
            "\n\nPreferred terms (use these exact spellings where they apply): Handy, 微信"
        );
    }

    #[test]
    fn glossary_skips_empty_entries_and_caps_size() {
        let words: Vec<String> = (0..300).map(|i| format!("word{i}")).collect();
        let glossary = custom_words_glossary(&words);
        assert!(glossary.len() < 1_200);
        assert!(glossary.contains("word0"));
        assert!(!glossary.contains("word299"));

        let words = vec!["a".to_string(), "  ".to_string(), "b".to_string()];
        assert!(custom_words_glossary(&words).ends_with("a, b"));
    }

    #[test]
    fn non_blank_transcription_is_kept() {
        assert!(!is_blank_transcription("hello"));
        assert!(!is_blank_transcription("  hello  "));
    }

    #[test]
    fn completed_operation_returns_its_output() {
        let result = tauri::async_runtime::block_on(complete_unless_cancelled(
            future::ready("done"),
            || false,
        ));

        assert_eq!(result, Some("done"));
    }

    #[test]
    fn pending_operation_stops_after_cancellation() {
        let cancelled = Arc::new(AtomicBool::new(false));
        let cancelled_for_thread = Arc::clone(&cancelled);
        let cancel_thread = thread::spawn(move || {
            thread::sleep(Duration::from_millis(10));
            cancelled_for_thread.store(true, Ordering::Release);
        });

        let result = tauri::async_runtime::block_on(complete_unless_cancelled(
            future::pending::<()>(),
            || cancelled.load(Ordering::Acquire),
        ));

        cancel_thread.join().unwrap();
        assert_eq!(result, None);
    }

    #[test]
    fn leading_think_block_is_stripped() {
        assert_eq!(
            strip_think_blocks("<think>pondering...</think>Cleaned text."),
            "Cleaned text."
        );
        assert_eq!(
            strip_think_blocks("  \n<think>multi\nline</think>\n  Cleaned text."),
            "Cleaned text."
        );
    }

    #[test]
    fn think_blocks_anywhere_are_stripped() {
        assert_eq!(
            strip_think_blocks("Answer<think>reconsider</think> continued."),
            "Answer continued."
        );
        assert_eq!(
            strip_think_blocks("<think>first</think>A<think>second</think>B<think>third</think>"),
            "AB"
        );
    }

    #[test]
    fn content_without_think_block_is_unchanged() {
        assert_eq!(strip_think_blocks("Cleaned text."), "Cleaned text.");
        assert_eq!(
            strip_think_blocks("Mentions <think> mid-sentence."),
            "Mentions <think> mid-sentence."
        );
        // Unclosed block: leave untouched rather than guess
        assert_eq!(
            strip_think_blocks("<think>never closed"),
            "<think>never closed"
        );
    }

    #[test]
    fn live_overlay_uses_streaming_states_only_for_streaming_models() {
        assert!(should_use_streaming_overlay(OverlayStyle::Live, true));
        assert!(!should_use_streaming_overlay(OverlayStyle::Live, false));
        assert!(!should_use_streaming_overlay(OverlayStyle::Minimal, true));
        assert!(!should_use_streaming_overlay(OverlayStyle::None, true));
    }

    #[test]
    fn stream_buffer_waits_for_a_minimum_before_flushing() {
        assert!(!should_flush_stream_buffer(""));
        assert!(!should_flush_stream_buffer("Hello."));
        assert!(!should_flush_stream_buffer(
            "twelve chars.".get(0..11).unwrap_or("")
        ));
        // At the minimum, a sentence/clause boundary flushes.
        assert!(should_flush_stream_buffer("twelve chars."));
        assert!(should_flush_stream_buffer("十二个字符了，现在可以。"));
        assert!(should_flush_stream_buffer("enough text\n"));
    }

    #[test]
    fn stream_buffer_flushes_at_the_hard_cap_without_punctuation() {
        assert!(should_flush_stream_buffer(&"x".repeat(48)));
        assert!(should_flush_stream_buffer(&"长".repeat(48)));
        // Just under the cap with no boundary still waits.
        assert!(!should_flush_stream_buffer(&"x".repeat(47)));
    }
}
