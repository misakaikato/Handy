use crate::settings::{PostProcessApiFormat, PostProcessProvider};
use futures_util::StreamExt;
use log::{debug, error, info};
use reqwest::header::{HeaderMap, HeaderValue, AUTHORIZATION, CONTENT_TYPE, REFERER, USER_AGENT};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashSet;
use std::error::Error as StdError;
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

/// How often a streaming response is polled for cancellation while waiting
/// for the next chunk.
const STREAM_CANCEL_POLL_INTERVAL: Duration = Duration::from_millis(25);

#[derive(Debug, Serialize)]
struct ChatMessage {
    role: String,
    content: String,
}

#[derive(Debug, Serialize)]
struct JsonSchema {
    name: String,
    strict: bool,
    schema: Value,
}

#[derive(Debug, Serialize)]
struct ResponseFormat {
    #[serde(rename = "type")]
    format_type: String,
    json_schema: JsonSchema,
}

#[derive(Debug, Serialize, Clone, Default, PartialEq)]
struct ReasoningConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    effort: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    exclude: Option<bool>,
}

/// Request fields used to ask an endpoint to skip reasoning/thinking.
/// Providers disagree on the field name and accepted values, so at most one of
/// these is set per request (see `reasoning_disable_params`).
#[derive(Debug, Serialize, Clone, Default, PartialEq)]
struct ReasoningParams {
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning_effort: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning: Option<ReasoningConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    thinking: Option<Value>,
}

impl ReasoningParams {
    fn is_empty(&self) -> bool {
        self.reasoning_effort.is_none() && self.reasoning.is_none() && self.thinking.is_none()
    }
}

/// Pick the reasoning-disable request fields an endpoint understands.
/// Unknown endpoints get the common OpenAI-style field; if they reject it,
/// the request is retried without it (see `send_chat_completion_with_schema`).
fn reasoning_disable_params(provider: &PostProcessProvider) -> ReasoningParams {
    let base_url = provider.base_url.to_lowercase();
    if base_url.contains("api.deepseek.com") {
        // DeepSeek rejects reasoning_effort "none" and uses its own field:
        // https://api-docs.deepseek.com/guides/thinking_mode
        ReasoningParams {
            thinking: Some(serde_json::json!({ "type": "disabled" })),
            ..Default::default()
        }
    } else if provider.id == "openrouter" {
        // OpenRouter nested object; exclude:true also keeps reasoning text out
        // of the response so it can't pollute structured-output JSON parsing
        ReasoningParams {
            reasoning: Some(ReasoningConfig {
                effort: Some("none".to_string()),
                exclude: Some(true),
            }),
            ..Default::default()
        }
    } else {
        ReasoningParams {
            reasoning_effort: Some("none".to_string()),
            ..Default::default()
        }
    }
}

/// Endpoints (base_url|model) that rejected the reasoning-disable fields with a
/// 4xx. Remembered for the lifetime of the process so every dictation after the
/// first skips the doomed attempt and goes straight to a plain request.
fn reasoning_rejections() -> &'static Mutex<HashSet<String>> {
    static REJECTED: OnceLock<Mutex<HashSet<String>>> = OnceLock::new();
    REJECTED.get_or_init(|| Mutex::new(HashSet::new()))
}

fn endpoint_key(provider: &PostProcessProvider, model: &str) -> String {
    format!("{}|{}", provider.base_url.trim_end_matches('/'), model)
}

fn is_known_rejected(key: &str) -> bool {
    reasoning_rejections()
        .lock()
        .map(|set| set.contains(key))
        .unwrap_or(false)
}

fn remember_rejection(key: String) {
    if let Ok(mut set) = reasoning_rejections().lock() {
        set.insert(key);
    }
}

#[derive(Debug, Serialize)]
struct ChatCompletionRequest {
    model: String,
    messages: Vec<ChatMessage>,
    stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    response_format: Option<ResponseFormat>,
    #[serde(flatten)]
    reasoning: ReasoningParams,
}

#[derive(Debug, Deserialize)]
struct ChatCompletionResponse {
    choices: Vec<ChatChoice>,
}

#[derive(Debug, Deserialize)]
struct ChatChoice {
    message: ChatMessageResponse,
    finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ChatMessageResponse {
    content: Option<String>,
}

/// Output budget for Anthropic Messages requests, where `max_tokens` is
/// required. Ample for a cleaned-up dictation.
const ANTHROPIC_MAX_TOKENS: u32 = 4096;

#[derive(Debug, Serialize)]
struct AnthropicMessagesRequest {
    model: String,
    max_tokens: u32,
    messages: Vec<ChatMessage>,
}

#[derive(Debug, Deserialize)]
struct AnthropicMessagesResponse {
    content: Vec<AnthropicContentBlock>,
    stop_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
enum AnthropicContentBlock {
    Text {
        text: String,
    },
    /// Thinking and any other non-text blocks
    #[serde(other)]
    Other,
}

impl AnthropicMessagesResponse {
    fn into_text(self) -> Option<String> {
        let text: String = self
            .content
            .into_iter()
            .filter_map(|block| match block {
                AnthropicContentBlock::Text { text } => Some(text),
                AnthropicContentBlock::Other => None,
            })
            .collect();
        (!text.is_empty()).then_some(text)
    }
}

/// Build headers for API requests based on provider type
fn build_headers(provider: &PostProcessProvider, api_key: &str) -> Result<HeaderMap, String> {
    let mut headers = HeaderMap::new();

    // Common headers
    headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
    headers.insert(
        REFERER,
        HeaderValue::from_static("https://github.com/cjpais/Handy"),
    );
    headers.insert(
        USER_AGENT,
        HeaderValue::from_static("Handy/1.0 (+https://github.com/cjpais/Handy)"),
    );
    headers.insert("X-Title", HeaderValue::from_static("Handy"));

    // Provider-specific auth headers
    if !api_key.is_empty() {
        if provider.api_format == PostProcessApiFormat::Anthropic {
            headers.insert(
                "x-api-key",
                HeaderValue::from_str(api_key)
                    .map_err(|e| format!("Invalid API key header value: {}", e))?,
            );
            headers.insert("anthropic-version", HeaderValue::from_static("2023-06-01"));
        } else {
            headers.insert(
                AUTHORIZATION,
                HeaderValue::from_str(&format!("Bearer {}", api_key))
                    .map_err(|e| format!("Invalid authorization header value: {}", e))?,
            );
        }
    }

    Ok(headers)
}

/// Create an HTTP client with provider-specific headers
fn create_client(provider: &PostProcessProvider, api_key: &str) -> Result<reqwest::Client, String> {
    let headers = build_headers(provider, api_key)?;
    reqwest::Client::builder()
        .default_headers(headers)
        .build()
        .map_err(|e| report_reqwest_error("Failed to build HTTP client", &e))
}

/// Format a bounded error source chain.
///
/// `reqwest::Error`'s Display implementation intentionally gives only a short
/// summary. Nested causes contain the useful transport details, such as a
/// certificate validation failure, an HTTP/2 error, or a connection reset.
/// Callers must skip source types whose Display text can quote payload data.
fn error_source_chain(error: &(dyn StdError + 'static)) -> Vec<String> {
    let mut causes = Vec::new();
    let mut source = error.source();

    // Defensive cap in case a third-party error exposes a cyclic source chain.
    for _ in 0..16 {
        let Some(cause) = source else {
            break;
        };
        causes.push(cause.to_string());
        source = cause.source();
    }

    causes
}

fn reqwest_error_kinds(error: &reqwest::Error) -> String {
    let mut kinds = Vec::new();

    if error.is_builder() {
        kinds.push("builder");
    }
    if error.is_connect() {
        kinds.push("connect");
    }
    if error.is_request() {
        kinds.push("request");
    }
    if error.is_redirect() {
        kinds.push("redirect");
    }
    if error.is_timeout() {
        kinds.push("timeout");
    }
    if error.is_status() {
        kinds.push("status");
    }
    if error.is_body() {
        kinds.push("body");
    }
    if error.is_decode() {
        kinds.push("decode");
    }
    if error.is_upgrade() {
        kinds.push("upgrade");
    }

    if kinds.is_empty() {
        "unknown".to_string()
    } else {
        kinds.join(", ")
    }
}

fn sanitized_url(url: &reqwest::Url) -> String {
    let mut url = url.clone();

    // Custom endpoints should not contain credentials or query-string tokens,
    // but omit them from diagnostics in case one does.
    let _ = url.set_username("");
    let _ = url.set_password(None);
    url.set_query(None);
    url.set_fragment(None);

    url.to_string()
}

fn sanitized_url_for_log(url: &str) -> String {
    reqwest::Url::parse(url)
        .map(|url| sanitized_url(&url))
        // Do not echo an invalid URL: the parse failure might have been caused
        // by sensitive data entered in the custom endpoint field.
        .unwrap_or_else(|_| "<invalid URL>".to_string())
}

fn report_reqwest_error(context: &str, error: &reqwest::Error) -> String {
    let kinds = reqwest_error_kinds(error);
    let url = error
        .url()
        .map(sanitized_url)
        .map(|url| format!(", url: {url}"))
        .unwrap_or_default();

    // serde_json's error text can quote values from a malformed response. That
    // response may contain transcription content, so retain the useful decode
    // classification but never put its nested source in logs or UI errors.
    let causes = if error.is_decode() {
        Vec::new()
    } else {
        error_source_chain(error)
    };
    let cause_details = if !causes.is_empty() {
        format!(": caused by: {}", causes.join(" -> "))
    } else if error.url().is_none() {
        // Reqwest's short Display text is safe when it cannot append a raw URL.
        format!(": {error}")
    } else {
        // The sanitized URL is already included above. Avoid formatting the
        // original error because its Display implementation includes the raw URL.
        String::new()
    };

    let details = format!("{context} (kind: {kinds}{url}){cause_details}");
    error!("{details}");
    details
}

/// Send a chat completion request to an OpenAI-compatible API
/// Returns Ok(Some(content)) on success, Ok(None) if response has no content,
/// or Err on actual errors (HTTP, parsing, etc.)
pub async fn send_chat_completion(
    provider: &PostProcessProvider,
    api_key: String,
    model: &str,
    prompt: String,
    disable_reasoning: bool,
) -> Result<Option<String>, String> {
    if provider.api_format == PostProcessApiFormat::Anthropic {
        return send_anthropic_message(provider, api_key, model, prompt).await;
    }

    send_chat_completion_with_schema(
        provider,
        api_key,
        model,
        prompt,
        None,
        None,
        disable_reasoning,
    )
    .await
}

/// Send a chat completion request with structured output support.
/// When json_schema is provided, uses structured outputs mode.
/// system_prompt is used as the system message when provided.
///
/// When disable_reasoning is set, the request carries the reasoning-disable
/// fields the endpoint is expected to understand. Not every OpenAI-compatible
/// endpoint accepts them (DeepSeek, Gemini's compat layer, and some OpenRouter
/// upstreams reject with 400), so a 400/422 answer to such a request triggers
/// one retry without the fields, and the rejection is remembered per
/// (base_url, model) so later requests skip the failing attempt entirely.
pub async fn send_chat_completion_with_schema(
    provider: &PostProcessProvider,
    api_key: String,
    model: &str,
    user_content: String,
    system_prompt: Option<String>,
    json_schema: Option<Value>,
    disable_reasoning: bool,
) -> Result<Option<String>, String> {
    let base_url = provider.base_url.trim_end_matches('/');
    let url = format!("{}/chat/completions", base_url);

    debug!(
        "Sending chat completion request to: {}",
        sanitized_url_for_log(&url)
    );

    let client = create_client(provider, &api_key)?;

    // Build messages vector
    let mut messages = Vec::new();

    // Add system prompt if provided
    if let Some(system) = system_prompt {
        messages.push(ChatMessage {
            role: "system".to_string(),
            content: system,
        });
    }

    // Add user message
    messages.push(ChatMessage {
        role: "user".to_string(),
        content: user_content,
    });

    // Build response_format if schema is provided
    let response_format = json_schema.map(|schema| ResponseFormat {
        format_type: "json_schema".to_string(),
        json_schema: JsonSchema {
            name: "transcription_output".to_string(),
            strict: true,
            schema,
        },
    });

    let key = endpoint_key(provider, model);
    let reasoning = if disable_reasoning && !is_known_rejected(&key) {
        reasoning_disable_params(provider)
    } else {
        ReasoningParams::default()
    };

    let mut request_body = ChatCompletionRequest {
        model: model.to_string(),
        messages,
        stream: false,
        response_format,
        reasoning,
    };

    let response = send_with_reasoning_retry(&client, &url, &mut request_body).await?;
    let status = response.status();

    if !status.is_success() {
        let error_text = response
            .text()
            .await
            .unwrap_or_else(|e| report_reqwest_error("Failed to read API error response", &e));
        return Err(format!(
            "API request failed with status {}: {}",
            status, error_text
        ));
    }

    let completion: ChatCompletionResponse = response
        .json()
        .await
        .map_err(|e| report_reqwest_error("Failed to parse API response", &e))?;

    let Some(choice) = completion.choices.into_iter().next() else {
        return Ok(None);
    };

    // A response cut off at the provider's output limit is incomplete; pasting
    // it would silently drop the end of the dictation (or, for structured
    // output, paste unparseable JSON).
    if choice.finish_reason.as_deref() == Some("length") {
        return Err("Response was truncated at the provider's output token limit".to_string());
    }

    Ok(choice.message.content)
}

/// Sends a chat-completions request, retrying once without the
/// reasoning-disable fields when the endpoint rejects them with a 400/422
/// (see `reasoning_disable_params`). Returns the final response whatever its
/// status; callers surface non-success statuses.
async fn send_with_reasoning_retry(
    client: &reqwest::Client,
    url: &str,
    request_body: &mut ChatCompletionRequest,
) -> Result<reqwest::Response, String> {
    let mut response = client
        .post(url)
        .json(&request_body)
        .send()
        .await
        .map_err(|e| report_reqwest_error("HTTP request failed", &e))?;
    let mut status = response.status();
    debug!(
        "Chat completion response received with status {} over {:?} from {}",
        status,
        response.version(),
        sanitized_url(response.url())
    );

    // A 400/422 on a request carrying reasoning-disable fields is almost always
    // the endpoint rejecting those fields — retry once without them.
    if !status.is_success()
        && matches!(status.as_u16(), 400 | 422)
        && !request_body.reasoning.is_empty()
    {
        let error_text = response.text().await.unwrap_or_else(|e| {
            report_reqwest_error("Failed to read reasoning rejection response", &e)
        });
        info!(
            "Endpoint rejected request with reasoning disabled (status {}): {}. Retrying without reasoning fields",
            status, error_text
        );

        request_body.reasoning = ReasoningParams::default();
        response = client
            .post(url)
            .json(&request_body)
            .send()
            .await
            .map_err(|e| report_reqwest_error("HTTP retry failed", &e))?;
        status = response.status();
        debug!(
            "Chat completion retry response received with status {} over {:?} from {}",
            status,
            response.version(),
            sanitized_url(response.url())
        );

        if status.is_success() {
            info!(
                "Retry without reasoning fields succeeded; '{}' (model '{}') will skip them from now on",
                sanitized_url_for_log(url),
                request_body.model
            );
            remember_rejection(endpoint_key_from_url(url, &request_body.model));
        }
    }

    Ok(response)
}

/// Reconstructs the rejection-memo key for a retry that succeeded without
/// reasoning fields. The base_url in the key omits the trailing
/// `/chat/completions` so it matches keys built by `endpoint_key`.
fn endpoint_key_from_url(url: &str, model: &str) -> String {
    let base = url.strip_suffix("/chat/completions").unwrap_or(url);
    format!("{base}|{model}")
}

// ------------------------------------------------------------- streaming

/// Incremental server-sent-events parser for `text/event-stream` responses.
///
/// Fed arbitrary UTF-8 chunks, it yields the joined `data:` payload of every
/// completely received event. Non-data lines (`event:`, `id:`, comments) and
/// incomplete trailing lines are held back until more data arrives; [`SseParser::finish`]
/// flushes whatever remains when the stream ends.
#[derive(Debug, Default)]
pub(crate) struct SseParser {
    buffer: String,
    current: Option<String>,
}

impl SseParser {
    pub(crate) fn push(&mut self, chunk: &str) -> Vec<String> {
        self.buffer.push_str(chunk);
        let mut events = Vec::new();
        while let Some(newline) = self.buffer.find('\n') {
            let mut line: String = self.buffer.drain(..=newline).collect();
            if line.ends_with('\n') {
                line.pop();
            }
            let line = line.trim_end_matches('\r');
            if line.is_empty() {
                if let Some(event) = self.current.take() {
                    events.push(event);
                }
            } else if let Some(data) = line.strip_prefix("data:") {
                // A single space after the colon is optional per the SSE spec.
                let data = data.strip_prefix(' ').unwrap_or(data);
                let event = self.current.get_or_insert_with(String::new);
                if !event.is_empty() {
                    event.push('\n');
                }
                event.push_str(data);
            }
        }
        events
    }

    /// Flushes a trailing event that arrived without its terminating blank
    /// line before the stream closed.
    pub(crate) fn finish(&mut self) -> Vec<String> {
        let leftover = std::mem::take(&mut self.buffer);
        let mut events = if leftover.is_empty() {
            Vec::new()
        } else {
            self.push(&format!("{leftover}\n"))
        };
        if let Some(event) = self.current.take() {
            events.push(event);
        }
        events
    }
}

/// Extracts the visible text delta from one streamed chat-completions event.
/// Only `choices[0].delta.content` counts: `reasoning_content` deltas (the
/// thinking some models stream separately) are deliberately ignored.
fn extract_stream_delta(payload: &str) -> Option<String> {
    let value: Value = serde_json::from_str(payload).ok()?;
    let delta = value.get("choices")?.get(0)?.get("delta")?;
    let content = delta.get("content")?.as_str()?;
    (!content.is_empty()).then(|| content.to_string())
}

/// Extracts `choices[0].finish_reason` from a streamed event, when present.
fn extract_stream_finish_reason(payload: &str) -> Option<String> {
    let value: Value = serde_json::from_str(payload).ok()?;
    value
        .get("choices")?
        .get(0)?
        .get("finish_reason")?
        .as_str()
        .map(|reason| reason.to_string())
}

/// Extracts a transport-level error object (`{"error":{...}}`) that some
/// endpoints deliver mid-stream instead of an HTTP error status.
fn extract_stream_error(payload: &str) -> Option<String> {
    let value: Value = serde_json::from_str(payload).ok()?;
    let message = value.get("error")?.get("message")?.as_str()?.to_string();
    (!message.is_empty()).then_some(message)
}

/// Longest suffix of `text` that is a proper prefix of `tag`, so a `<think>`
/// or `</think>` split across chunk boundaries is still recognized.
fn partial_tag_suffix(text: &str, tag: &str) -> usize {
    let bytes = text.as_bytes();
    let max = tag.len() - 1;
    for length in (1..=max).rev() {
        if length > bytes.len() {
            continue;
        }
        if text.as_bytes()[bytes.len() - length..] == tag.as_bytes()[..length] {
            return length;
        }
    }
    0
}

/// Streaming companion of the think stripping applied to complete responses:
/// feeds deltas in, gets the visible text out. A `<think>...</think>` block
/// split across any chunk boundaries is swallowed whole; the swallowed text
/// is kept until the block closes so a stream that ends inside an unterminated
/// block can still emit it verbatim (matching the non-streaming strip, which
/// leaves unclosed blocks untouched).
#[derive(Debug, Default)]
pub(crate) struct ThinkFilter {
    in_think: bool,
    think_text: String,
    /// Trailing partial tag held back until the next chunk resolves it.
    held: String,
}

impl ThinkFilter {
    pub(crate) fn push(&mut self, chunk: &str) -> String {
        let mut input = std::mem::take(&mut self.held);
        input.push_str(chunk);
        let mut visible = String::new();

        loop {
            if self.in_think {
                let Some(end) = input.find("</think>") else {
                    let hold = partial_tag_suffix(&input, "</think>");
                    let keep = input.len() - hold;
                    self.think_text.push_str(&input[..keep]);
                    self.held = input[keep..].to_string();
                    break;
                };
                self.think_text.push_str(&input[..end]);
                self.think_text.clear();
                input.drain(..end + "</think>".len());
                self.in_think = false;
            } else {
                let Some(start) = input.find("<think>") else {
                    let hold = partial_tag_suffix(&input, "<think>");
                    let keep = input.len() - hold;
                    visible.push_str(&input[..keep]);
                    self.held = input[keep..].to_string();
                    break;
                };
                visible.push_str(&input[..start]);
                input.drain(..start + "<think>".len());
                self.in_think = true;
            }
        }

        visible
    }

    /// Emits whatever a stream ending here leaves: a held partial tag outside
    /// a think block, or an unterminated think block kept verbatim.
    pub(crate) fn finish(&mut self) -> String {
        let held = std::mem::take(&mut self.held);
        let think_text = std::mem::take(&mut self.think_text);
        let in_think = self.in_think;
        self.in_think = false;
        if in_think {
            format!("<think>{think_text}{held}")
        } else {
            held
        }
    }
}

/// Outcome of a fully streamed rewrite. The visible text itself was already
/// delivered through `on_text` as it arrived; callers keep their own copy.
#[derive(Debug, Clone)]
pub(crate) struct StreamingOutcome {
    /// Time from request start to the first visible content token.
    pub first_token: Option<Duration>,
}

/// State accumulated while consuming one streaming response.
struct StreamAccumulator {
    filter: ThinkFilter,
    first_token: Option<Duration>,
    finish_reason: Option<String>,
    started: Instant,
}

impl StreamAccumulator {
    fn new(started: Instant) -> Self {
        Self {
            filter: ThinkFilter::default(),
            first_token: None,
            finish_reason: None,
            started,
        }
    }

    /// Processes one SSE payload. Returns `true` when the stream signaled
    /// `[DONE]`.
    fn on_event(&mut self, payload: &str, on_text: &mut impl FnMut(&str)) -> Result<bool, String> {
        if payload == "[DONE]" {
            return Ok(true);
        }
        if let Some(error) = extract_stream_error(payload) {
            return Err(format!("API stream error: {error}"));
        }
        if let Some(reason) = extract_stream_finish_reason(payload) {
            self.finish_reason = Some(reason);
        }
        if let Some(delta) = extract_stream_delta(payload) {
            if self.first_token.is_none() {
                self.first_token = Some(self.started.elapsed());
            }
            let visible = self.filter.push(&delta);
            if !visible.is_empty() {
                on_text(&visible);
            }
        }
        Ok(false)
    }

    /// Emits any text a closing stream still holds (a trailing event without
    /// its blank line, or an unterminated think block) into `on_text`.
    fn finish(&mut self, mut on_text: impl FnMut(&str)) {
        let tail = self.filter.finish();
        if !tail.is_empty() {
            on_text(&tail);
        }
    }

    fn into_outcome(self) -> Result<Option<StreamingOutcome>, String> {
        // A response cut off at the provider's output limit is incomplete;
        // surfacing it as an error lets the caller fall back to pasting the
        // remainder, exactly like the non-streaming path refuses truncated
        // output.
        if self.finish_reason.as_deref() == Some("length") {
            return Err("Response was truncated at the provider's output token limit".to_string());
        }
        Ok(Some(StreamingOutcome {
            first_token: self.first_token,
        }))
    }
}

/// Streams a chat completion from an OpenAI-compatible endpoint, invoking
/// `on_text` with each visible chunk (after think stripping) as it arrives.
///
/// Returns `Ok(Some(outcome))` when the stream completed, `Ok(None)` when
/// `is_cancelled` turned true before completion (nothing is typed after
/// that point), and `Err` on transport/HTTP/stream errors — including a
/// response truncated at the provider's output limit, which is surfaced as
/// an error after the already-delivered text, exactly like the
/// non-streaming path.
#[allow(clippy::too_many_arguments)]
pub(crate) async fn send_chat_completion_streaming(
    provider: &PostProcessProvider,
    api_key: String,
    model: &str,
    system_prompt: Option<String>,
    user_content: String,
    disable_reasoning: bool,
    mut on_text: impl FnMut(&str),
    is_cancelled: impl Fn() -> bool,
) -> Result<Option<StreamingOutcome>, String> {
    if is_cancelled() {
        return Ok(None);
    }

    let base_url = provider.base_url.trim_end_matches('/');
    let url = format!("{base_url}/chat/completions");
    debug!(
        "Sending streaming chat completion request to: {}",
        sanitized_url_for_log(&url)
    );

    let client = create_client(provider, &api_key)?;

    let mut messages = Vec::new();
    if let Some(system) = system_prompt {
        messages.push(ChatMessage {
            role: "system".to_string(),
            content: system,
        });
    }
    messages.push(ChatMessage {
        role: "user".to_string(),
        content: user_content,
    });

    let key = endpoint_key(provider, model);
    let reasoning = if disable_reasoning && !is_known_rejected(&key) {
        reasoning_disable_params(provider)
    } else {
        ReasoningParams::default()
    };

    let mut request_body = ChatCompletionRequest {
        model: model.to_string(),
        messages,
        stream: true,
        response_format: None,
        reasoning,
    };

    let started = Instant::now();
    let response = send_with_reasoning_retry(&client, &url, &mut request_body).await?;
    let status = response.status();
    if !status.is_success() {
        let error_text = response
            .text()
            .await
            .unwrap_or_else(|e| report_reqwest_error("Failed to read API error response", &e));
        return Err(format!(
            "API request failed with status {}: {}",
            status, error_text
        ));
    }

    let mut stream = response.bytes_stream();
    let mut parser = SseParser::default();
    let mut accumulator = StreamAccumulator::new(started);
    let mut done = false;

    loop {
        if is_cancelled() {
            debug!("Streaming chat completion cancelled");
            return Ok(None);
        }

        let chunk = match tokio::time::timeout(STREAM_CANCEL_POLL_INTERVAL, stream.next()).await {
            Err(_) => continue,
            Ok(None) => break,
            Ok(Some(Ok(bytes))) => bytes,
            Ok(Some(Err(e))) => return Err(report_reqwest_error("Streaming response failed", &e)),
        };

        let chunk = String::from_utf8_lossy(&chunk);
        for payload in parser.push(&chunk) {
            if accumulator.on_event(&payload, &mut on_text)? {
                done = true;
                break;
            }
        }
        if done {
            break;
        }
    }

    if !done {
        for payload in parser.finish() {
            if accumulator.on_event(&payload, &mut on_text)? {
                break;
            }
        }
    }

    accumulator.finish(|visible| on_text(visible));
    accumulator.into_outcome()
}

/// Send a single-prompt request to an Anthropic Messages API (`/messages`).
/// Used by every provider with the Anthropic format (the built-in Anthropic
/// provider and a custom provider set to it). None of them use structured
/// output, and thinking is off unless requested, so no reasoning fields are
/// needed.
async fn send_anthropic_message(
    provider: &PostProcessProvider,
    api_key: String,
    model: &str,
    prompt: String,
) -> Result<Option<String>, String> {
    let url = format!("{}/messages", provider.base_url.trim_end_matches('/'));

    debug!(
        "Sending Anthropic Messages request to: {}",
        sanitized_url_for_log(&url)
    );

    let request_body = AnthropicMessagesRequest {
        model: model.to_string(),
        max_tokens: ANTHROPIC_MAX_TOKENS,
        messages: vec![ChatMessage {
            role: "user".to_string(),
            content: prompt,
        }],
    };

    let response = create_client(provider, &api_key)?
        .post(&url)
        .json(&request_body)
        .send()
        .await
        .map_err(|e| report_reqwest_error("HTTP request failed", &e))?;
    let status = response.status();
    debug!(
        "Anthropic Messages response received with status {} over {:?} from {}",
        status,
        response.version(),
        sanitized_url(response.url())
    );

    if !status.is_success() {
        let error_text = response
            .text()
            .await
            .unwrap_or_else(|e| report_reqwest_error("Failed to read API error response", &e));
        return Err(format!(
            "API request failed with status {}: {}",
            status, error_text
        ));
    }

    let message: AnthropicMessagesResponse = response
        .json()
        .await
        .map_err(|e| report_reqwest_error("Failed to parse API response", &e))?;

    // A response cut off at the token budget is incomplete; pasting it would
    // silently drop the end of the dictation.
    if message.stop_reason.as_deref() == Some("max_tokens") {
        return Err(format!(
            "Response was truncated at the {} token limit",
            ANTHROPIC_MAX_TOKENS
        ));
    }

    Ok(message.into_text())
}

/// Fetch available models from an OpenAI-compatible API
/// Returns a list of model IDs
pub async fn fetch_models(
    provider: &PostProcessProvider,
    api_key: String,
) -> Result<Vec<String>, String> {
    let base_url = provider.base_url.trim_end_matches('/');
    let url = format!("{}/models", base_url);

    debug!("Fetching models from: {}", sanitized_url_for_log(&url));

    let client = create_client(provider, &api_key)?;

    let response = client
        .get(&url)
        .send()
        .await
        .map_err(|e| report_reqwest_error("Failed to fetch models", &e))?;

    let status = response.status();
    debug!(
        "Model list response received with status {} over {:?} from {}",
        status,
        response.version(),
        sanitized_url(response.url())
    );
    if !status.is_success() {
        let error_text = response
            .text()
            .await
            .unwrap_or_else(|e| report_reqwest_error("Failed to read model list error", &e));
        return Err(format!(
            "Model list request failed ({}): {}",
            status, error_text
        ));
    }

    let parsed: serde_json::Value = response
        .json()
        .await
        .map_err(|e| report_reqwest_error("Failed to parse model list response", &e))?;

    let mut models = Vec::new();

    // Handle OpenAI format: { data: [ { id: "..." }, ... ] }
    if let Some(data) = parsed.get("data").and_then(|d| d.as_array()) {
        for entry in data {
            if let Some(id) = entry.get("id").and_then(|i| i.as_str()) {
                models.push(id.to_string());
            } else if let Some(name) = entry.get("name").and_then(|n| n.as_str()) {
                models.push(name.to_string());
            }
        }
    }
    // Handle array format: [ "model1", "model2", ... ]
    else if let Some(array) = parsed.as_array() {
        for entry in array {
            if let Some(model) = entry.as_str() {
                models.push(model.to_string());
            }
        }
    }

    Ok(models)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fmt;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    #[derive(Debug)]
    struct TestError {
        message: &'static str,
        source: Option<Box<TestError>>,
    }

    impl fmt::Display for TestError {
        fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
            formatter.write_str(self.message)
        }
    }

    impl StdError for TestError {
        fn source(&self) -> Option<&(dyn StdError + 'static)> {
            self.source
                .as_deref()
                .map(|source| source as &(dyn StdError + 'static))
        }
    }

    fn provider(id: &str, base_url: &str) -> PostProcessProvider {
        PostProcessProvider {
            id: id.to_string(),
            label: id.to_string(),
            base_url: base_url.to_string(),
            allow_base_url_edit: true,
            models_endpoint: None,
            supports_structured_output: false,
            api_format: PostProcessApiFormat::OpenAi,
        }
    }

    fn anthropic_format_provider(base_url: &str) -> PostProcessProvider {
        PostProcessProvider {
            api_format: PostProcessApiFormat::Anthropic,
            ..provider("custom", base_url)
        }
    }

    fn request_json(reasoning: ReasoningParams) -> Value {
        let request = ChatCompletionRequest {
            model: "test-model".to_string(),
            messages: vec![ChatMessage {
                role: "user".to_string(),
                content: "hi".to_string(),
            }],
            stream: false,
            response_format: None,
            reasoning,
        };
        serde_json::to_value(&request).unwrap()
    }

    async fn serve_one_response(status: &str, body: &str) -> String {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let response = format!(
            "HTTP/1.1 {status}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
            body.len()
        );

        tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.unwrap();
            let mut request = [0_u8; 2048];
            let _ = stream.read(&mut request).await.unwrap();
            stream.write_all(response.as_bytes()).await.unwrap();
        });

        format!("http://{address}")
    }

    #[test]
    fn error_source_chain_includes_all_nested_causes() {
        let error = TestError {
            message: "request failed",
            source: Some(Box::new(TestError {
                message: "TLS handshake failed",
                source: Some(Box::new(TestError {
                    message: "unknown certificate authority",
                    source: None,
                })),
            })),
        };

        assert_eq!(
            error_source_chain(&error),
            vec!["TLS handshake failed", "unknown certificate authority"]
        );
    }

    #[test]
    fn log_url_sanitization_removes_credentials_and_tokens() {
        let url = "https://user:password@example.com/v1/models?api_key=secret#private";
        assert_eq!(sanitized_url_for_log(url), "https://example.com/v1/models");
    }

    #[test]
    fn invalid_log_urls_are_not_echoed() {
        assert_eq!(
            sanitized_url_for_log("not a URL containing secret"),
            "<invalid URL>"
        );
    }

    #[tokio::test]
    async fn decode_error_does_not_echo_response_values() {
        let base_url =
            serve_one_response("200 OK", r#"{"choices":"PRIVATE TRANSCRIPTION CONTENT"}"#).await;
        let error = reqwest::get(base_url)
            .await
            .unwrap()
            .json::<ChatCompletionResponse>()
            .await
            .unwrap_err();

        let details = report_reqwest_error("Failed to parse API response", &error);
        assert!(details.contains("kind: decode"));
        assert!(!details.contains("PRIVATE TRANSCRIPTION CONTENT"));
    }

    #[tokio::test]
    async fn raw_error_url_is_not_reintroduced_without_a_source() {
        let base_url = serve_one_response("400 Bad Request", "bad request").await;
        let error = reqwest::get(format!(
            "{base_url}/private?api_key=SECRET_QUERY_TOKEN#private"
        ))
        .await
        .unwrap()
        .error_for_status()
        .unwrap_err();

        let details = report_reqwest_error("Request failed", &error);
        assert!(details.contains(&format!("url: {base_url}/private")));
        assert!(!details.contains("SECRET_QUERY_TOKEN"));
        assert!(!details.contains("#private"));
    }

    #[test]
    fn requests_explicitly_disable_streaming() {
        let json = request_json(ReasoningParams::default());
        assert_eq!(json["stream"], false);
    }

    #[test]
    fn default_reasoning_params_serialize_to_no_fields() {
        let json = request_json(ReasoningParams::default());
        assert!(json.get("reasoning_effort").is_none());
        assert!(json.get("reasoning").is_none());
        assert!(json.get("thinking").is_none());
    }

    #[test]
    fn custom_provider_uses_top_level_reasoning_effort() {
        let params = reasoning_disable_params(&provider("custom", "http://localhost:11434/v1"));
        let json = request_json(params);
        assert_eq!(json["reasoning_effort"], "none");
        assert!(json.get("reasoning").is_none());
        assert!(json.get("thinking").is_none());
    }

    #[test]
    fn openrouter_uses_nested_reasoning_object() {
        let params =
            reasoning_disable_params(&provider("openrouter", "https://openrouter.ai/api/v1"));
        let json = request_json(params);
        assert!(json.get("reasoning_effort").is_none());
        assert_eq!(json["reasoning"]["effort"], "none");
        assert_eq!(json["reasoning"]["exclude"], true);
        assert!(json.get("thinking").is_none());
    }

    #[test]
    fn deepseek_base_url_uses_thinking_disabled() {
        let params = reasoning_disable_params(&provider("custom", "https://api.deepseek.com"));
        let json = request_json(params);
        assert!(json.get("reasoning_effort").is_none());
        assert!(json.get("reasoning").is_none());
        assert_eq!(json["thinking"]["type"], "disabled");
    }

    #[test]
    fn reasoning_params_is_empty_tracks_all_fields() {
        assert!(ReasoningParams::default().is_empty());
        assert!(!ReasoningParams {
            reasoning_effort: Some("none".to_string()),
            ..Default::default()
        }
        .is_empty());
        assert!(!ReasoningParams {
            thinking: Some(serde_json::json!({ "type": "disabled" })),
            ..Default::default()
        }
        .is_empty());
    }

    #[test]
    fn rejection_memo_is_keyed_by_base_url_and_model() {
        let deepseek = provider("custom", "https://api.deepseek.com/");
        let key = endpoint_key(&deepseek, "deepseek-chat");
        assert_eq!(key, "https://api.deepseek.com|deepseek-chat");
        assert!(!is_known_rejected(&key));
        remember_rejection(key.clone());
        assert!(is_known_rejected(&key));
        // A different model on the same endpoint is tracked separately
        assert!(!is_known_rejected(&endpoint_key(&deepseek, "other-model")));
    }

    #[test]
    fn anthropic_format_uses_api_key_headers() {
        let headers = build_headers(
            &anthropic_format_provider("https://example.com/anthropic/v1"),
            "key",
        )
        .unwrap();
        assert_eq!(headers["x-api-key"], "key");
        assert_eq!(headers["anthropic-version"], "2023-06-01");
        assert!(headers.get(AUTHORIZATION).is_none());

        let headers =
            build_headers(&provider("custom", "http://localhost:11434/v1"), "key").unwrap();
        assert_eq!(headers[AUTHORIZATION], "Bearer key");
        assert!(headers.get("x-api-key").is_none());
    }

    #[test]
    fn anthropic_request_carries_required_max_tokens() {
        let request = AnthropicMessagesRequest {
            model: "claude-haiku".to_string(),
            max_tokens: ANTHROPIC_MAX_TOKENS,
            messages: vec![ChatMessage {
                role: "user".to_string(),
                content: "hi".to_string(),
            }],
        };
        let json = serde_json::to_value(&request).unwrap();
        assert_eq!(json["max_tokens"], ANTHROPIC_MAX_TOKENS);
        assert_eq!(json["messages"][0]["role"], "user");
        assert!(json.get("stream").is_none());
        assert!(json.get("reasoning_effort").is_none());
    }

    // ------------------------------------------------------- streaming

    /// Serves a raw SSE-style response body and closes the connection.
    async fn serve_sse_response(body: &str) -> String {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let response = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\n{body}"
        );
        tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.unwrap();
            let mut request = [0_u8; 4096];
            let _ = stream.read(&mut request).await.unwrap();
            stream.write_all(response.as_bytes()).await.unwrap();
        });
        format!("http://{address}")
    }

    #[test]
    fn sse_parser_emits_complete_events() {
        let mut parser = SseParser::default();
        // Two events arrive in one chunk.
        let events = parser.push("data: {\"a\":1}\n\ndata: [DONE]\n\n");
        assert_eq!(events, vec![r#"{"a":1}"#, "[DONE]"]);
    }

    #[test]
    fn sse_parser_handles_split_chunks_and_crlf() {
        let mut parser = SseParser::default();
        let mut events = Vec::new();
        events.extend(parser.push("data: {\"a\":"));
        events.extend(parser.push("1}\r"));
        events.extend(parser.push("\n\r\n"));
        events.extend(parser.push("data: [DONE]\r\n\r\n"));
        assert_eq!(events, vec![r#"{"a":1}"#, "[DONE]"]);
    }

    #[test]
    fn sse_parser_ignores_comments_and_non_data_lines() {
        let mut parser = SseParser::default();
        let events = parser.push(": keep-alive\nevent: chunk\nid: 7\ndata: payload\n\n");
        assert_eq!(events, vec!["payload"]);
    }

    #[test]
    fn sse_parser_finish_flushes_unterminated_event() {
        let mut parser = SseParser::default();
        assert!(parser.push("data: tail").is_empty());
        assert_eq!(parser.finish(), vec!["tail"]);
    }

    #[test]
    fn stream_delta_takes_only_content() {
        let payload = r#"{"choices":[{"delta":{"content":"Hello"}}]}"#;
        assert_eq!(extract_stream_delta(payload).as_deref(), Some("Hello"));

        // reasoning_content deltas must never surface as text
        let reasoning = r#"{"choices":[{"delta":{"reasoning_content":"thinking..."}}]}"#;
        assert_eq!(extract_stream_delta(reasoning), None);

        // both present: only content counts
        let mixed = r#"{"choices":[{"delta":{"reasoning_content":"hmm","content":"!"}}]}"#;
        assert_eq!(extract_stream_delta(mixed).as_deref(), Some("!"));

        // empty content, role-only deltas, and non-choices frames
        assert_eq!(
            extract_stream_delta(r#"{"choices":[{"delta":{"content":""}}]}"#),
            None
        );
        assert_eq!(
            extract_stream_delta(r#"{"choices":[{"delta":{"role":"assistant"}}]}"#),
            None
        );
        assert_eq!(extract_stream_delta("{}"), None);
    }

    #[test]
    fn stream_finish_reason_and_errors_are_extracted() {
        assert_eq!(
            extract_stream_finish_reason(r#"{"choices":[{"finish_reason":"stop"}]}"#).as_deref(),
            Some("stop")
        );
        assert_eq!(extract_stream_finish_reason(r#"{"choices":[{}]}"#), None);

        assert_eq!(
            extract_stream_error(r#"{"error":{"message":"rate limited"}}"#).as_deref(),
            Some("rate limited")
        );
        assert_eq!(extract_stream_error(r#"{"choices":[]}"#), None);
    }

    #[test]
    fn think_filter_strips_blocks_across_arbitrary_chunk_boundaries() {
        // For every possible split point of this stream, the visible output
        // must reconstruct exactly the non-think text.
        let full = "<think>reasoning 123</think>Hello, world!<think>more</think> Bye.";
        for split in 0..full.len() {
            let (head, tail) = full.split_at(split);
            let mut filter = ThinkFilter::default();
            let mut visible = filter.push(head);
            visible.push_str(&filter.push(tail));
            visible.push_str(&filter.finish());
            assert_eq!(visible, "Hello, world! Bye.", "split at {split}");
        }
    }

    #[test]
    fn think_filter_preserves_text_mentioning_tags() {
        let mut filter = ThinkFilter::default();
        let visible = filter.push("a < thinker> b </thinkable> c");
        assert_eq!(visible, "a < thinker> b </thinkable> c");
    }

    #[test]
    fn think_filter_unterminated_block_is_kept_verbatim() {
        let mut filter = ThinkFilter::default();
        let visible = filter.push("<think>never closed");
        assert_eq!(visible, "");
        assert_eq!(filter.finish(), "<think>never closed");
    }

    #[test]
    fn streaming_request_sets_stream_true() {
        let request = ChatCompletionRequest {
            model: "m".to_string(),
            messages: vec![],
            stream: true,
            response_format: None,
            reasoning: ReasoningParams::default(),
        };
        assert_eq!(serde_json::to_value(&request).unwrap()["stream"], true);
    }

    #[tokio::test]
    async fn streaming_delivers_text_and_first_token() {
        let base_url = serve_sse_response(
            "data: {\"choices\":[{\"delta\":{\"reasoning_content\":\"ignore me\"}}]}\n\n\
             data: {\"choices\":[{\"delta\":{\"content\":\"Hello\"}}]}\n\n\
             data: {\"choices\":[{\"delta\":{\"content\":\" world\"}}]}\n\n\
             data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n\
             data: [DONE]\n\n",
        )
        .await;

        let mut seen = String::new();
        let result = send_chat_completion_streaming(
            &provider("custom", &base_url),
            "key".to_string(),
            "test-model",
            None,
            "raw text".to_string(),
            false,
            |delta| seen.push_str(delta),
            || false,
        )
        .await
        .unwrap()
        .expect("stream should complete");

        assert_eq!(seen, "Hello world");
        assert!(result.first_token.is_some(), "first token must be recorded");
    }

    #[tokio::test]
    async fn streaming_strips_think_blocks_from_typed_text() {
        let base_url = serve_sse_response(
            "data: {\"choices\":[{\"delta\":{\"content\":\"<think>sec\"}}]}\n\n\
             data: {\"choices\":[{\"delta\":{\"content\":\"ret\"}}]}\n\n\
             data: {\"choices\":[{\"delta\":{\"content\":\"</think>Clean\"}}]}\n\n\
             data: [DONE]\n\n",
        )
        .await;

        let mut seen = String::new();
        send_chat_completion_streaming(
            &provider("custom", &base_url),
            "key".to_string(),
            "test-model",
            None,
            "raw text".to_string(),
            false,
            |delta| seen.push_str(delta),
            || false,
        )
        .await
        .unwrap()
        .expect("stream should complete");

        assert_eq!(seen, "Clean");
    }

    #[tokio::test]
    async fn streaming_surfaces_truncated_responses_as_errors() {
        let base_url = serve_sse_response(
            "data: {\"choices\":[{\"delta\":{\"content\":\"partial\"}}]}\n\n\
             data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"length\"}]}\n\n\
             data: [DONE]\n\n",
        )
        .await;

        let result = send_chat_completion_streaming(
            &provider("custom", &base_url),
            "key".to_string(),
            "test-model",
            None,
            "raw text".to_string(),
            false,
            |delta| {
                let _ = delta;
            },
            || false,
        )
        .await;

        assert!(result.unwrap_err().contains("truncated"));
    }

    #[tokio::test]
    async fn streaming_cancellation_returns_none_without_typing() {
        let base_url =
            serve_sse_response("data: {\"choices\":[{\"delta\":{\"content\":\"too late\"}}]}\n\n")
                .await;

        let mut seen = String::new();
        let result = send_chat_completion_streaming(
            &provider("custom", &base_url),
            "key".to_string(),
            "test-model",
            None,
            "raw text".to_string(),
            false,
            |delta| seen.push_str(delta),
            || true,
        )
        .await;

        assert!(result.unwrap().is_none());
        assert!(seen.is_empty(), "nothing may be typed after cancellation");
    }

    #[tokio::test]
    async fn streaming_mid_stream_error_object_fails_the_call() {
        let base_url =
            serve_sse_response("data: {\"error\":{\"message\":\"upstream exploded\"}}\n\n").await;

        let result = send_chat_completion_streaming(
            &provider("custom", &base_url),
            "key".to_string(),
            "test-model",
            None,
            "raw text".to_string(),
            false,
            |delta| {
                let _ = delta;
            },
            || false,
        )
        .await;

        assert_eq!(result.unwrap_err(), "API stream error: upstream exploded");
    }

    #[test]
    fn anthropic_response_keeps_only_text_blocks() {
        let response: AnthropicMessagesResponse = serde_json::from_str(
            r#"{"content":[{"type":"thinking","thinking":"hmm"},{"type":"text","text":"Hello"},{"type":"text","text":" world"}]}"#,
        )
        .unwrap();
        assert_eq!(response.into_text().as_deref(), Some("Hello world"));

        let empty: AnthropicMessagesResponse = serde_json::from_str(r#"{"content":[]}"#).unwrap();
        assert_eq!(empty.into_text(), None);
    }

    #[tokio::test]
    async fn anthropic_format_parses_messages_response() {
        let base_url = serve_one_response(
            "200 OK",
            r#"{"type":"message","role":"assistant","content":[{"type":"text","text":"Cleaned text"}]}"#,
        )
        .await;
        let result = send_chat_completion(
            &anthropic_format_provider(&base_url),
            "key".to_string(),
            "claude-haiku",
            "raw text".to_string(),
            true,
        )
        .await;
        assert_eq!(result, Ok(Some("Cleaned text".to_string())));
    }

    #[tokio::test]
    async fn chat_completion_rejects_truncated_response() {
        let base_url = serve_one_response(
            "200 OK",
            r#"{"choices":[{"message":{"content":"Cut off"},"finish_reason":"length"}]}"#,
        )
        .await;
        let result = send_chat_completion(
            &provider("custom", &base_url),
            "key".to_string(),
            "test-model",
            "raw text".to_string(),
            false,
        )
        .await;
        assert!(result.unwrap_err().contains("truncated"));
    }

    #[tokio::test]
    async fn anthropic_format_rejects_truncated_response() {
        let base_url = serve_one_response(
            "200 OK",
            r#"{"type":"message","stop_reason":"max_tokens","content":[{"type":"text","text":"Cut off"}]}"#,
        )
        .await;
        let result = send_chat_completion(
            &anthropic_format_provider(&base_url),
            "key".to_string(),
            "claude-haiku",
            "raw text".to_string(),
            true,
        )
        .await;
        assert!(result.unwrap_err().contains("truncated"));
    }
}
