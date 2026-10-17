//! Dictation statistics: pure aggregation over per-dictation metrics.
//!
//! The pipeline records one [`DictationMetric`] per completed dictation (see
//! `actions::PipelineTimings`); this module turns rows into the summaries the
//! Stats page renders — counts, characters, per-stage average/p50/p90, and
//! per-day buckets — plus the CSV export. Everything here is pure so it can
//! be unit-tested without a database; persistence lives in
//! `managers::history` and the platform glue (frontmost-app name) stays at
//! the call sites.

use chrono::{DateTime, Local, NaiveDate};
use serde::{Deserialize, Serialize};
use specta::Type;

/// One completed dictation, as persisted in `dictation_metrics`.
#[derive(Clone, Debug, Serialize, Deserialize, Type)]
pub struct DictationMetric {
    /// Unix timestamp (seconds) of the dictation.
    pub timestamp: i64,
    /// Frontmost application at output time, when it could be read.
    pub app_name: Option<String>,
    /// Characters of the final output text.
    pub char_count: u64,
    /// Model run duration (including any model-load wait).
    pub transcribe_ms: u64,
    /// LLM post-processing call duration (near zero when disabled).
    pub post_process_ms: u64,
    /// Streamed rewrite: time to the first visible token.
    pub first_token_ms: Option<u64>,
    /// Output stage: paste (keystroke path) or typing window (streamed).
    pub output_ms: u64,
    /// Whole stop-to-output pipeline.
    pub total_ms: u64,
    /// Whether the output was typed as it streamed (vs. pasted at the end).
    pub streamed: bool,
    /// Whether post-processing was requested for this dictation.
    pub post_processed: bool,
}

/// Average and percentile latencies of one pipeline stage.
#[derive(Clone, Copy, Debug, Default, Serialize, Type)]
pub struct StageStats {
    /// Number of dictations contributing to this stage.
    pub count: u64,
    pub avg_ms: f64,
    pub p50_ms: u64,
    pub p90_ms: u64,
}

/// One day of activity, bucketed in the user's local timezone.
#[derive(Clone, Debug, Serialize, Type)]
pub struct DailyStat {
    /// Local date, `YYYY-MM-DD`.
    pub date: String,
    pub dictations: u64,
    pub characters: u64,
    pub avg_total_ms: f64,
}

/// Everything the Stats page shows for one range.
#[derive(Clone, Debug, Serialize, Type)]
pub struct StatsSummary {
    pub range_days: u32,
    pub dictations: u64,
    pub characters: u64,
    pub streamed: u64,
    pub post_processed: u64,
    pub transcribe: StageStats,
    pub post_process: StageStats,
    pub first_token: StageStats,
    pub output: StageStats,
    pub total: StageStats,
    /// One bucket per day of the range, oldest first (empty days included).
    pub daily: Vec<DailyStat>,
}

/// A learned-words event row, for the "recently learned" list.
#[derive(Clone, Debug, Serialize, Type)]
pub struct LearnedWordEvent {
    pub word: String,
    /// Unix timestamp (seconds) when the word was learned.
    pub timestamp: i64,
}

/// Linear-interpolation percentile (the common "type 7" estimate, matching
/// numpy's default). `p` is 0–100. Returns 0 for an empty slice; `values`
/// does not need to be sorted.
pub fn percentile(mut values: Vec<u64>, p: f64) -> u64 {
    if values.is_empty() {
        return 0;
    }
    values.sort_unstable();
    let p = p.clamp(0.0, 100.0);
    if values.len() == 1 {
        return values[0];
    }
    let rank = (p / 100.0) * (values.len() - 1) as f64;
    let low = rank.floor() as usize;
    let high = (low + 1).min(values.len() - 1);
    let frac = rank - low as f64;
    let interpolated = values[low] as f64 + (values[high] as f64 - values[low] as f64) * frac;
    interpolated.round() as u64
}

fn stage_stats(values: Vec<u64>) -> StageStats {
    if values.is_empty() {
        return StageStats::default();
    }
    let count = values.len() as u64;
    let sum: u64 = values.iter().sum();
    let avg = sum as f64 / count as f64;
    let p50 = percentile(values.clone(), 50.0);
    let p90 = percentile(values, 90.0);
    StageStats {
        count,
        avg_ms: avg,
        p50_ms: p50,
        p90_ms: p90,
    }
}

/// Local calendar date (`YYYY-MM-DD`) of a unix timestamp.
pub fn local_date_string(timestamp: i64) -> String {
    DateTime::from_timestamp(timestamp, 0)
        .map(|utc| utc.with_timezone(&Local).date_naive().to_string())
        .unwrap_or_default()
}

/// Unix timestamp of the start of `date` in the local timezone; ambiguous
/// DST midnights resolve to the earlier interpretation.
pub fn local_midnight_unix(date: NaiveDate) -> i64 {
    use chrono::TimeZone;
    date.and_hms_opt(0, 0, 0)
        .and_then(|naive| Local.from_local_datetime(&naive).earliest())
        .map(|dt| dt.timestamp())
        .unwrap_or_else(|| {
            date.and_hms_opt(0, 0, 0)
                .map_or(0, |n| n.and_utc().timestamp())
        })
}

fn pick_values<'a>(
    in_range: &[&'a DictationMetric],
    f: impl Fn(&'a DictationMetric) -> u64,
) -> Vec<u64> {
    in_range.iter().map(|m| f(m)).collect()
}

/// Aggregates `metrics` (any timestamps; they are filtered by local date)
/// into a summary covering the `range_days` calendar days ending on `today`.
pub fn summarize(metrics: &[DictationMetric], range_days: u32, today: NaiveDate) -> StatsSummary {
    let range_days = range_days.max(1);
    let first_day = today - chrono::Days::new(range_days as u64 - 1);

    let in_range: Vec<&DictationMetric> = metrics
        .iter()
        .filter(|m| {
            let Some(utc) = DateTime::from_timestamp(m.timestamp, 0) else {
                return false;
            };
            let date = utc.with_timezone(&Local).date_naive();
            date >= first_day && date <= today
        })
        .collect();

    let mut daily: Vec<DailyStat> = Vec::with_capacity(range_days as usize);
    for i in 0..range_days {
        let day = first_day + chrono::Days::new(i as u64);
        let day_key = day.to_string();
        let buckets: Vec<&&DictationMetric> = in_range
            .iter()
            .filter(|m| local_date_string(m.timestamp) == day_key)
            .collect();
        let totals: Vec<u64> = buckets.iter().map(|m| m.total_ms).collect();
        daily.push(DailyStat {
            date: day_key,
            dictations: buckets.len() as u64,
            characters: buckets.iter().map(|m| m.char_count).sum(),
            avg_total_ms: if totals.is_empty() {
                0.0
            } else {
                totals.iter().sum::<u64>() as f64 / totals.len() as f64
            },
        });
    }

    StatsSummary {
        range_days,
        dictations: in_range.len() as u64,
        characters: in_range.iter().map(|m| m.char_count).sum(),
        streamed: in_range.iter().filter(|m| m.streamed).count() as u64,
        post_processed: in_range.iter().filter(|m| m.post_processed).count() as u64,
        transcribe: stage_stats(pick_values(&in_range, |m| m.transcribe_ms)),
        post_process: stage_stats(pick_values(&in_range, |m| m.post_process_ms)),
        // First-token latency only exists for streamed rewrites; skipped
        // entries must not drag the average down as zeros.
        first_token: stage_stats(in_range.iter().filter_map(|m| m.first_token_ms).collect()),
        output: stage_stats(pick_values(&in_range, |m| m.output_ms)),
        total: stage_stats(pick_values(&in_range, |m| m.total_ms)),
        daily,
    }
}

/// Renders metrics as CSV, oldest first: one header row plus one row per
/// dictation. Timestamps are rendered in the user's timezone; fields are
/// escaped with the usual RFC 4180 rules.
pub fn metrics_to_csv(metrics: &mut [DictationMetric]) -> String {
    metrics.sort_by_key(|m| m.timestamp);
    let mut out = String::from("timestamp,app,characters,transcribe_ms,post_process_ms,first_token_ms,output_ms,total_ms,streamed,post_processed\n");
    for m in metrics.iter() {
        let ts = DateTime::from_timestamp(m.timestamp, 0)
            .map(|utc| utc.with_timezone(&Local).to_rfc3339())
            .unwrap_or_default();
        let app = csv_field(m.app_name.as_deref().unwrap_or(""));
        out.push_str(&format!(
            "{},{},{},{},{},{},{},{},{},{}\n",
            csv_field(&ts),
            app,
            m.char_count,
            m.transcribe_ms,
            m.post_process_ms,
            m.first_token_ms.map(|v| v.to_string()).unwrap_or_default(),
            m.output_ms,
            m.total_ms,
            m.streamed,
            m.post_processed,
        ));
    }
    out
}

/// Escapes one CSV field (RFC 4180: quote when needed, double the quotes).
pub fn csv_field(value: &str) -> String {
    if value.contains(',') || value.contains('"') || value.contains('\n') || value.contains('\r') {
        format!("\"{}\"", value.replace('"', "\"\""))
    } else {
        value.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::NaiveDate;

    fn metric(ts: i64, chars: u64, total: u64) -> DictationMetric {
        DictationMetric {
            timestamp: ts,
            app_name: Some("TestApp".to_string()),
            char_count: chars,
            transcribe_ms: total / 2,
            post_process_ms: 10,
            first_token_ms: None,
            output_ms: 50,
            total_ms: total,
            streamed: false,
            post_processed: false,
        }
    }

    fn date(s: &str) -> NaiveDate {
        NaiveDate::parse_from_str(s, "%Y-%m-%d").expect("valid date")
    }

    #[test]
    fn percentile_interpolates_linearly() {
        // rank 0.5·(5-1)=2 → exact element
        assert_eq!(percentile(vec![10, 20, 30, 40, 50], 50.0), 30);
        // rank 0.9·4=3.6 → 40 + 0.6·(50-40) = 46
        assert_eq!(percentile(vec![10, 20, 30, 40, 50], 90.0), 46);
        // unsorted input is sorted first
        assert_eq!(percentile(vec![50, 10, 40, 20, 30], 50.0), 30);
    }

    #[test]
    fn percentile_handles_edges() {
        assert_eq!(percentile(vec![], 90.0), 0);
        assert_eq!(percentile(vec![7], 10.0), 7);
        assert_eq!(percentile(vec![7], 99.0), 7);
        assert_eq!(percentile(vec![1, 2], 0.0), 1);
        assert_eq!(percentile(vec![1, 2], 100.0), 2);
    }

    #[test]
    fn summarize_counts_only_days_inside_the_range() {
        let today = date("2026-07-10");
        let metrics = vec![
            metric(local_midnight_unix(date("2026-07-10")) + 3600, 10, 100),
            metric(local_midnight_unix(date("2026-07-09")) + 3600, 20, 200),
            metric(local_midnight_unix(date("2026-07-01")) + 3600, 99, 999),
        ];

        let week = summarize(&metrics, 7, today);
        assert_eq!(week.dictations, 2);
        assert_eq!(week.characters, 30);
        assert_eq!(week.daily.len(), 7);
        // oldest bucket first, empty days included
        assert_eq!(
            week.daily.first().map(|d| d.date.as_str()),
            Some("2026-07-04")
        );
        let last = week.daily.last().expect("non-empty");
        assert_eq!(last.date, "2026-07-10");
        assert_eq!(last.dictations, 1);
        assert_eq!(last.characters, 10);
        assert_eq!(last.avg_total_ms, 100.0);
    }

    #[test]
    fn summarize_today_range_has_one_bucket() {
        let today = date("2026-07-10");
        let metrics = vec![metric(local_midnight_unix(today) + 60, 5, 40)];

        let s = summarize(&metrics, 1, today);
        assert_eq!(s.range_days, 1);
        assert_eq!(s.daily.len(), 1);
        assert_eq!(s.dictations, 1);
    }

    #[test]
    fn summarize_aggregates_stage_stats() {
        let today = date("2026-07-10");
        let mut m1 = metric(local_midnight_unix(today) + 60, 1, 100);
        m1.transcribe_ms = 100;
        m1.streamed = true;
        m1.first_token_ms = Some(30);
        let mut m2 = metric(local_midnight_unix(today) + 120, 2, 200);
        m2.transcribe_ms = 300;
        m2.post_processed = true;

        let s = summarize(&[m1, m2], 1, today);
        assert_eq!(s.transcribe.count, 2);
        assert_eq!(s.transcribe.avg_ms, 200.0);
        // p50 of [100, 300] = 200; p90 = 280
        assert_eq!(s.transcribe.p50_ms, 200);
        assert_eq!(s.transcribe.p90_ms, 280);
        // first token only counts streamed entries
        assert_eq!(s.first_token.count, 1);
        assert_eq!(s.first_token.p50_ms, 30);
        assert_eq!(s.streamed, 1);
        assert_eq!(s.post_processed, 1);
        assert_eq!(s.total.count, 2);
    }

    #[test]
    fn summarize_with_no_metrics_returns_zeroed_summary() {
        let s = summarize(&[], 30, date("2026-07-10"));
        assert_eq!(s.dictations, 0);
        assert_eq!(s.characters, 0);
        assert_eq!(s.daily.len(), 30);
        assert!(s.daily.iter().all(|d| d.dictations == 0));
        assert_eq!(s.transcribe.avg_ms, 0.0);
        assert_eq!(s.transcribe.p90_ms, 0);
    }

    #[test]
    fn csv_escapes_and_orders_rows() {
        let today = date("2026-07-10");
        let mut m1 = metric(local_midnight_unix(today) + 60, 3, 120);
        m1.app_name = Some("Editor, \"Pro\"".to_string());
        m1.first_token_ms = Some(45);
        m1.streamed = true;
        let m2 = metric(local_midnight_unix(today) + 120, 4, 80);

        let csv = metrics_to_csv(&mut [m1, m2]);
        let lines: Vec<&str> = csv.lines().collect();
        assert_eq!(
            lines[0],
            "timestamp,app,characters,transcribe_ms,post_process_ms,first_token_ms,output_ms,total_ms,streamed,post_processed"
        );
        assert_eq!(lines.len(), 3);
        assert!(lines[1].contains("\"Editor, \"\"Pro\"\"\""));
        assert!(lines[1].ends_with(",45,50,120,true,false"));
        assert!(lines[2].ends_with(",50,80,false,false"));
    }

    #[test]
    fn csv_field_quotes_only_when_needed() {
        assert_eq!(csv_field("plain"), "plain");
        assert_eq!(csv_field("a,b"), "\"a,b\"");
        assert_eq!(csv_field("say \"hi\""), "\"say \"\"hi\"\"\"");
        assert_eq!(csv_field("line\nbreak"), "\"line\nbreak\"");
    }

    #[test]
    fn local_date_string_uses_local_calendar_day() {
        // 2026-07-10T00:30Z is still July 9 in the Americas.
        let ts = DateTime::parse_from_rfc3339("2026-07-10T00:30:00Z")
            .expect("valid ts")
            .timestamp();
        let day = local_date_string(ts);
        assert!(day == "2026-07-10" || day == "2026-07-09");
    }
}
