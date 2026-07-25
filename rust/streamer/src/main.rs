//! Minimal asynchronous SSE chat client and concurrent load generator.

use std::collections::HashMap;
use std::env;
use std::fmt;
use std::io::{self, Write};
use std::sync::Arc;
use std::time::{Duration, Instant};

use futures_util::StreamExt;
use reqwest::{Client, Response};
use serde::{Deserialize, Serialize};
use serde_json::json;
use tokio::sync::Semaphore;
use tokio::task::JoinSet;

const MODEL: &str = "Qwen/Qwen3-4B-Instruct-2507";

type BoxError = Box<dyn std::error::Error + Send + Sync>;

#[derive(Debug)]
struct CliError(String);

impl fmt::Display for CliError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl std::error::Error for CliError {}

#[derive(Debug)]
struct Config {
    command: Command,
    base_url: String,
    prompt: String,
    max_tokens: u64,
}

#[derive(Debug)]
enum Command {
    Chat,
    Load {
        requests: usize,
        concurrency: usize,
        cancel_every: usize,
        cancel_after_events: usize,
        cancel_max_tokens: Option<u64>,
        metrics_timeout: Duration,
    },
}

impl Config {
    fn parse() -> Result<Self, CliError> {
        let mut args = env::args().skip(1);
        let command_name = args.next().ok_or_else(usage)?;
        if matches!(command_name.as_str(), "-h" | "--help" | "help") {
            return Err(usage());
        }
        let mut base_url = "http://127.0.0.1:8000".to_owned();
        let mut prompt = "Count from 1 to 8, separated by spaces.".to_owned();
        let mut max_tokens = 32_u64;
        let mut requests = 8_usize;
        let mut concurrency = 4_usize;
        let mut cancel_every = 0_usize;
        let mut cancel_after_events = 1_usize;
        let mut cancel_max_tokens = None;
        let mut metrics_timeout = Duration::from_secs(15);

        while let Some(flag) = args.next() {
            let value = args
                .next()
                .ok_or_else(|| CliError(format!("missing value for {flag}")))?;
            match flag.as_str() {
                "--base-url" => base_url = value,
                "--prompt" => prompt = value,
                "--max-tokens" => max_tokens = parse_positive(&flag, &value)?,
                "--requests" => requests = parse_positive(&flag, &value)?,
                "--concurrency" => concurrency = parse_positive(&flag, &value)?,
                "--cancel-every" => cancel_every = parse_nonnegative(&flag, &value)?,
                "--cancel-after-events" => cancel_after_events = parse_nonnegative(&flag, &value)?,
                "--cancel-max-tokens" => cancel_max_tokens = Some(parse_positive(&flag, &value)?),
                "--metrics-timeout-seconds" => {
                    metrics_timeout = Duration::from_secs(parse_positive(&flag, &value)?)
                }
                _ => return Err(CliError(format!("unknown option: {flag}"))),
            }
        }
        if prompt.is_empty() {
            return Err(CliError("--prompt must not be empty".to_owned()));
        }
        let command = match command_name.as_str() {
            "chat" => Command::Chat,
            "load" => {
                if concurrency > requests {
                    return Err(CliError(
                        "--concurrency must not exceed --requests".to_owned(),
                    ));
                }
                Command::Load {
                    requests,
                    concurrency,
                    cancel_every,
                    cancel_after_events,
                    cancel_max_tokens,
                    metrics_timeout,
                }
            }
            _ => return Err(usage()),
        };
        Ok(Self {
            command,
            base_url: base_url.trim_end_matches('/').to_owned(),
            prompt,
            max_tokens,
        })
    }
}

fn parse_positive<T>(flag: &str, value: &str) -> Result<T, CliError>
where
    T: std::str::FromStr + PartialOrd + From<u8>,
{
    let parsed = value
        .parse::<T>()
        .map_err(|_| CliError(format!("{flag} requires a positive integer")))?;
    if parsed < T::from(1) {
        return Err(CliError(format!("{flag} requires a positive integer")));
    }
    Ok(parsed)
}

fn parse_nonnegative<T>(flag: &str, value: &str) -> Result<T, CliError>
where
    T: std::str::FromStr,
{
    value
        .parse::<T>()
        .map_err(|_| CliError(format!("{flag} requires a non-negative integer")))
}

fn usage() -> CliError {
    CliError(
        "usage:\n  forge-streamer chat [--base-url URL] [--prompt TEXT] \
         [--max-tokens N]\n  forge-streamer load [--base-url URL] \
         [--prompt TEXT] [--max-tokens N] [--requests N] \
         [--concurrency N] [--cancel-every N] \
         [--cancel-after-events N] [--cancel-max-tokens N] \
         [--metrics-timeout-seconds N]"
            .to_owned(),
    )
}

#[derive(Debug, Deserialize)]
struct Chunk {
    #[serde(default)]
    choices: Vec<Choice>,
    #[serde(default)]
    error: Option<ApiError>,
}

#[derive(Debug, Deserialize)]
struct Choice {
    delta: Delta,
    finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
struct Delta {
    #[serde(default)]
    content: String,
}

#[derive(Debug, Deserialize)]
struct ApiError {
    message: String,
}

#[derive(Debug)]
enum Event {
    Chunk(Chunk),
    Done,
}

#[derive(Debug, Default)]
struct SseDecoder {
    buffer: Vec<u8>,
}

impl SseDecoder {
    fn push(&mut self, bytes: &[u8]) -> Result<Vec<Event>, BoxError> {
        self.buffer.extend_from_slice(bytes);
        let mut events = Vec::new();
        while let Some((boundary, delimiter_len)) = find_boundary(&self.buffer) {
            let frame = self.buffer.drain(..boundary).collect::<Vec<_>>();
            self.buffer.drain(..delimiter_len);
            if let Some(event) = decode_frame(&frame)? {
                events.push(event);
            }
        }
        Ok(events)
    }

    fn finish(&self) -> Result<(), CliError> {
        if self.buffer.iter().all(u8::is_ascii_whitespace) {
            Ok(())
        } else {
            Err(CliError("SSE stream ended inside an event".to_owned()))
        }
    }
}

fn find_boundary(buffer: &[u8]) -> Option<(usize, usize)> {
    let lf = buffer
        .windows(2)
        .position(|window| window == b"\n\n")
        .map(|index| (index, 2));
    let crlf = buffer
        .windows(4)
        .position(|window| window == b"\r\n\r\n")
        .map(|index| (index, 4));
    match (lf, crlf) {
        (Some(left), Some(right)) => Some(if left.0 <= right.0 { left } else { right }),
        (Some(boundary), None) | (None, Some(boundary)) => Some(boundary),
        (None, None) => None,
    }
}

fn decode_frame(frame: &[u8]) -> Result<Option<Event>, BoxError> {
    let text = std::str::from_utf8(frame)?;
    let data = text
        .lines()
        .filter_map(|line| line.strip_prefix("data:"))
        .map(str::trim_start)
        .collect::<Vec<_>>()
        .join("\n");
    if data.is_empty() {
        return Ok(None);
    }
    if data == "[DONE]" {
        return Ok(Some(Event::Done));
    }
    Ok(Some(Event::Chunk(serde_json::from_str(&data)?)))
}

#[derive(Debug, Serialize)]
struct RequestResult {
    index: usize,
    status: &'static str,
    text: String,
    content_events: u64,
    ttft_seconds: Option<f64>,
    mean_itl_seconds: Option<f64>,
    duration_seconds: f64,
    finish_reason: Option<String>,
}

impl RequestResult {
    fn failed(index: usize, started: Instant, error: &dyn fmt::Display) -> Self {
        Self {
            index,
            status: "failed",
            text: error.to_string(),
            content_events: 0,
            ttft_seconds: None,
            mean_itl_seconds: None,
            duration_seconds: started.elapsed().as_secs_f64(),
            finish_reason: None,
        }
    }
}

async fn post_stream(
    client: &Client,
    base_url: &str,
    prompt: &str,
    max_tokens: u64,
) -> Result<Response, BoxError> {
    let response = client
        .post(format!("{base_url}/v1/chat/completions"))
        .json(&json!({
            "model": MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "stream": true,
            "max_tokens": max_tokens,
            "temperature": 0.0
        }))
        .send()
        .await?;
    if !response.status().is_success() {
        let status = response.status();
        let body = response.text().await?;
        return Err(CliError(format!("HTTP {status}: {body}")).into());
    }
    Ok(response)
}

async fn cancel_request(client: &Client, base_url: &str, request_id: &str) -> Result<(), BoxError> {
    let response = client
        .post(format!("{base_url}/v1/requests/{request_id}/cancel"))
        .send()
        .await?;
    if !response.status().is_success() {
        let status = response.status();
        let body = response.text().await?;
        return Err(CliError(format!(
            "cancellation for {request_id} returned HTTP {status}: {body}"
        ))
        .into());
    }
    Ok(())
}

async fn consume(
    client: &Client,
    base_url: &str,
    prompt: &str,
    max_tokens: u64,
    index: usize,
    cancel_after_events: Option<usize>,
    print_content: bool,
) -> Result<RequestResult, BoxError> {
    let started = Instant::now();
    let response = post_stream(client, base_url, prompt, max_tokens).await?;
    let request_id = response
        .headers()
        .get("x-forge-request-id")
        .ok_or_else(|| CliError("response omitted x-forge-request-id".to_owned()))?
        .to_str()?
        .to_owned();
    let mut cancellation_sent = false;
    if cancel_after_events == Some(0) {
        cancel_request(client, base_url, &request_id).await?;
        cancellation_sent = true;
    }
    let mut stream = response.bytes_stream();
    let mut decoder = SseDecoder::default();
    let mut text = String::new();
    let mut content_events = 0_u64;
    let mut first_content_at = None;
    let mut previous_content_at = None;
    let mut itl_sum = Duration::ZERO;
    let mut itl_count = 0_u64;
    let mut finish_reason = None;
    let mut done = false;

    while let Some(chunk) = stream.next().await {
        for event in decoder.push(&chunk?)? {
            match event {
                Event::Done => done = true,
                Event::Chunk(chunk) => {
                    if let Some(error) = chunk.error {
                        return Err(
                            CliError(format!("server stream error: {}", error.message)).into()
                        );
                    }
                    let choice = chunk
                        .choices
                        .first()
                        .ok_or_else(|| CliError("SSE chunk has no choice".to_owned()))?;
                    if let Some(reason) = &choice.finish_reason {
                        finish_reason = Some(reason.clone());
                    }
                    if !choice.delta.content.is_empty() {
                        let now = Instant::now();
                        first_content_at.get_or_insert(now);
                        if let Some(previous) = previous_content_at {
                            itl_sum += now.duration_since(previous);
                            itl_count += 1;
                        }
                        previous_content_at = Some(now);
                        content_events += 1;
                        text.push_str(&choice.delta.content);
                        if print_content {
                            print!("{}", choice.delta.content);
                            io::stdout().flush()?;
                        }
                        if !cancellation_sent
                            && cancel_after_events.is_some_and(|limit| {
                                limit > 0
                                    && content_events >= u64::try_from(limit).unwrap_or(u64::MAX)
                            })
                        {
                            cancel_request(client, base_url, &request_id).await?;
                            cancellation_sent = true;
                        }
                    }
                }
            }
        }
        if done {
            break;
        }
    }
    decoder.finish()?;
    if !done {
        return Err(CliError("stream ended without data: [DONE]".to_owned()).into());
    }
    Ok(RequestResult {
        index,
        status: if cancellation_sent {
            if finish_reason.as_deref() != Some("cancelled") {
                return Err(CliError(format!(
                    "server acknowledged cancellation but finished with {finish_reason:?}"
                ))
                .into());
            }
            "cancelled"
        } else {
            "finished"
        },
        text,
        content_events,
        ttft_seconds: first_content_at.map(|instant| instant.duration_since(started).as_secs_f64()),
        mean_itl_seconds: mean_duration(itl_sum, itl_count),
        duration_seconds: started.elapsed().as_secs_f64(),
        finish_reason,
    })
}

fn mean_duration(sum: Duration, count: u64) -> Option<f64> {
    (count > 0).then(|| sum.as_secs_f64() / count as f64)
}

#[derive(Clone, Debug, Default)]
struct Metrics(HashMap<String, f64>);

impl Metrics {
    async fn fetch(client: &Client, base_url: &str) -> Result<Self, BoxError> {
        let response = client.get(format!("{base_url}/metrics")).send().await?;
        if !response.status().is_success() {
            return Err(CliError(format!(
                "metrics endpoint returned HTTP {}",
                response.status()
            ))
            .into());
        }
        Ok(Self::parse(&response.text().await?))
    }

    fn parse(text: &str) -> Self {
        let values = text
            .lines()
            .filter(|line| !line.starts_with('#'))
            .filter_map(|line| {
                let (name, value) = line.rsplit_once(' ')?;
                Some((name.to_owned(), value.parse::<f64>().ok()?))
            })
            .collect();
        Self(values)
    }

    fn value(&self, name: &str) -> f64 {
        self.0.get(name).copied().unwrap_or(0.0)
    }

    fn delta(&self, before: &Self, name: &str) -> f64 {
        self.value(name) - before.value(name)
    }
}

const REQUESTS: &str = "forge_requests_total";
const FINISHED: &str = "forge_requests_terminal_total{status=\"finished\"}";
const CANCELLED: &str = "forge_requests_terminal_total{status=\"cancelled\"}";
const FAILED: &str = "forge_requests_terminal_total{status=\"failed\"}";
const TTFT_SUM: &str = "forge_time_to_first_text_seconds_sum";
const TTFT_COUNT: &str = "forge_time_to_first_text_seconds_count";
const ITL_SUM: &str = "forge_inter_text_latency_seconds_sum";
const ITL_COUNT: &str = "forge_inter_text_latency_seconds_count";
const DURATION_SUM: &str = "forge_request_duration_seconds_sum";
const DURATION_COUNT: &str = "forge_request_duration_seconds_count";

#[derive(Debug, Serialize)]
struct ClientSummary {
    requests: usize,
    finished: usize,
    cancelled: usize,
    failed: usize,
    content_events: u64,
    wall_seconds: f64,
    request_duration_sum_seconds: f64,
    mean_ttft_seconds: Option<f64>,
    mean_itl_seconds: Option<f64>,
    p50_duration_seconds: Option<f64>,
    p95_duration_seconds: Option<f64>,
}

#[derive(Debug, Serialize)]
struct ServerDelta {
    requests: u64,
    finished: u64,
    cancelled: u64,
    failed: u64,
    ttft_count: u64,
    duration_count: u64,
    mean_ttft_seconds: Option<f64>,
    mean_itl_seconds: Option<f64>,
    mean_duration_seconds: Option<f64>,
}

#[derive(Debug, Serialize)]
struct Agreement {
    passed: bool,
    count_errors: Vec<String>,
    latency_errors: Vec<String>,
    mean_duration_difference_seconds: Option<f64>,
    mean_duration_tolerance_seconds: Option<f64>,
    latency_note: &'static str,
}

#[derive(Debug, Serialize)]
struct LoadReport {
    client: ClientSummary,
    server_delta: ServerDelta,
    agreement: Agreement,
    results: Vec<RequestResult>,
}

fn rounded_count(value: f64) -> u64 {
    value.max(0.0).round() as u64
}

fn metric_mean(after: &Metrics, before: &Metrics, sum: &str, count: &str) -> Option<f64> {
    let count = after.delta(before, count);
    (count > 0.0).then(|| after.delta(before, sum) / count)
}

async fn wait_for_terminal_metrics(
    client: &Client,
    base_url: &str,
    before: &Metrics,
    expected: usize,
    timeout: Duration,
) -> Result<Metrics, BoxError> {
    let deadline = Instant::now() + timeout;
    loop {
        let metrics = Metrics::fetch(client, base_url).await?;
        let terminal = metrics.delta(before, FINISHED)
            + metrics.delta(before, CANCELLED)
            + metrics.delta(before, FAILED);
        if terminal.round() >= expected as f64 {
            return Ok(metrics);
        }
        if Instant::now() >= deadline {
            return Err(CliError(format!(
                "server recorded {terminal:.0}/{expected} terminal requests within {}s",
                timeout.as_secs()
            ))
            .into());
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
}

async fn run_load(
    config: &Config,
    requests: usize,
    concurrency: usize,
    cancel_every: usize,
    cancel_after_events: usize,
    cancel_max_tokens: Option<u64>,
    metrics_timeout: Duration,
) -> Result<LoadReport, BoxError> {
    let client = Client::builder()
        .timeout(Duration::from_secs(300))
        .no_proxy()
        .build()?;
    let before = Metrics::fetch(&client, &config.base_url).await?;
    let semaphore = Arc::new(Semaphore::new(concurrency));
    let wall_started = Instant::now();
    let mut tasks = JoinSet::new();
    for index in 0..requests {
        let permit = Arc::clone(&semaphore).acquire_owned().await?;
        let client = client.clone();
        let base_url = config.base_url.clone();
        let prompt = config.prompt.clone();
        let cancellation =
            (cancel_every > 0 && (index + 1) % cancel_every == 0).then_some(cancel_after_events);
        let max_tokens = if cancellation.is_some() {
            cancel_max_tokens.unwrap_or(config.max_tokens)
        } else {
            config.max_tokens
        };
        tasks.spawn(async move {
            let _permit = permit;
            let started = Instant::now();
            consume(
                &client,
                &base_url,
                &prompt,
                max_tokens,
                index,
                cancellation,
                false,
            )
            .await
            .unwrap_or_else(|error| RequestResult::failed(index, started, error.as_ref()))
        });
    }
    let mut results = Vec::with_capacity(requests);
    while let Some(result) = tasks.join_next().await {
        results.push(result?);
    }
    results.sort_by_key(|result| result.index);
    let wall_seconds = wall_started.elapsed().as_secs_f64();
    let after = wait_for_terminal_metrics(
        &client,
        &config.base_url,
        &before,
        requests,
        metrics_timeout,
    )
    .await?;

    let finished = results
        .iter()
        .filter(|result| result.status == "finished")
        .count();
    let cancelled = results
        .iter()
        .filter(|result| result.status == "cancelled")
        .count();
    let failed = results
        .iter()
        .filter(|result| result.status == "failed")
        .count();
    let content_events = results.iter().map(|result| result.content_events).sum();
    let request_duration_sum_seconds: f64 =
        results.iter().map(|result| result.duration_seconds).sum();
    let ttft = results
        .iter()
        .filter_map(|result| result.ttft_seconds)
        .collect::<Vec<_>>();
    let itl = results
        .iter()
        .filter_map(|result| result.mean_itl_seconds)
        .collect::<Vec<_>>();
    let durations = results
        .iter()
        .map(|result| result.duration_seconds)
        .collect::<Vec<_>>();
    let client_mean_ttft = mean(&ttft);
    let client_mean_itl = mean(&itl);
    let client_mean_duration =
        (requests > 0).then(|| request_duration_sum_seconds / requests as f64);

    let server = ServerDelta {
        requests: rounded_count(after.delta(&before, REQUESTS)),
        finished: rounded_count(after.delta(&before, FINISHED)),
        cancelled: rounded_count(after.delta(&before, CANCELLED)),
        failed: rounded_count(after.delta(&before, FAILED)),
        ttft_count: rounded_count(after.delta(&before, TTFT_COUNT)),
        duration_count: rounded_count(after.delta(&before, DURATION_COUNT)),
        mean_ttft_seconds: metric_mean(&after, &before, TTFT_SUM, TTFT_COUNT),
        mean_itl_seconds: metric_mean(&after, &before, ITL_SUM, ITL_COUNT),
        mean_duration_seconds: metric_mean(&after, &before, DURATION_SUM, DURATION_COUNT),
    };
    let mut count_errors = Vec::new();
    compare_count(&mut count_errors, "requests", requests, server.requests);
    compare_count(&mut count_errors, "finished", finished, server.finished);
    compare_count(&mut count_errors, "cancelled", cancelled, server.cancelled);
    compare_count(&mut count_errors, "failed", failed, server.failed);
    compare_count(
        &mut count_errors,
        "duration samples",
        requests,
        server.duration_count,
    );
    let client_ttft_count = u64::try_from(ttft.len()).unwrap_or(u64::MAX);
    let max_server_ttft_count =
        client_ttft_count.saturating_add(u64::try_from(cancelled).unwrap_or(u64::MAX));
    if server.ttft_count < client_ttft_count || server.ttft_count > max_server_ttft_count {
        count_errors.push(format!(
            "TTFT samples: client={}, server={}, cancelled={cancelled}",
            ttft.len(),
            server.ttft_count
        ));
    }
    let mean_duration_difference_seconds = client_mean_duration
        .zip(server.mean_duration_seconds)
        .map(|(client_mean, server_mean)| (client_mean - server_mean).abs());
    let mean_duration_tolerance_seconds = server
        .mean_duration_seconds
        .map(|server_mean| 1.0_f64.max(server_mean * 0.5));
    let mut latency_errors = Vec::new();
    if mean_duration_difference_seconds
        .zip(mean_duration_tolerance_seconds)
        .is_some_and(|(difference, tolerance)| difference > tolerance)
    {
        latency_errors.push(format!(
            "mean duration difference {:.6}s exceeded {:.6}s tolerance",
            mean_duration_difference_seconds.unwrap_or_default(),
            mean_duration_tolerance_seconds.unwrap_or_default()
        ));
    }
    let agreement = Agreement {
        passed: count_errors.is_empty() && latency_errors.is_empty() && failed == 0,
        count_errors,
        latency_errors,
        mean_duration_difference_seconds,
        mean_duration_tolerance_seconds,
        latency_note: "Request, terminal-status, and duration counts match exactly. The scheduler may record TTFT/ITL before an explicit cancellation is processed, so server TTFT samples may exceed observed client samples by at most the cancellation count. Client duration includes transport; mean client/server duration must differ by no more than max(1s, 50% of the server mean).",
    };
    Ok(LoadReport {
        client: ClientSummary {
            requests,
            finished,
            cancelled,
            failed,
            content_events,
            wall_seconds,
            request_duration_sum_seconds,
            mean_ttft_seconds: client_mean_ttft,
            mean_itl_seconds: client_mean_itl,
            p50_duration_seconds: percentile(&durations, 0.50),
            p95_duration_seconds: percentile(&durations, 0.95),
        },
        server_delta: server,
        agreement,
        results,
    })
}

fn compare_count(errors: &mut Vec<String>, name: &str, client: usize, server: u64) {
    if u64::try_from(client).ok() != Some(server) {
        errors.push(format!("{name}: client={client}, server={server}"));
    }
}

fn mean(values: &[f64]) -> Option<f64> {
    (!values.is_empty()).then(|| values.iter().sum::<f64>() / values.len() as f64)
}

fn percentile(values: &[f64], fraction: f64) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let index = (sorted.len() as f64 * fraction).ceil() as usize - 1;
    sorted.get(index).copied()
}

#[tokio::main]
async fn main() {
    if let Err(error) = run().await {
        eprintln!("forge-streamer: {error}");
        std::process::exit(2);
    }
}

async fn run() -> Result<(), BoxError> {
    let config = Config::parse()?;
    match &config.command {
        Command::Chat => {
            let client = Client::builder()
                .timeout(Duration::from_secs(300))
                .no_proxy()
                .build()?;
            let result = consume(
                &client,
                &config.base_url,
                &config.prompt,
                config.max_tokens,
                0,
                None,
                true,
            )
            .await?;
            println!();
            eprintln!(
                "status={} ttft_seconds={:.6} duration_seconds={:.6}",
                result.status,
                result.ttft_seconds.unwrap_or_default(),
                result.duration_seconds
            );
        }
        Command::Load {
            requests,
            concurrency,
            cancel_every,
            cancel_after_events,
            cancel_max_tokens,
            metrics_timeout,
        } => {
            let report = run_load(
                &config,
                *requests,
                *concurrency,
                *cancel_every,
                *cancel_after_events,
                *cancel_max_tokens,
                *metrics_timeout,
            )
            .await?;
            println!("{}", serde_json::to_string(&report)?);
            if !report.agreement.passed {
                return Err(CliError(format!(
                    "client/server metrics disagree: {:?}",
                    report.agreement.count_errors
                ))
                .into());
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decoder_handles_fragmented_lf_and_crlf_frames() {
        let mut decoder = SseDecoder::default();
        assert!(decoder
            .push(b"data: {\"choices\":[{\"del")
            .unwrap()
            .is_empty());
        let events = decoder
            .push(b"ta\":{\"content\":\"Hi\"},\"finish_reason\":null}]}\r\n\r\ndata: [DONE]\n\n")
            .unwrap();
        assert_eq!(events.len(), 2);
        match &events[0] {
            Event::Chunk(chunk) => {
                assert_eq!(chunk.choices[0].delta.content, "Hi");
                assert!(chunk.choices[0].finish_reason.is_none());
            }
            Event::Done => panic!("expected chunk"),
        }
        assert!(matches!(events[1], Event::Done));
        decoder.finish().unwrap();
    }

    #[test]
    fn metrics_parser_preserves_labelled_sample_names() {
        let metrics = Metrics::parse(
            "# HELP ignored documentation\n\
             forge_requests_total 8\n\
             forge_requests_terminal_total{status=\"cancelled\"} 2\n",
        );
        assert_eq!(metrics.value(REQUESTS), 8.0);
        assert_eq!(metrics.value(CANCELLED), 2.0);
    }

    #[test]
    fn percentile_uses_nearest_rank_at_or_above_fraction() {
        assert_eq!(percentile(&[4.0, 1.0, 3.0, 2.0], 0.50), Some(2.0));
        assert_eq!(percentile(&[4.0, 1.0, 3.0, 2.0], 0.95), Some(4.0));
        assert_eq!(percentile(&[], 0.50), None);
    }
}
