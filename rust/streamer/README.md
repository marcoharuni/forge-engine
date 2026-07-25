# ForgeEngine Rust streamer

This crate is a small asynchronous client for ForgeEngine's streaming
OpenAI-compatible endpoint. It does not contain or duplicate inference code.

Stream one chat:

```bash
cargo run --release --manifest-path rust/streamer/Cargo.toml -- \
  chat --base-url http://127.0.0.1:8000 --prompt "Count from 1 to 8."
```

Run eight requests with at most four active at once, cancelling every fourth
request immediately after admission:

```bash
cargo run --release --manifest-path rust/streamer/Cargo.toml -- \
  load --base-url http://127.0.0.1:8000 --requests 8 --concurrency 4 \
  --cancel-every 4 --cancel-after-events 0 --cancel-max-tokens 512
```

The load command emits one JSON report. Client-side TTFT, inter-text latency,
duration percentiles, and wall time include HTTP and streaming transport.
Server-side metrics begin after admission. Request, terminal-status, TTFT, and
duration sample counts are compared automatically. Request, terminal-status,
and duration counts must agree exactly. The server may record TTFT before a
disconnect reaches the client, so its TTFT count may exceed the client count by
at most the cancellation count. Mean client/server duration must be within
`max(1 second, 50% of the server mean)`; both latency views remain separate
because their measurement boundaries differ. A zero
`--cancel-after-events` value calls ForgeEngine's cancellation endpoint
immediately after reading the request ID from the response header; a positive
value cancels after that many non-empty content events. Rust consumes the
terminal SSE event before classifying the request, so its status is confirmed
by the server rather than inferred from a dropped connection.
`--cancel-max-tokens` can give cancellation targets a longer limit than normal
load requests when testing cancellation after content events.
