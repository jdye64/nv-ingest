# retriever service

Run the long-lived retriever ingest HTTP service, or submit documents to a
running one.

Two subcommands:

- `retriever service start` — start the FastAPI/Uvicorn server (`uvicorn` +
  the worker pool described in `retriever-service.yaml`).
- `retriever service ingest <files…>` — submit document files to a running
  server. Streams progress via SSE by default, or polls when `--no-sse`.

If flags look stale, re-check `retriever service <cmd> --help`.

## When to use this

- The same pipeline will be hit by many ingest requests (model load is
  amortised across requests — far cheaper than one-shot CLI).
- You want a long-running HTTP/SSE endpoint to drive from another service
  / UI.
- You're running `retriever pipeline run … --run-mode service` (which talks
  to a server you started with `retriever service start`).

**Use a different command when:**

- You only have a handful of one-shot ingests → [ingest](ingest.md) or
  [pipeline](pipeline.md) with `--run-mode inprocess`/`batch`.

## Canonical invocations

### Start the server

Default config (auto-discovers `retriever-service.yaml` next to the
service package, then `./retriever-service.yaml`, then `$HOME/...`):

```bash
retriever service start
```

Explicit config + override host/port + log level:

```bash
retriever service start \
  --config retriever-service.yaml \
  --host 0.0.0.0 --port 7670 \
  --log-level debug --log-file /var/log/retriever-service.log
```

With auth + custom NIM key + restrict to GPUs 0,1:

```bash
retriever service start \
  --api-token "$NEMO_RETRIEVER_API_TOKEN" \
  --nim-api-key "$NVIDIA_API_KEY" \
  --gpu-devices 0,1
```

### Submit documents to a running server

SSE streaming (default) — best for interactive use:

```bash
retriever service ingest data/multimodal_test.pdf
```

Custom server URL + multiple files + auth + bumped concurrency:

```bash
retriever service ingest data/pdfs/*.pdf \
  --server-url http://retriever:7670 \
  --concurrency 16 \
  --api-token "$NEMO_RETRIEVER_API_TOKEN"
```

Polling mode (when SSE is blocked by a proxy):

```bash
retriever service ingest data/big.pdf --no-sse --poll-interval 2.0
```

## Inputs

### `start`
- All flags are optional; YAML is the source of truth, CLI flags override.

### `ingest`
- **Positional `FILES`** — one or more file paths. Globs are expanded by
  your shell; the command does not expand globs itself.

## Key flags

### `start`
| Flag | Default | Notes |
|---|---|---|
| `--config`, `-c` | autodiscover | YAML path. |
| `--host` | from YAML | Bind address override. |
| `--port`, `-p` | from YAML | Listen port override. |
| `--log-level` | from YAML | `debug`, `info`, `warning`, `error`. |
| `--log-file` | from YAML | Server log file. |
| `--nim-api-key` | `$NVIDIA_API_KEY` | Token for NIM endpoints. |
| `--gpu-devices` | from YAML | Comma-separated CUDA device IDs. |
| `--api-token` | `$NEMO_RETRIEVER_API_TOKEN` | Required bearer when set. |

### `ingest`
| Flag | Default | Notes |
|---|---|---|
| `--server-url`, `-s` | `http://localhost:7670` | Base URL. |
| `--sse / --no-sse` | `--sse` | SSE streaming vs polling. |
| `--poll-interval` | `2.0` s | Polling cadence when `--no-sse`. |
| `--concurrency` | `8` | Max concurrent uploads. |
| `--api-token` | `$NEMO_RETRIEVER_API_TOKEN` | Bearer for every request. |

## Outputs

- **`start`**: blocks; uvicorn logs to stdout / `--log-file`. The server
  process exposes `/v1/ingest`, `/v1/query`, `/v1/health`, and the SSE
  stream endpoints.
- **`ingest`**: streams per-page completion events; final return shows
  per-document upload_complete / document_complete / upload_failed totals.
  Returns non-zero if any document failed.

## Common failure modes

- **`Address already in use`** — another retriever (or anything) is
  listening on the configured port; override with `--port`.
- **`401 Unauthorized`** — server was started with `--api-token`; clients
  must pass the same token via `--api-token` or
  `$NEMO_RETRIEVER_API_TOKEN`.
- **SSE hangs behind a reverse proxy** — proxy is buffering. Use
  `--no-sse --poll-interval 2`.
- **`HTTP 503` / "service starting"** — server is still loading models;
  retry after the first ingest finishes warming things up.

## Related

- [ingest](ingest.md) / [pipeline](pipeline.md) — non-service execution
  modes (one-shot CLI).
- [query](query.md) — querying the LanceDB table the service wrote to is
  identical to querying any other table.
- See `nemo_retriever/src/nemo_retriever/service/retriever-service.yaml`
  for the default config layout.
