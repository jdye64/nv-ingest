# nemo-retriever (Rust)

`nemo_retriever_rust/` is a from-scratch Rust rewrite of the NeMo Retriever
**service** layer. It targets the same HTTP API as the Python implementation
in `nemo_retriever/`, exchanging FastAPI + Ray for Axum + Tokio.

The Python implementation is **not** removed; you pick which one to deploy
via a single Helm value (`runtime: python|rust`).

## Why a Rust rewrite?

* **Lower latency / higher throughput.** Tokio + `reqwest` HTTP/2 keep-alive
  keeps NIM connections warm across requests. There is no per-request fork /
  ProcessPoolExecutor handshake, and the in-memory pipeline runs as plain
  async tasks instead of Ray actors.
* **Smaller container image.** A statically-linked `retriever-rs` binary is
  ~25 MB and the runtime image clocks in around 80 MB — versus a multi-GB
  Python+CUDA image for the original service.
* **Single binary, single configuration file.** The Rust runtime reads the
  same `retriever-service.yaml` Pydantic schema (mirrored as serde structs),
  so you can move a workload from python → rust by swapping the image.

## What's intentionally **not** ported

These were declared out of scope by the user when starting this rewrite:

* `batch` run_mode (`retriever batch ...`) and the entire Ray Data pipeline.
* Local Hugging Face / GPU inference. The Rust executor only knows how to
  call remote NIM endpoints over HTTP.
* The Python `pipeline` subcommands (sweep, eval, recall). These continue
  to work via the Python CLI; the Rust binary only ships `service` and
  `client` subcommands.

If a feature isn't ported, the corresponding HTTP endpoint either returns
the same shape as the Python service (when feasible) or
`501 Not Implemented` so callers can fall back gracefully.

## Layout

```
nemo_retriever_rust/
├── Cargo.toml                ← workspace manifest
├── rust-toolchain.toml       ← pin stable rustc
├── crates/
│   ├── nrr-core/             ← config, models, policy, job tracker, event bus, sidecar store
│   ├── nrr-nim/              ← async HTTP clients for embed/OCR/page-elements/...
│   ├── nrr-vdb/              ← LanceDB row builder + HTTP client to vectordb pod
│   ├── nrr-pipeline/         ← in-memory executor + bounded mpsc worker pool
│   ├── nrr-service/          ← Axum app, middleware, routes, gateway proxy, metrics
│   ├── nrr-cli/              ← `retriever-rs` binary
│   └── nrr-client/           ← async client SDK
└── docker/
    └── Dockerfile.rust       ← multi-stage build → ~80 MB runtime image
```

## Build & run

```bash
# Local build (requires Rust ≥ 1.95):
cd nemo_retriever_rust
cargo build --release --bin retriever-rs

# Boot the service in standalone mode on port 7670:
./target/release/retriever-rs service start \
  --config crates/nrr-core/assets/retriever-service.yaml

# Probe it:
curl -s http://localhost:7670/v1/health | jq
```

```bash
# Docker build (run from the repo root):
docker build \
  -f nemo_retriever_rust/docker/Dockerfile.rust \
  -t nemo-retriever-rust:dev .
```

## Helm

```bash
# Deploy the python runtime (default):
helm install retriever nemo_retriever/helm

# Deploy the rust runtime:
helm install retriever nemo_retriever/helm --set runtime=rust

# Override the rust image tag:
helm install retriever nemo_retriever/helm \
  --set runtime=rust \
  --set runtimeImages.rust.repository=my-registry/nemo-retriever-rust \
  --set runtimeImages.rust.tag=v0.1.0
```

The `runtime` toggle switches the container image, image-pull policy, and
`args` on the main retriever Deployment. Everything else (ConfigMap,
Service, HPA, ingress, …) is unchanged. The dedicated `vectordb` Deployment
continues to run the python LanceDB pod for both runtimes.

## API parity matrix

| Endpoint                                          | Python | Rust |
|---------------------------------------------------|:------:|:----:|
| `GET  /v1/health`                                 | ✅     | ✅   |
| `GET  /v1/metrics`                                | ✅     | ✅   |
| `GET  /v1/admin/pool-stats`                       | ✅     | ✅   |
| `GET  /v1/ingest/pipeline-config`                 | ✅     | ✅   |
| `POST /v1/ingest/job`                             | ✅     | ✅   |
| `GET  /v1/ingest/job/{id}`                        | ✅     | ✅   |
| `GET  /v1/ingest/job/{id}/documents`              | ✅     | ✅   |
| `GET  /v1/ingest/job/{id}/document/{doc_id}`      | ✅     | ✅   |
| `POST /v1/ingest/job/{id}/document`               | ✅     | ✅   |
| `POST /v1/ingest/job/{id}/page`                   | ✅     | ✅   |
| `POST /v1/ingest/job/{id}/whole`                  | ✅     | ✅   |
| `POST /v1/ingest/sidecar`                         | ✅     | ✅   |
| `DELETE /v1/ingest/sidecar/{id}`                  | ✅     | ✅   |
| `GET  /v1/ingest/status/{id}`                     | ✅     | ✅   |
| `GET  /v1/ingest/page/status/{id}`                | ✅     | ✅   |
| `GET  /v1/ingest/document/status/{id}`            | ✅     | ✅   |
| `POST /v1/ingest/status/batch`                    | ✅     | ✅   |
| `GET  /v1/ingest/job/{id}/events` (SSE)           | ✅     | ✅   |
| `POST /v1/internal/job-callback`                  | ✅     | ✅   |
| `POST /v1/query`                                  | ✅     | ✅ (proxies to vectordb pod) |

## Development workflow

```bash
cargo fmt --all                # format
cargo clippy --workspace -- -D warnings   # lints
cargo test  --workspace        # unit tests (10 inside nrr-core today)
cargo build --workspace        # full release build
```

## Known parity gaps

* **PDF chart / table / OCR detection** is delegated to remote NIMs only —
  there is no local YOLO / pdfium fallback path. Configure
  `nim_endpoints.page_elements_invoke_url` etc. in your YAML to enable it.
* **VectorDB writes** flow through the dedicated `vectordb` pod over HTTP.
  A direct `lancedb-rs`-backed writer behind a Cargo feature flag is on the
  roadmap.
* **Audio / video extraction** stages are not wired in this initial rewrite;
  uploads of those types currently get treated as opaque text blobs.
