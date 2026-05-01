# Service Module Architecture

This document describes the internal architecture of the `nemo_retriever.service` module — the FastAPI-based service that provides HTTP-driven document ingestion, vector search, and reranking.

---

## Table of Contents

- [High-Level Overview](#high-level-overview)
- [Directory Layout](#directory-layout)
- [Request Lifecycle](#request-lifecycle)
  - [Ingest Flow](#ingest-flow)
  - [Query Flow](#query-flow)
  - [Rerank Flow](#rerank-flow)
- [Core Subsystems](#core-subsystems)
  - [Application Factory & Lifespan](#application-factory--lifespan)
  - [Configuration](#configuration)
  - [Processing Pool](#processing-pool)
  - [Database Layer](#database-layer)
  - [Spool (Durability)](#spool-durability)
  - [Event Bus & SSE Streaming](#event-bus--sse-streaming)
  - [Authentication](#authentication)
  - [Metrics](#metrics)
  - [Failure Classification](#failure-classification)
- [Design Decisions](#design-decisions)
- [Client Library](#client-library)
- [Developer Guide: Extending the Service](#developer-guide-extending-the-service)
  - [Adding a New Router](#adding-a-new-router)
  - [Adding a New NIM Endpoint](#adding-a-new-nim-endpoint)
  - [Adding a New Pydantic Model](#adding-a-new-pydantic-model)
  - [Adding a New Configuration Section](#adding-a-new-configuration-section)
  - [Adding a Capability Flag](#adding-a-capability-flag)
  - [Adding a Pipeline Stage](#adding-a-pipeline-stage)
  - [Adding a New SSE Event Type](#adding-a-new-sse-event-type)

---

## High-Level Overview

```
                 ┌─────────────────────────────────────────────┐
                 │              FastAPI Application             │
                 │                                             │
   HTTP ────────►│  Routers ─────► Processing Pool ──► Workers │
   requests      │    │                  │                     │
                 │    │            Batch Buffer                │
                 │    │                  │                     │
                 │    ▼                  ▼                     │
                 │  SQLite DB     Worker Processes             │
                 │    │           (operator chains)            │
                 │    │                  │                     │
                 │    │                  ▼                     │
                 │    │            NIM Endpoints               │
                 │    │          (OCR, embed, etc.)            │
                 │    │                  │                     │
                 │    ▼                  ▼                     │
                 │  Event Bus ◄──── Results ──────► SQLite DB  │
                 │    │                                        │
                 │    ▼                                        │
                 │  SSE Stream ──────────────────────► Clients │
                 └─────────────────────────────────────────────┘
```

The service operates as a single-process asyncio application (the "main process") that manages a pool of worker subprocesses via `ProcessPoolExecutor`. Each worker builds its own operator chain at startup and processes document pages in batches. Results are written directly to SQLite by workers and published as SSE events by the main process.

Two orthogonal features — **vector search** (`/v1/query`) and **reranking** (`/v1/rerank`) — run synchronously in the main process via `asyncio.to_thread`, delegating to external NIM endpoints. They do not use the processing pool.

---

## Directory Layout

```
service/
├── ARCHITECTURE.md          # This file
├── __init__.py
├── app.py                   # FastAPI application factory + lifespan
├── auth.py                  # Bearer-token authentication middleware
├── cli.py                   # Typer CLI: `retriever service start/ingest`
├── client.py                # Async client for ingesting documents
├── config.py                # Pydantic configuration models + YAML loader
├── event_bus.py             # In-memory pub/sub for SSE streaming
├── failure_types.py         # FailureType enum + exception classifier
├── metrics.py               # Prometheus instrumentation + gauge refresh
├── retriever-service.yaml   # Bundled default configuration
├── spool.py                 # Durable page spool (write-ahead)
├── db/
│   ├── __init__.py
│   ├── engine.py            # Thread-local SQLite connections + DDL
│   └── repository.py        # CRUD operations for all tables
├── models/
│   ├── __init__.py
│   ├── document.py          # Document ORM model + ProcessingStatus enum
│   ├── job.py               # Job ORM model
│   ├── metrics.py           # ProcessingMetric ORM model
│   ├── page_processing_log.py  # Per-page audit log model
│   ├── page_result.py       # PageResult ORM model
│   ├── requests.py          # Pydantic request models (API input)
│   └── responses.py         # Pydantic response models (API output)
├── processing/
│   ├── __init__.py
│   └── pool.py              # ProcessPoolExecutor + batch buffer
└── routers/
    ├── __init__.py
    ├── ingest.py            # POST /v1/ingest, job management, batch upload
    ├── internal.py          # Internal/admin endpoints
    ├── metrics.py           # GET /v1/ingest_metrics
    ├── query.py             # POST /v1/query (vector search)
    ├── rerank.py            # POST /v1/rerank (cross-encoder reranking)
    ├── stream.py            # SSE streaming endpoints
    └── system.py            # GET /v1/health, GET /v1/capabilities
```

---

## Request Lifecycle

### Ingest Flow

This is the primary flow — uploading documents for extraction and embedding.

```
Client                     Service                    Worker Process
  │                          │                              │
  │  POST /v1/ingest         │                              │
  │  (multipart: file+meta)  │                              │
  │─────────────────────────►│                              │
  │                          │  1. Parse multipart          │
  │                          │  2. SHA-256 content hash     │
  │                          │  3. Spool to disk (fsync)    │
  │                          │  4. Insert document row      │
  │                          │  5. Enqueue into BatchBuffer │
  │  202 Accepted ◄──────────│                              │
  │                          │                              │
  │                          │  When batch full OR timeout: │
  │                          │  Submit to ProcessPoolExecutor
  │                          │─────────────────────────────►│
  │                          │                              │  6. Read bytes from spool
  │                          │                              │  7. Build DataFrame
  │                          │                              │  8. Run operator chain
  │                          │                              │     (PDF parse → OCR →
  │                          │                              │      table detect →
  │                          │                              │      embed → ...)
  │                          │                              │  9. Write results to SQLite
  │                          │  BatchWorkerResult ◄─────────│
  │                          │                              │
  │                          │  10. Publish SSE events      │
  │  SSE: page_complete ◄────│                              │
  │  SSE: metrics_update ◄───│                              │
  │  SSE: document_complete ◄│                              │
  │  SSE: job_complete ◄─────│                              │
```

**Key design points:**

- **Accept-before-process**: The endpoint returns 202 immediately after spooling to disk. Processing is fully asynchronous.
- **Batch dispatch**: Pages accumulate in a `_BatchBuffer` (default 32) and flush either when full or after a timeout (default 2s). This ensures NIM endpoints receive efficient batches.
- **Worker isolation**: Each worker is a separate OS process with its own operator chain. No shared mutable state, no C-library thread safety concerns.
- **Dual-write**: Workers write results to SQLite directly. The main process receives a lightweight `BatchWorkerResult` for SSE event publishing only.

### Query Flow

Vector search — synchronous, does not use the processing pool.

```
Client                     Service
  │                          │
  │  POST /v1/query          │
  │  {"query": "...",        │
  │   "top_k": 10}           │
  │─────────────────────────►│
  │                          │  1. Validate config (embed endpoint)
  │                          │  2. Call embedding NIM (asyncio.to_thread)
  │                          │  3. Search LanceDB (asyncio.to_thread)
  │                          │  4. Build QueryResponse
  │  200 QueryResponse ◄─────│
```

The query router accepts single or batch queries (`str | list[str]`), with optional hybrid search (vector + BM25 via LanceDB's `RRFReranker`).

### Rerank Flow

Cross-encoder reranking — also synchronous, delegates to the reranker NIM.

```
Client                     Service
  │                          │
  │  POST /v1/rerank         │
  │  {"query": "...",        │
  │   "passages": [...]}     │
  │─────────────────────────►│
  │                          │  1. Validate config (rerank endpoint)
  │                          │  2. Extract text from passages
  │                          │  3. Call reranker NIM (asyncio.to_thread)
  │                          │  4. Sort by score, apply top_n
  │                          │  5. Build RerankResponse
  │  200 RerankResponse ◄────│
```

The rerank endpoint accepts passages as arbitrary dicts (must contain `text`). All extra keys survive the round-trip, so clients can send full hit dicts from `/v1/query` and get them back with `rerank_score` added.

---

## Core Subsystems

### Application Factory & Lifespan

**File:** `app.py`

`create_app(config)` builds the FastAPI application:

1. Configures root logging (console + rotating file)
2. Applies resource limits (CPU affinity, memory rlimit, CUDA devices)
3. Conditionally attaches `BearerAuthMiddleware`
4. Registers all routers under `/v1`
5. Sets up Prometheus instrumentation

The `_lifespan` async context manager handles startup and shutdown:

- **Startup**: initializes SQLite, creates `EventBus`, optionally creates `SpoolStore`, starts `ProcessingPool`, recovers spooled pages, starts metrics refresh loop.
- **Shutdown**: cancels metrics task, drains pool (with configurable timeout), shuts down pool, closes DB.

### Configuration

**File:** `config.py`

All configuration is Pydantic-validated with `extra="forbid"` to catch typos. The top-level `ServiceConfig` composes:

| Section | Class | Purpose |
|---------|-------|---------|
| `server` | `ServerConfig` | Host/port bindings |
| `logging` | `LoggingConfig` | Level, file, format |
| `database` | `DatabaseConfig` | SQLite path |
| `processing` | `ProcessingConfig` | Workers, batch size, timeout |
| `nim_endpoints` | `NimEndpointsConfig` | NIM URLs (comma-separated for multi-NIM) |
| `resources` | `ResourceLimitsConfig` | Memory, CPU, GPU constraints |
| `auth` | `AuthConfig` | Bearer token + bypass paths |
| `drain` | `DrainConfig` | Graceful shutdown timeout |
| `spool` | `SpoolConfig` | Durable page spool settings |
| `event_bus` | `EventBusConfig` | SSE overflow policy + queue sizing |
| `vector_store` | `VectorStoreConfig` | LanceDB URI, table, top_k, embedding model |
| `reranker` | `RerankerConfig` | Reranker model name, default top_n |

**Config precedence** (highest to lowest):
1. CLI flags (`--port`, `--embed-url`, etc.)
2. `./retriever-service.yaml` in the working directory
3. Bundled default `retriever-service.yaml` in the package

`load_config()` merges YAML and CLI overrides using dotted-key notation (e.g., `server.port`).

### Processing Pool

**File:** `processing/pool.py`

The pool has three main components:

**`ProcessingPool`** — main-process coordinator:
- Owns the `ProcessPoolExecutor` (spawned, not forked, for C-library safety)
- Distributes comma-separated NIM URLs round-robin across workers
- Manages drain/cancel state
- Publishes SSE events when workers complete

**`_BatchBuffer`** — thread-safe page accumulator:
- Accepts individual pages from the ingest router
- Auto-flushes when reaching `batch_size` or after `timeout_s`
- Enforces a hard cap (`num_workers * batch_size`) to bound memory
- Returns `False` when full, causing the ingest router to 503

**Worker processes** — isolated operator chains:
- Each worker calls `_worker_initializer` once, which builds the full `nv-ingest` operator chain
- `_run_pipeline_batch` receives page descriptors, builds a DataFrame, runs the chain, and writes results to SQLite
- Provenance columns (`_page_document_id`, etc.) track which output rows belong to which input page through content-explosion stages

**Multi-NIM load balancing:** When a NIM URL field contains commas (e.g., `ocr_invoke_url: "http://nim1:8000,http://nim2:8000"`), workers are assigned URLs round-robin so each worker talks to exactly one endpoint per NIM type.

### Database Layer

**Files:** `db/engine.py`, `db/repository.py`

**`DatabaseEngine`** manages thread-local SQLite connections:
- WAL journal mode for concurrent multi-process writes
- 60-second busy timeout to handle write contention
- Application-level retry with exponential backoff via `execute_with_retry`
- Additive schema migrations (`_safe_add_column`) so upgrades don't require dropping the DB

**Schema** (5 tables):

| Table | Purpose |
|-------|---------|
| `jobs` | One row per uploaded file. Tracks total pages, completion count. |
| `documents` | One row per page. Links to parent job, stores spool path. |
| `page_results` | Pipeline output rows (JSON content), keyed by document. |
| `processing_metrics` | Per-model detection/invocation counts per document. |
| `page_processing_log` | Audit trail: timing, failure type, error messages. |

**`Repository`** provides typed CRUD operations. All write methods wrap with `execute_with_retry` to transparently handle `OperationalError: database is locked`.

### Spool (Durability)

**File:** `spool.py`

The spool closes the gap between "page accepted (202)" and "page dispatched to a worker". Without it, an OOM kill or pod restart loses every accepted-but-unprocessed page.

**Write path:**
```
1. Accept multipart upload
2. SpoolStore.write(sha256, bytes)     # atomic: write → fsync → rename
3. INSERT document row (spool_path=...)
4. Return 202
```

**Recovery path (startup):**
```
1. Query documents WHERE status IN ('queued', 'processing') AND spool_path IS NOT NULL
2. For each: read spool file → pool.try_submit()
```

**Cleanup:**
A background asyncio task periodically queries terminal documents (complete/failed/cancelled), unlinks their spool files, and clears `spool_path`.

**Layout:** `<root>/<sha[0:2]>/<sha[2:4]>/<sha>.bin` — sharded so no directory exceeds ~256 files.

### Event Bus & SSE Streaming

**File:** `event_bus.py`

The `EventBus` is a per-key fan-out pub/sub system. Keys are `document_id` or `job_id` strings. Events published under a key are delivered to every subscription for that key.

**Backpressure model** (three layered defenses):

1. **Per-subscription event-type filter** — subscribers declare which event types they care about. Filtered events never touch the queue.

2. **Priority-drop load shedding** — when the queue fills past a watermark (75%), low-priority events (`metrics_update`, `page_complete`, `document_complete`) are silently dropped. Terminal events (`job_complete`, `status_change`, `page_result`) are always preserved.

3. **Stream overflow sentinel** — if a terminal event hits a full queue, the subscription is marked overflowed, the queue is drained, and a `stream_overflow` event is injected. The client reconnects with `Last-Event-ID` to resume from the replay buffer.

**Overflow policies** (configurable):

| Policy | Behavior |
|--------|----------|
| `drop_low_priority` (default) | Synchronous, never blocks. Sheds low-priority events. |
| `backpressure` | Async `await Queue.put()` with timeout. Back-pressures the worker pool. |
| `block` | Same as backpressure but waits forever. Zero loss, unbounded latency. |

**Replay buffer:** Every event gets a monotonic `seq` number and is stored in a per-key ring buffer (default 1024 events). Reconnecting clients send `Last-Event-ID` and replay missed events.

### Authentication

**File:** `auth.py`

`BearerAuthMiddleware` is a Starlette middleware that:
- Compares the incoming `Authorization` header against the configured token using `hmac.compare_digest` (constant-time)
- Bypasses configured paths (default: `/v1/health`, `/docs`, `/openapi.json`, `/redoc`) so Kubernetes probes and API docs work without tokens
- Is only attached when `auth.api_token` is non-empty — zero overhead when auth is disabled

### Metrics

**File:** `metrics.py`

Two layers of Prometheus metrics at `GET /metrics`:

1. **HTTP-level** (via `prometheus-fastapi-instrumentator`): request totals, latency histograms, in-progress counts by route and status code.

2. **Service-specific gauges** (refreshed every 15s by a background task):
   - `nemo_retriever_pool_workers` — configured worker count
   - `nemo_retriever_pool_capacity_pages` — free batch buffer slots
   - `nemo_retriever_pool_buffered_pages` — pages waiting for dispatch
   - `nemo_retriever_pool_in_flight_batches` — batches currently executing
   - `nemo_retriever_pool_draining` — 1 during graceful shutdown
   - `nemo_retriever_jobs{status=...}` — job counts by status

### Failure Classification

**File:** `failure_types.py`

The `FailureType` enum classifies per-page failures into actionable categories:

| Type | Meaning | Retryable? |
|------|---------|------------|
| `pdf_parse` | Corrupt/password-protected PDF | No |
| `nim_timeout` | NIM endpoint timed out | Yes |
| `nim_5xx` | NIM returned server error | Yes |
| `nim_4xx` | NIM returned client error | No (check input) |
| `oom` | Out of memory | Maybe (reduce batch) |
| `internal` | Pipeline bug | No (file an issue) |
| `cancelled` | Cancelled before processing | N/A |
| `unknown` | Unrecognized exception | Depends |

`categorize_exception(exc)` inspects the exception type and message string to map real exceptions to these categories.

---

## Design Decisions

### Why `ProcessPoolExecutor` instead of threads?

The document processing pipeline uses C-extension libraries (pypdfium2, image processing) that are not thread-safe and hold the GIL during CPU-intensive operations. Separate processes eliminate both problems:
- Each process has its own address space — no C-library thread safety issues
- True CPU parallelism for image processing and model inference
- Clean isolation: a segfault in one worker doesn't crash the service

### Why batch buffering?

NIM endpoints and GPU models are most efficient with larger batches. The `_BatchBuffer` accumulates pages and dispatches them as a single DataFrame, so downstream models see 32 samples instead of 1. The configurable timeout ensures latency stays bounded for small jobs.

### Why SQLite instead of Postgres?

The service targets single-node deployments (one pod with a PV). SQLite's WAL mode handles concurrent writes from multiple worker processes with zero operational overhead. The busy-timeout + retry pattern makes write contention transparent. For multi-node deployments, the DB layer could be swapped via the `Repository` abstraction.

### Why spool-then-accept?

The 202 response is a contract: "your work will not be lost." Without the spool, that contract is broken by any crash between accept and processing. The fsync + atomic rename pattern guarantees the page bytes survive any failure mode short of disk corruption.

### Why drop-on-overflow instead of blocking?

The default `drop_low_priority` policy ensures the worker callback thread (which publishes SSE events) never blocks. A blocked worker thread back-pressures the `ProcessPoolExecutor`, which cascades to 503s on the ingest endpoint. The dropped events (`metrics_update`, `page_complete`) are reconstructible from REST endpoints, so the client loses observability but not correctness.

### Why `asyncio.to_thread` for query/rerank?

The embedding NIM call and LanceDB search are blocking I/O operations. Running them in `asyncio.to_thread` keeps the event loop responsive for SSE streaming and health checks, without the complexity of a worker process for what is a simple RPC + query.

---

## Client Library

**File:** `client.py`

`RetrieverServiceClient` is a full-featured async client that:

1. **Splits PDFs** client-side using pypdfium2 (each page uploaded individually)
2. **Uploads pages concurrently** (configurable concurrency, default 8)
3. **Opens SSE in parallel** with uploads — no events are missed
4. **Handles back-pressure** — retries on 503 with exponential backoff, respects `Retry-After` header
5. **Reconnects on transport failure** — TCP resets (common on Kubernetes NodePort) trigger automatic reconnection with `Last-Event-ID`
6. **Rich live display** — compact progress bars, active job table, detection metrics
7. **Streaming generator** — `aingest_documents_stream()` yields per-page results as they arrive

The CLI command `retriever service ingest` wraps this client:

```bash
retriever service ingest doc1.pdf doc2.pdf \
    --server-url http://localhost:7670 \
    --concurrency 16 \
    --api-token $TOKEN
```

---

## Developer Guide: Extending the Service

### Adding a New Router

1. Create `routers/my_feature.py`:

```python
from __future__ import annotations

import logging
from fastapi import APIRouter, HTTPException, Request

from nemo_retriever.service.config import ServiceConfig

logger = logging.getLogger(__name__)
router = APIRouter(tags=["my_feature"])


@router.post("/my_feature", summary="Do something useful")
async def my_feature(request: Request) -> dict:
    config: ServiceConfig = request.app.state.config
    # Access shared state via request.app.state:
    #   config, repository, processing_pool, event_bus, spool_store
    return {"status": "ok"}
```

2. Register in `app.py`:

```python
from nemo_retriever.service.routers import ..., my_feature

app.include_router(my_feature.router, prefix="/v1")
```

**Convention:** All NIM-calling routers use `asyncio.to_thread` for blocking operations, catch specific exception types, and return descriptive `HTTPException` details with 400/503 status codes.

### Adding a New NIM Endpoint

1. Add the URL field to `NimEndpointsConfig` in `config.py`:

```python
class NimEndpointsConfig(BaseModel):
    # ... existing fields ...
    my_nim_invoke_url: str | None = None
```

2. Add a CLI flag in `cli.py` if desired:

```python
my_nim_url: Optional[str] = typer.Option(None, "--my-nim-url", help="...")
# Then in the overrides dict:
if my_nim_url is not None:
    overrides["nim_endpoints.my_nim_invoke_url"] = my_nim_url
```

3. Add the NIM URL field to the `_NIM_URL_FIELDS` tuple in `processing/pool.py` if it should be round-robin distributed across workers:

```python
_NIM_URL_FIELDS = (
    # ... existing ...
    "my_nim_invoke_url",
)
```

### Adding a New Pydantic Model

**Request models** go in `models/requests.py`:

```python
class MyRequest(BaseModel):
    """JSON body for ``POST /v1/my_feature``."""
    field: str = Field(..., description="Required field.")
    optional_field: int | None = Field(default=None, ge=1, le=100)
```

**Response models** go in `models/responses.py`:

```python
class MyResponse(BaseModel):
    """Response for ``POST /v1/my_feature``."""
    result: str
    count: int = Field(description="How many items were processed.")
```

**DB/ORM models** go in their own file under `models/` (e.g., `models/my_entity.py`) and follow the `to_row()` / `from_row()` pattern used by `Document` and `Job`.

### Adding a New Configuration Section

1. Define the Pydantic model in `config.py`:

```python
class MyFeatureConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    threshold: float = 0.5
```

2. Add it to `ServiceConfig`:

```python
class ServiceConfig(BaseModel):
    # ... existing fields ...
    my_feature: MyFeatureConfig = Field(default_factory=MyFeatureConfig)
```

3. Access it from any router via `request.app.state.config.my_feature`.

4. Override from YAML:

```yaml
my_feature:
  enabled: true
  threshold: 0.8
```

### Adding a Capability Flag

The capabilities endpoint lets clients introspect what the server supports.

1. Add the field to `CapabilitiesResponse` in `models/responses.py`:

```python
class CapabilitiesResponse(BaseModel):
    # ... existing fields ...
    my_feature: CapabilityFlag
```

2. Wire it in `routers/system.py`:

```python
return CapabilitiesResponse(
    # ... existing ...
    my_feature=_flag_for(nim.my_nim_invoke_url),
)
```

### Adding a Pipeline Stage

Pipeline stages (operators) live outside this module in `nemo_retriever.graph`, but they are wired into the service via the processing pool.

1. Implement the operator (subclass of `AbstractOperator` with CPU/GPU variants)
2. Register it in the graph builder (`nemo_retriever.graph.ingestor_runtime.build_graph`)
3. If it requires a NIM endpoint, add the URL field per the [Adding a New NIM Endpoint](#adding-a-new-nim-endpoint) guide
4. The pool's `_build_operator_chain` will automatically pick it up via `build_graph` → `resolve_graph` → `_linearize`

### Adding a New SSE Event Type

1. Publish the event from the pool's result handler in `processing/pool.py`:

```python
self._publish_event(
    doc_id,
    {
        "event": "my_event",
        "document_id": doc_id,
        "job_id": job_id,
        "custom_field": value,
    },
)
```

2. If the event is low-priority (safe to drop under load), add it to `_DEFAULT_PRIORITY_DROP` in `event_bus.py`:

```python
_DEFAULT_PRIORITY_DROP: frozenset[str] = frozenset({
    "metrics_update",
    "page_complete",
    "document_complete",
    "my_event",  # safe to drop — client can reconstruct from REST
})
```

3. Handle the event in `client.py`'s `_handle_sse_event` and/or `_stream_consumer` if the CLI client needs to react to it.

**Important:** Terminal events (`job_complete`, `status_change`) must NOT be added to the priority-drop set — they are needed for correctness.

---

## API Endpoint Summary

| Method | Path | Router | Purpose |
|--------|------|--------|---------|
| POST | `/v1/ingest` | `ingest` | Upload a single page for processing |
| POST | `/v1/ingest/batch` | `ingest` | Upload multiple pages in one request |
| POST | `/v1/ingest/job` | `ingest` | Upload a full PDF for server-side splitting |
| GET | `/v1/ingest/status/{doc_id}` | `ingest` | Document status + results |
| GET | `/v1/ingest/job/{job_id}` | `ingest` | Job status |
| GET | `/v1/ingest/job/{job_id}/results` | `ingest` | Full reassembled job results |
| GET | `/v1/ingest/jobs` | `ingest` | Paginated job listing |
| GET | `/v1/ingest/jobs/summary` | `ingest` | Aggregate job counts by status |
| POST | `/v1/ingest/job/{job_id}/cancel` | `ingest` | Cancel a job |
| POST | `/v1/ingest/stream/{doc_id}` | `stream` | SSE stream for one document |
| POST | `/v1/ingest/stream/jobs` | `stream` | SSE stream for multiple jobs |
| POST | `/v1/query` | `query` | Embed queries + search LanceDB |
| POST | `/v1/rerank` | `rerank` | Rerank passages via NIM |
| GET | `/v1/health` | `system` | Liveness probe + pool snapshot |
| GET | `/v1/capabilities` | `system` | Discover enabled NIM endpoints + features |
| GET | `/v1/ingest_metrics` | `metrics` | Per-file/per-page processing metrics |
| GET | `/metrics` | (prometheus) | Prometheus scrape endpoint |
