# Retriever Service Architecture

FastAPI application for gateway, standalone, realtime, and batch deployment roles.

## API Endpoint Summary

| Method | Path | Description |
|--------|------|-------------|
| GET | `/v1/health` | Liveness / readiness |
| GET | `/v1/dashboard` | Gateway dashboard SPA |
| GET | `/v1/dashboard/api/overview` | Cluster overview JSON |
| GET | `/v1/dashboard/api/config` | Redacted `RetrieverServiceConfig` + `@configured` metadata |
| GET | `/v1/dashboard/api/jobs` | Job list (SSE snapshot stream) |
| POST | `/v1/dashboard/api/vdb/query` | Proxied VDB query |
| GET | `/v1/admin/pool_stats` | Worker pool statistics |

## Core Subsystems

### Configuration

Unified configuration lives in `nemo_retriever.config`:

- `RetrieverServiceConfig` — service sections plus `pipeline_defaults` for SDK params
- `load_config()` — YAML discovery (`retriever-config.yaml`, `retriever-service.yaml`, bundled defaults)
- `@configured` — declares config dependencies, injects pipeline defaults, registers impact metadata for docs/agents

Service startup loads YAML via `load_config()` which installs the active config in a process context (`set_config`).

### Dashboard

Gateway-only React SPA under `/v1/dashboard`. Static assets in `dashboard/static/`. The **Config** view (`#config`) shows the effective redacted configuration tree and `@configured` resolver metadata (impact tags + rationale).

### Pipeline execution

`services/pipeline_executor.py` builds stage params from NIM/local settings merged with `pipeline_defaults`. Decorated builders: `build_extract_params`, `build_embed_params`, `build_asr_params`, `build_caption_params`.

## Directory Layout

```
service/
  app.py                 # FastAPI factory
  config.py              # Re-exports nemo_retriever.config (backward compat)
  retriever-service.yaml # Bundled default YAML
  routers/
    dashboard.py         # Dashboard API + config viewer endpoint
  dashboard/static/      # SPA (views/config.jsx = configuration page)
  services/
    pipeline_executor.py # @configured param builders
```

## Design Decisions

- **Single YAML, all run modes:** `pipeline_defaults` lets inprocess/batch share the same defaults as service without duplicating params in code.
- **NIM over local:** Remote endpoint URLs always win when both NIM and in-pod HF models are configured for a stage.
- **ConfigJustification tags:** Every `@configured` site records impact axes (PERFORMANCE, ACCURACY, LATENCY, …) plus a rationale string for agent/doc consumption.

## Developer Guide

### Add a new configured resolver

1. Decorate the builder with `@configured(section=..., model=..., justification=..., rationale=...)`.
2. Accept an optional injected param (e.g. `extract: ExtractParams | None = None`).
3. Merge service/NIM overrides onto the injected defaults.
4. Regenerate docs: `python docs/scripts/generate_config_docs.py`.

### Extend the dashboard

1. Add a JSON route under `routers/dashboard.py`.
2. Add `views/<name>.jsx` and register it in `static/index.html` load order.
3. Wire navigation in `views/layout.jsx` and routing in `views/app.jsx`.
