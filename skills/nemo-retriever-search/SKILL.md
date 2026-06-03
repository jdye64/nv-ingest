---
name: nemo-retriever-search
description: "Use when searching ingested documents via the NeMo Retriever Search web app or its HTTP/MCP API. Requires a running retriever service (`retriever service start`). Not for local LanceDB-only `retriever query` workflows unless the service is up."
license: Apache-2.0
allowed-tools: Bash Write Read
---

# nemo-retriever-search

NeMo Retriever Search is a Google-minimal UI and agent-friendly API over the retriever **service** corpus.

## Prerequisites

1. Start the retriever service:

```bash
retriever service start
```

2. Start the search app:

```bash
retriever search start --service-url http://localhost:7670 --port 8200
```

3. Open `http://localhost:8200` or call the API directly.

If the service is unreachable, the search app returns HTTP 503 with: *Start the retriever service: `retriever service start`*.

## HTTP API (agent-friendly)

Base URL: `http://localhost:8200`

### Corpus status

```bash
curl -s http://localhost:8200/api/v1/status | jq .
```

### Search

```bash
curl -s -X POST http://localhost:8200/api/v1/search \
  -H 'Content-Type: application/json' \
  -d '{"query":"What is the revenue?","top_k":10}' | tee /tmp/search.json | jq '.hits[] | {rank, source, page_number, text_preview, export}'
```

Each hit includes `export.text_url`, `export.json_url`, and `export.summary_url` for downstream tools.

### Export a hit as plain text

```bash
HIT_ID=$(jq -r '.hits[0].hit_id' /tmp/search.json)
curl -s "http://localhost:8200/api/v1/hits/${HIT_ID}/export?format=text"
```

### Ingest documents (optional)

Upload via UI **+** button, or:

```bash
curl -s -X POST http://localhost:8200/api/v1/ingest \
  -F 'files=@./docs/report.pdf' | jq .
```

## MCP tools

When the search app is running, MCP is mounted at `/mcp`:

| Tool | Purpose |
|------|---------|
| `get_corpus_status` | Check service + vectordb readiness |
| `search_corpus` | Semantic search; returns agent-friendly hit table |
| `ingest_documents` | Ingest local file paths via the service |
| `export_hit` | Fetch hit text/json/summary by `hit_id` |

## Workflow for agents

1. `get_corpus_status` — confirm `total_rows > 0` (or ingest first).
2. `search_corpus` with the user's question — save `search_id` and hit `hit_id`s.
3. `export_hit` with `format=text` for the top hit when verbatim chunk text is needed.
4. Synthesize the answer citing `source` and `page_number` from the hit JSON.

## Authentication

If the retriever service uses bearer auth, pass the same token to search:

```bash
retriever search start --api-token "$NEMO_RETRIEVER_API_TOKEN"
```

Export URLs and search calls inherit the server-side token configured at launch.
