# Corpus suite POC

Self-contained proof of concept for **page-scoped ingestion**, a **small DAG of steps per page**, **local metadata + blob storage**, **lexical search (SQLite FTS5)**, and **HTTP APIs** suited to agents (upload, status, structured retrieval with citations).

This lives alongside the NeMo Retriever library and does not depend on Ray, Milvus, or remote services.

## Layout

| Path | Role |
|------|------|
| `src/corpus_suite_poc/ingest/` | Validate uploads, fingerprint blobs, register documents |
| `src/corpus_suite_poc/pipeline/` | Per-page DAG definitions and in-process runner |
| `src/corpus_suite_poc/store/` | SQLite catalog + content-addressed files on disk |
| `src/corpus_suite_poc/index/` | FTS index maintenance for chunk text |
| `src/corpus_suite_poc/query/` | Search facade (lexical now; vector hook reserved) |
| `src/corpus_suite_poc/agent_api/` | FastAPI application and route modules |
| `src/corpus_suite_poc/ops/` | Health and lightweight counters |

## Quickstart

From `corpus_suite_poc/`:

```bash
uv venv && uv pip install -e .
uv run corpus-suite serve --host 127.0.0.1 --port 8765
```

Ingest a PDF (or `.txt` treated as a single page):

```bash
curl -s -F "file=@/path/to/doc.pdf" http://127.0.0.1:8765/v1/documents | jq
curl -s -X POST "http://127.0.0.1:8765/v1/documents/{id}/process" | jq
curl -s "http://127.0.0.1:8765/v1/query?q=tables&limit=5" | jq
```

Environment:

- `CORPUS_DATA_DIR` — root for SQLite and `blobs/` (default: `./.corpus_data`).

## Notes

- Processing runs **in-process** with a concurrency limit for low-latency experiments without a separate queue service.
- **Hybrid** retrieval currently uses **FTS BM25-style ranking** via FTS5 `bm25()`; a dense vector stage can be added behind the same query interface.
