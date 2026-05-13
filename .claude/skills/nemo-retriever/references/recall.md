# retriever recall

Batch retrieval against a labelled query CSV → top-k hits per query +
recall@1/@5/@10 metrics. Use this once an ingest has populated a LanceDB
table and you have a ground-truth CSV of `query → expected (pdf, page)`.

Two subcommands:

- `retriever recall vdb-recall run` — recommended default; prints metrics
  and (optionally) top-k hits per query in Rich tables.
- `retriever recall vdb-recall recall-with-main` — older variant; also runs
  retrieval and prints recall@k but uses a slightly different output format
  (per-query plain-text dump, gold pdf/page split out). Prefer `run` unless
  you specifically need the legacy printout.

If flags look stale, re-check `retriever recall vdb-recall <cmd> --help`.

## When to use this

- You already ingested PDFs into LanceDB and have a `query,pdf,page` (or
  `query,pdf_page`) ground-truth CSV (e.g. `data/bo767_query_gt.csv`).
- You want recall@k metrics, not end-to-end QA grading. (For QA grading,
  use [eval](eval.md).)
- You're iterating on ingest/embedding settings and want a fast
  retrieval-only scorer.

**Use a different command when:**

- You want LLM-judged answer quality → [eval](eval.md).
- You want a single ad-hoc lookup → [query](query.md).
- You want BEIR-format datasets → [pipeline](pipeline.md) with
  `--evaluation-mode beir`.

## Canonical invocations

Top-5 hits per query + recall metrics, against the default LanceDB table
(auto-discovers `bo767_query_gt.csv` or `data/bo767_query_gt.csv`):

```bash
retriever recall vdb-recall run
```

Custom table + query CSV + top-10 hits:

```bash
retriever recall vdb-recall run \
  --query-csv data/my_query_gt.csv \
  --lancedb-uri ./my-lancedb \
  --table-name my-corpus \
  --top-k 10
```

Limit to first 50 queries (fast iteration):

```bash
retriever recall vdb-recall run --limit 50 --no-print-hits
```

Use a remote NIM embedder instead of local HF:

```bash
retriever recall vdb-recall run \
  --embedding-http-endpoint http://localhost:8012/v1 \
  --embedding-api-key "$NVIDIA_API_KEY"
```

## Inputs

- **`--query-csv PATH`** (default `bo767_query_gt.csv`) — CSV with either
  `query,pdf_page` or `query,pdf,page` columns. `pdf_page` is the joined
  form (`<pdf_basename>_<page_number>`); separate `pdf` / `page` columns
  are also accepted and auto-joined internally.
- **`--lancedb-uri` / `--table-name`** — must match what `ingest` wrote to.

## Outputs

- Per-query: query text, gold (pdf, page), top-k hits (text snippet + source
  + distance), printed via Rich.
- Summary: `recall@1`, `recall@5`, `recall@10` (always computed at fixed
  k=1/5/10 regardless of `--top-k`).

## Key flags

| Flag | Default | Notes |
|---|---|---|
| `--query-csv` | `bo767_query_gt.csv` | Ground-truth CSV. |
| `--top-k` | `5` (run) / `10` (recall-with-main) | Hits printed; metrics are still @1/5/10. |
| `--limit` | `None` | Cap number of queries (debugging). |
| `--lancedb-uri` | `lancedb` | Must match ingest. |
| `--table-name` | `nv-ingest` | Must match ingest. |
| `--vector-column` | `vector` | Column holding embeddings. |
| `--embedding-http-endpoint` | `None` | HTTP NIM URL for query embedder. |
| `--embedding-grpc-endpoint` | `None` | gRPC `host:port` for query embedder. |
| `--embedding-endpoint` | `None` | Single field; auto-detects http vs gRPC. |
| `--embedding-model` | `nvidia/llama-nemotron-embed-1b-v2` | Embedder name. |
| `--local-hf-device` | autodetect | `cuda`, `cpu`, `cuda:0`, … for local HF. |
| `--local-hf-batch-size` | `64` | Local HF batch size. |
| `--local-query-embed-backend` | `hf` | `hf` or `vllm` when no remote endpoint. |
| `--print-hits / --no-print-hits` | `--print-hits` | Suppress per-query Rich dump. |

## CSV format

```
query,pdf,page
"How much revenue is expected in 2023?",1102434,3
"What types of statistics were utilized?",1096078,3
```

or:

```
query,pdf_page
"How much revenue is expected in 2023?",1102434_4
"What types of statistics were utilized?",1096078_4
```

Note: `pdf_page` uses `<pdf_basename>_<page_number_one_indexed>`. The
ground-truth CSV pages and the LanceDB `metadata.page_number` must use the
same indexing convention; the bundled `bo767_query_gt.csv` is one-indexed.

## Common failure modes

- **All recall metrics = 0** — gold page numbering vs `metadata.page_number`
  mismatch (one-indexed vs zero-indexed). Inspect a few hits with
  `--print-hits` and compare the printed `page_number` to the gold value.
- **`Query CSV not found at '…'`** — the default
  `bo767_query_gt.csv` lookup didn't find the file; pass `--query-csv`
  explicitly.
- **Slow first run** — same vLLM startup cost as `ingest`/`query`. Use
  `--limit 20` while iterating on knobs.
- **`local query embed backend must be 'hf' or 'vllm'`** — typo in
  `--local-query-embed-backend`; only those two values are accepted.

## Related

- [ingest](ingest.md) — populate the LanceDB table this command reads.
- [query](query.md) — same retrieval path, one query at a time.
- [eval](eval.md) — LLM-judged QA quality on top of retrieval.
- [pipeline](pipeline.md) — runs recall (or BEIR) automatically as a
  post-ingest step.
