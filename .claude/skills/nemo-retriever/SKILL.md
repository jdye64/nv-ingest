---
name: nemo-retriever
description: Run the NeMo Retriever Library `retriever` CLI from Claude Code. Use when the user wants to ingest documents (PDF / image / audio / video / txt / html), query a LanceDB table, compute recall@k against a query CSV, run a QA evaluation sweep, drive the full extract→embed→VDB pipeline, run per-stage tools (pdf, chart, vector-store), start or call the retriever service, benchmark actors, or compare runs. Triggered by phrases like "retriever ingest", "retriever query", "retriever recall", "retriever eval", "retriever pipeline", "retriever service", "ingest these PDFs", "query my LanceDB table", "compute recall@10", "run a QA eval", or any mention of the `retriever` CLI / `nemo-retriever` library.
---

# nemo-retriever

Run the NeMo Retriever Library `retriever` CLI.

The user almost always wants one of these flows:

| The user wants to… | Use |
|---|---|
| Turn PDFs into a searchable LanceDB table | `retriever ingest` → [references/ingest.md](references/ingest.md) |
| Search an existing LanceDB table | `retriever query` → [references/query.md](references/query.md) |
| Score retrieval against a labelled CSV | `retriever recall` → [references/recall.md](references/recall.md) |
| Grade end-to-end QA quality | `retriever eval` → [references/eval.md](references/eval.md) |
| Full pipeline with knobs (audio / video / NIMs / BEIR) | `retriever pipeline` → [references/pipeline.md](references/pipeline.md) |
| Run / call the long-lived service | `retriever service` → [references/service.md](references/service.md) |
| Upload pre-computed embeddings | `retriever vector-store` → [references/vector-store.md](references/vector-store.md) |
| Single-stage PDF extraction | `retriever pdf` → [references/pdf.md](references/pdf.md) |
| Single-stage chart enrichment | `retriever chart` → [references/chart.md](references/chart.md) |
| Compare two runs | `retriever compare` → [references/compare.md](references/compare.md) |
| Throughput benchmarks for actors | `retriever benchmark` → [references/benchmark.md](references/benchmark.md) |

`retriever local`, `retriever audio`, `retriever txt`, `retriever html`, `retriever image`, `retriever harness` are also available; fall back to `retriever <subcommand> --help` for those.

## How to use this skill

1. **Pick the smallest subcommand that does the job.** If the user just wants
   "ingest these PDFs and let me search them" — `retriever ingest` then
   `retriever query`. Do **not** reach for `retriever pipeline` unless the user
   needs per-stage tuning, remote NIMs, audio/video, BEIR, or QA eval.

2. **Read the matching `references/*.md` before running anything non-trivial.**
   Each reference file documents: when to use it, canonical invocations, key
   flags, common failure modes, and links to related commands. They are kept
   short on purpose so they cost few tokens to load.

3. **Run the command.** Default form: `retriever <subcommand> [args…]`. If the
   exact flag set isn't covered in the reference file, run `retriever
   <subcommand> --help` to see the live signature — the references are
   intentionally minimal and may lag the CLI.

4. **Verify it worked.** Two helpers in `scripts/`:
   - `scripts/inspect_lancedb.py <uri> <table>` — print row count, schema, and
     a few sample rows from a LanceDB table after `ingest` / `pipeline run`.
   - `scripts/recall_smoke_test.sh <pdf> <query_csv>` — minimal end-to-end
     smoke test: ingest → recall, with recall@1/@5/@10 printed.

## Canonical happy path

```bash
retriever ingest data/multimodal_test.pdf
retriever query "what is in chart 1?" --top-k 3
python .claude/skills/nemo-retriever/scripts/inspect_lancedb.py lancedb nv-ingest
```

That covers ~80% of "use the retriever CLI" requests. Anything else (recall,
eval, pipeline, service, single-stage tools, benchmark, compare) lives in
the matching reference.

## Defaults to assume

Unless the user says otherwise, every subcommand uses:

- LanceDB URI: `lancedb` (a directory in cwd)
- Table name: `nv-ingest`
- Embedder: `nvidia/llama-nemotron-embed-1b-v2` (text) /
  `nvidia/llama-nemotron-embed-vl-1b-v2` (query, VL)
- Run mode: `inprocess` (no Ray) for `ingest` / one-shot CLI; `batch` (Ray Data)
  for `pipeline run`; `service` only when the user explicitly invokes a
  running `retriever service`.

If the user passes `--lancedb-uri` / `--table-name` to `ingest`, **carry the
same values through to `query` / `recall` / `eval`** — mismatched URI/table
is the single most common cause of empty results.

## Common failure modes (any subcommand)

- **First run is slow.** vLLM startup + HuggingFace model download. Subsequent
  in-process runs are fast; one-shot CLI invocations always pay this cost.
- **`Table 'nv-ingest' was not found`** — wrong `--table-name` / `--lancedb-uri`,
  or ingest hasn't run yet.
- **Empty result array / 0% recall** — same root cause as above, OR an empty
  table from a failed ingest. Run `scripts/inspect_lancedb.py` to confirm.
- **Missing optional dependency error at import** — the corresponding sub-app
  is registered lazily in `main.py`; install the `[gpu]` / `[llm]` /
  `[benchmark]` extra the error references (or use `retriever service` to
  offload work to a server that already has them).

## If the user gives `$ARGUMENTS`

Run `retriever $ARGUMENTS` directly. If `$ARGUMENTS` is empty, run
`retriever --help` and summarize the top-level subcommands; then ask which
one they want.
