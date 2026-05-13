# retriever eval

End-to-end QA evaluation: retrieve → generate answers with an LLM → score
them (programmatic + LLM judge), tiered output. Heavier than
[recall](recall.md); only reach for this when the user wants answer quality
metrics, not just retrieval hits.

Three subcommands:

- `retriever eval run` — main entrypoint. Runs the full QA sweep.
- `retriever eval export` — dump LanceDB retrieval results into a
  `FileRetriever` JSON, useful for offline eval or sharing fixtures.
- `retriever eval build-page-index` — turn extraction Parquets into a
  page-level markdown index (enables full-page mode).

If flags look stale, re-check `retriever eval <cmd> --help`.

## When to use this

- You want **answer quality** metrics (token-F1, LLM judge score), not just
  recall@k.
- You have a QA dataset that the eval module knows how to load (e.g.
  `bo767`, custom datasets via the loader registry).
- You're comparing generators / judges / retrieval configs.

**Use a different command when:**

- You only need recall@k → [recall](recall.md).
- You want to score multiple ingest variants → put them in the sweep config
  and let `eval run` handle it, or wrap with `retriever harness`.

## Canonical invocations

Config-driven (preferred — the config is the source of truth):

```bash
retriever eval run --config configs/eval_sweep.yaml
```

Env-driven (Docker / CI mode — same knobs, env-vars instead of YAML):

```bash
RETRIEVAL_FILE=data/retrieval.json \
QA_DATASET=bo767 \
GEN_MODEL='nvidia_nim/nvidia/llama-3.3-nemotron-super-49b-v1.5' \
JUDGE_MODEL='nvidia_nim/mistralai/mixtral-8x22b-instruct-v0.1' \
QA_TOP_K=5 \
retriever eval run --from-env
```

(Or use `LANCEDB_URI=lancedb LANCEDB_TABLE=nv-ingest` instead of
`RETRIEVAL_FILE` to retrieve directly from LanceDB.)

Export LanceDB retrievals to a JSON fixture:

```bash
retriever eval export \
  --lancedb-uri lancedb \
  --lancedb-table nv-ingest \
  --query-csv data/queries.csv \
  --output data/retrieval.json \
  --top-k 5
```

Build a page-level markdown index from extraction Parquets:

```bash
retriever eval build-page-index \
  --parquet-dir ./extracted_parquet \
  --output ./page_index.json
```

## Inputs

### `run`

Either:

- **`--config PATH`** — YAML/JSON sweep config (see structure below), or
- **`--from-env`** — read knobs from env-vars (`RETRIEVAL_FILE` *xor*
  `LANCEDB_URI`, `QA_DATASET`, `GEN_MODEL`, `JUDGE_MODEL`, `QA_TOP_K`,
  `QA_MAX_WORKERS`, `QA_LIMIT`, `MIN_COVERAGE`, `RESULTS_DIR`,
  `GEN_MODELS=name:model,name:model,…`, etc.).

Exactly one of `--config` / `--from-env` is required; they're mutually
exclusive.

### `export`

- **`--query-csv PATH`** — CSV with at least `query` (optionally `answer`).
- **`--output PATH`** — JSON path to write.
- **`--lancedb-uri` / `--lancedb-table`** — source table.
- **`--top-k`** (default 5) — chunks per query.
- **`--embedder`** (default `nvidia/llama-nemotron-embed-1b-v2`).
- **`--page-index PATH`** — optional; enables full-page mode.

### `build-page-index`

- **`--parquet-dir PATH`** — directory of `*.parquet` extraction outputs
  (typically written by `retriever pipeline run --save-intermediate`).
- **`--output PATH`** — JSON to write.

## Config structure (sweep YAML)

```yaml
execution:
  runs: 1
  top_k: 5
  max_workers: 4
  limit: 0        # 0 = no cap
  min_coverage: 0.0
dataset:
  source: bo767                       # name registered in ground_truth.py
  ground_truth_dir: data
retrieval:
  type: lancedb                       # or "file"
  lancedb_uri: lancedb
  lancedb_table: nv-ingest
  embedder: nvidia/llama-nemotron-embed-1b-v2
  save_path: null                      # optional fixture write-out
  # type: file alternative:
  # file_path: data/retrieval.json
models:
  generator:
    model: nvidia_nim/nvidia/llama-3.3-nemotron-super-49b-v1.5
    api_base: null
    api_key: ${NVIDIA_API_KEY}
    temperature: 0.0
  _judge:
    model: nvidia_nim/mistralai/mixtral-8x22b-instruct-v0.1
    api_base: null
    api_key: ${NVIDIA_API_KEY}
evaluations:
  - generator: generator
    judge: _judge
    runs: 1
output:
  results_dir: data/test_retrieval
```

## Outputs

`run` writes one result JSON per `evaluations[]` entry into
`output.results_dir` and prints a multi-tier summary:

- **Tier 1 — Retrieval quality**: answer-in-context rate (correct page is
  among top-k).
- **Tier 2 — Programmatic answer quality**: mean token-F1 per generator.
- **Tier 3 — LLM judge**: mean score (0–5) per generator, latency,
  score distribution, error counts.
- **Failure breakdown**: bucketed per generator.

Exits non-zero if any sweep run fails.

## Key flags / env vars

| Flag / env | Default | Notes |
|---|---|---|
| `--config` | — | YAML/JSON sweep file. |
| `--from-env` | off | Use env vars instead. |
| `QA_DATASET` | required | Name in `ground_truth.py` registry. |
| `RETRIEVAL_FILE` *or* `LANCEDB_URI` | one required | File-mode vs lancedb-mode. |
| `LANCEDB_TABLE` | `nv-ingest` | When using `LANCEDB_URI`. |
| `EMBEDDER` | `nvidia/llama-nemotron-embed-1b-v2` | Query-time. |
| `QA_TOP_K` | `5` | Top-k passed to retriever. |
| `QA_MAX_WORKERS` | `4` | Concurrent LLM calls. |
| `QA_LIMIT` | `0` | Cap number of QA pairs (`0` = no cap). |
| `MIN_COVERAGE` | `0.0` | Abort if retrieval covers <X% of queries. |
| `GEN_MODELS` | — | `name:model,name:model,…` for multi-generator sweeps. |
| `JUDGE_MODEL` | `nvidia_nim/mistralai/mixtral-8x22b-instruct-v0.1` | LLM judge. |
| `LITELLM_DEBUG=1` | off | Verbose LiteLLM request/response logging. |

## Common failure modes

- **`ERROR: set RETRIEVAL_FILE … or LANCEDB_URI … with --from-env`** —
  required env var missing; pick one mode.
- **`ERROR: --config and --from-env are mutually exclusive`** — pass only one.
- **`Coverage: 23%` then abort** — `MIN_COVERAGE` is set higher than your
  retrieval actually achieves; either fix retrieval or lower the threshold.
- **`No module named 'litellm'`** — install the `[llm]` extra; eval requires
  it. The CLI defers the import to inside the command body, so
  `retriever --help` / other subcommands still work without it.
- **Judge gives mean_score ≈ 0 with high `error_count`** — wrong
  `JUDGE_API_KEY` / `JUDGE_API_BASE`, or the judge model rejected the
  prompt. Inspect `--- Judge errors ---` section in the run summary.

## Related

- [recall](recall.md) — recall@k only, much faster, no LLMs.
- [pipeline](pipeline.md) — can invoke this directly as the post-ingest
  step via `--evaluation-mode qa --eval-config <file>`.
- [query](query.md) — single ad-hoc retrieval.
