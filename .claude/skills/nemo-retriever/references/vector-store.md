# retriever vector-store

Upload pre-computed text embeddings (from `retriever local stage5` or any
compatible tool) into a LanceDB table. Useful when the embedding step
already ran offline and you only need the VDB upload.

One subcommand: `retriever vector-store stage run`.

If flags look stale, re-check `retriever vector-store stage run --help`.

## When to use this

- You already have `*.text_embeddings.json` files (from `retriever local
  stage5` or equivalent) and just want them in LanceDB.
- You're rebuilding the index with different partition/sub-vector settings.
- You want to append a second corpus to an existing table.

**Use a different command when:**

- You want to ingest PDFs end-to-end → [ingest](ingest.md).
- You want every pipeline knob → [pipeline](pipeline.md).

## Canonical invocations

Upload a directory of embeddings, overwrite table, build IVF-HNSW-SQ index:

```bash
retriever vector-store stage run \
  --input-dir ./embeddings/
```

Append to existing table, recurse subdirs, custom DB path:

```bash
retriever vector-store stage run \
  --input-dir ./embeddings/ \
  --recursive \
  --append \
  --lancedb-uri ./my-lancedb \
  --table-name my-corpus
```

Skip indexing (faster bulk load; build the index later):

```bash
retriever vector-store stage run --input-dir ./embeddings/ --no-create-index
```

Tune IVF index for a small corpus:

```bash
retriever vector-store stage run \
  --input-dir ./embeddings/ \
  --num-partitions 8 \
  --num-sub-vectors 64
```

## Inputs

- **`--input-dir DIR`** — directory containing `*.text_embeddings.json` files.

Each input file is expected to follow the stage5 schema (vector column +
`metadata.content_metadata.page_number` + `pdf_basename` / `path` /
`source_id`). The bulk loader unfolds these into LanceDB rows.

## Key flags

| Flag | Default | Notes |
|---|---|---|
| `--input-dir` | required | Source directory. |
| `--recursive / --no-recursive` | `--no-recursive` | Scan subdirectories. |
| `--limit` | `None` | Optionally cap number of input files. |
| `--lancedb-uri` | `lancedb` | Destination DB. |
| `--table-name` | `nv-ingest` | Destination table. |
| `--overwrite / --append` | `--overwrite` | Replace table or append. |
| `--create-index / --no-create-index` | `--create-index` | Build vector index after upload. |
| `--index-type` | `IVF_HNSW_SQ` | LanceDB index type. |
| `--metric` | `l2` | Distance metric. |
| `--num-partitions` | `16` | IVF partition count. |
| `--num-sub-vectors` | `256` | Sub-vector count for SQ. |

## Outputs

- LanceDB table at `<--lancedb-uri>/<--table-name>.lance` with one row per
  source embedding. Each row includes `vector`, `pdf_basename`,
  `page_number`, `path`, `source_id`.
- Stdout summary: `files=… processed=… skipped=… failed=…
  lancedb_uri=… table=…`.

## Common failure modes

- **`failed=N` for all files** — schema mismatch in the JSON inputs;
  `*.text_embeddings.json` must follow the stage5 layout (vector + metadata
  fields).
- **`Clamping num_partitions from 16 to 7`** — IVF needs `num_partitions <
  row_count`; informational, not an error.
- **Index build very slow on small corpora** — pass `--no-create-index`
  during bulk loading, then build the index with LanceDB SDK once data is
  in place.

## Related

- [ingest](ingest.md) — the full chain that produces these embeddings.
- [pipeline](pipeline.md) — alternative full pipeline with `--vdb-op
  lancedb`.
- [query](query.md) — reads what this command writes.
