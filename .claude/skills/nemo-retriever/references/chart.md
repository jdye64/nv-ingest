# retriever chart

Single-stage chart enrichment — take a primitives DataFrame (from
`retriever pdf` / `retriever pipeline run --save-intermediate`) and run the
chart extractor over chart-typed primitives.

Two subcommands:

- `retriever chart stage run` — load a primitives file, run chart
  enrichment, write the enriched DataFrame. This is the working command.
- `retriever chart stage graphic-elements` — not implemented yet
  (placeholder; prints "graphic-elements command is not implemented yet.").

If flags look stale, re-check `retriever chart stage <cmd> --help`.

## When to use this

- You ran PDF extraction (with `--extract-charts`) and want the chart
  primitives enriched with chart-derived data (axes, series, etc.) without
  re-running the whole pipeline.

**Use a different command when:**

- You want the full chain → [ingest](ingest.md) / [pipeline](pipeline.md).
- You haven't extracted yet → [pdf](pdf.md).

## Canonical invocations

Run on a Parquet primitives file, write `<input>.chart.parquet`:

```bash
retriever chart stage run --input extraction.parquet
```

Custom output path + YAML config:

```bash
retriever chart stage run \
  --input extraction.parquet \
  --output extraction.enriched.parquet \
  --config ingest-config.yaml
```

JSONL input/output also works (the format is autodetected from the
extension):

```bash
retriever chart stage run --input primitives.jsonl --output enriched.jsonl
```

## Inputs

- **`--input PATH`** — `.parquet`, `.jsonl`, or `.json` primitives file with
  a `metadata` column.
- **`--config PATH`** — optional ingest YAML; uses the `chart` section.
  Autodiscovers `./ingest-config.yaml` then `$HOME/.ingest-config.yaml`.

## Key flags

| Flag | Default | Notes |
|---|---|---|
| `--input` | required | Primitives file (parquet/jsonl/json). |
| `--output` | `<input>.chart<ext>` | Enriched output path. |
| `--config` | autodiscover | YAML config; `chart` section. |

## Outputs

- One enriched DataFrame in the same format as the input. Chart-typed rows
  get additional fields populated (chart data extracted by the configured
  chart extractor / NIM).
- Stdout: `Done wrote=<path> rows=<n>`.

## Common failure modes

- **`Failed reading file: <input>`** — input format isn't one of the
  supported extensions, or the file is missing a `metadata` column.
- **`graphic-elements command is not implemented yet`** — the
  `graphic-elements` subcommand is a placeholder; use `stage run` instead.
- **Chart extractor NIM unreachable** — when the `chart` section of the
  ingest YAML points at a remote NIM, that endpoint must be live. Check
  the YAML's `chart.endpoints.*` values.

## Related

- [pdf](pdf.md) — produces the primitives this command reads.
- [pipeline](pipeline.md) — runs PDF extraction + chart enrichment +
  everything else in one shot.
- See `ingest-config.yaml` `chart:` section for endpoint / method
  configuration.
