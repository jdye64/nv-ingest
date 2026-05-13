# retriever pdf

Single-stage PDF extraction — no embedding, no LanceDB. Runs the PDF
extractor (pdfium, pdfium_hybrid, OCR, Nemotron Parse, or Tika) over a
directory of PDFs and writes per-document JSON sidecars.

One subcommand: `retriever pdf stage page-elements`. (Currently the only
exposed PDF stage; future stages may be added under `stage`.)

If flags look stale, re-check `retriever pdf stage page-elements --help`.

## When to use this

- You need raw extracted primitives (text, table, chart, image regions)
  for inspection / debugging, **without** the embedding/VDB step.
- You're iterating on extraction `--method` / DPI / NIM endpoint settings.
- You want the `<pdf>.pdf_extraction.json` sidecars to feed
  `retriever chart` / `retriever vector-store` / your own tools.

**Use a different command when:**

- You want the full chain → [ingest](ingest.md).
- You want per-stage tuning + downstream operators → [pipeline](pipeline.md).

## Canonical invocations

PDFium (pure-Python, no NIMs):

```bash
retriever pdf stage page-elements \
  --input-dir data/pdfs/ \
  --method pdfium
```

PDFium-hybrid with YOLOX page-elements NIM:

```bash
retriever pdf stage page-elements \
  --input-dir data/pdfs/ \
  --method pdfium_hybrid \
  --yolox-http-endpoint http://page-elements:8000/v1/infer \
  --auth-token "$NVIDIA_API_KEY" \
  --extract-text --extract-tables --extract-charts
```

Nemotron Parse (LLM-style structured extraction):

```bash
retriever pdf stage page-elements \
  --input-dir data/pdfs/ \
  --method nemotron_parse \
  --nemotron-parse-http-endpoint http://nemotron-parse:8000/v1 \
  --nemotron-parse-model-name nvidia/nemotron-parse-v1
```

YAML-driven (CLI flags override YAML):

```bash
retriever pdf stage page-elements --config ingest-config.yaml
```

## Inputs

- **`--input-dir DIR`** — recursively scanned for `*.pdf`. Required (unless
  set via YAML).
- **`--config PATH`** — optional ingest YAML; auto-discovers
  `./ingest-config.yaml` then `$HOME/.ingest-config.yaml`.

## Key flags

| Flag | Default | Notes |
|---|---|---|
| `--input-dir` | (or YAML) | PDF source directory. |
| `--config` | autodiscover | YAML config; CLI flags override. |
| `--method` | `pdfium` | `pdfium`, `pdfium_hybrid`, `ocr`, `nemotron_parse`, `tika`. |
| `--auth-token` | `$NVIDIA_API_KEY` | Bearer token for NIMs. |
| `--yolox-grpc-endpoint` / `--yolox-http-endpoint` | — | Required for `pdfium_hybrid`. |
| `--nemotron-parse-grpc-endpoint` / `--nemotron-parse-http-endpoint` | — | Required for `nemotron_parse`. |
| `--nemotron-parse-model-name` | schema default | Override Nemotron Parse model. |
| `--extract-text/-tables/-charts/-infographics/-page-as-image` | text on, rest off | Toggle primitive types. |
| `--text-depth` | `page` | `page` or `document`. |
| `--render-mode` | `fit_to_model` | `fit_to_model` (~93 DPI) or `full_dpi`. |
| `--write-json-outputs / --no-write-json-outputs` | on | Sidecar JSON per PDF. |
| `--json-output-dir` | next to input | Override sidecar output dir. |
| `--limit` | `None` | Cap number of PDFs (debugging). |

## Outputs

- One `<pdf>.pdf_extraction.json` per input PDF (alongside the source PDF
  by default, or under `--json-output-dir`).
- Each sidecar contains the extracted DataFrame as JSON with one row per
  primitive: `text`, `metadata.content_metadata.{page_number, type, …}`,
  `metadata.source_metadata.{source_id, source_name}`,
  `metadata.image_metadata.bbox` where applicable.

## Common failure modes

- **`--yolox-*-endpoint` required for method 'pdfium' family** — pure
  pdfium needs no NIM, but `pdfium_hybrid` requires YOLOX for layout
  detection. Pick `--method pdfium` if you want pure-Python, or supply the
  endpoint.
- **Auth 401 on NIM endpoints** — `--auth-token` not set; either pass it
  or export `NVIDIA_API_KEY`.
- **JSON sidecars not written** — file system write failed; the error is
  logged but does not fail the run (best-effort write). Use
  `--json-output-dir` to point at a writable directory.

## Related

- [ingest](ingest.md) / [pipeline](pipeline.md) — full chain that ends in
  LanceDB.
- [chart](chart.md) — runs after PDF extraction to enrich chart primitives.
- [vector-store](vector-store.md) — upload embeddings produced from
  these sidecars.
