# retriever pipeline

End-to-end graph-based ingestion pipeline with full per-stage knobs:
extract → embed → optional dedup/caption/store → VDB upload → optional
recall/BEIR/QA eval. Heavier-weight than `retriever ingest`; use this when
the user needs *any* of:

- non-PDF input (`txt`, `html`, `image`, `audio`, `video`, `doc`)
- remote NIM endpoints (page-elements, OCR, embedding, reranker)
- per-stage tuning (batch sizes, actor counts, GPU fractions)
- post-ingest BEIR / QA evaluation
- saving extraction Parquets (for `eval build-page-index`)
- text chunking, dedup, captioning, image storage
- Ray Data ("batch") or service-mode execution

One subcommand: `retriever pipeline run <INPUT_PATH>`.

If flags look stale, re-check `retriever pipeline run --help` — there are
**~80 flags** grouped into Rich panels (I/O, Extraction, NIM endpoints,
Embedding, Dedup/Caption, Audio, Video, Ray, VDB, Evaluation, Service,
Observability).

## When to use this

- The user explicitly says "pipeline" or names a knob (DPI, dedup, caption,
  store, OCR endpoint, embedder URL, Ray, BEIR, …).
- Input type is anything other than PDF (`--input-type audio|video|image|…`).
- The user wants the run to also produce evaluation metrics in the same
  invocation.

**Use a different command when:**

- Just want PDFs → LanceDB with defaults → [ingest](ingest.md).
- Pre-computed embeddings → LanceDB → [vector-store](vector-store.md).
- Long-running web server → [service](service.md).

## Canonical invocations

### Default — PDFs, batch (Ray Data), local everything

```bash
retriever pipeline run data/pdfs/
```

This runs extract → embed → LanceDB upload → recall (only if input-type=audio
with `--recall-match-mode=audio_segment`; otherwise eval is skipped silently
unless the query CSV exists). For PDFs you usually want one of the modes below.

### Local in-process mode (no Ray) — good for debugging

```bash
retriever pipeline run data/pdfs/ \
  --run-mode inprocess \
  --no-vdb         # extract+embed only, no LanceDB write
```

### Remote NIMs — talk to a NIM stack for page-elements / OCR / embed

```bash
retriever pipeline run data/pdfs/ \
  --run-mode batch \
  --page-elements-invoke-url http://localhost:8001/v1/infer \
  --ocr-invoke-url           http://localhost:8003/v1/infer \
  --embed-invoke-url         http://localhost:8012/v1 \
  --api-key "$NVIDIA_API_KEY"
```

When a `*-invoke-url` is set the corresponding GPU fraction is forced to
`0.0` automatically (you'll see a warning).

### Service mode — delegate the whole pipeline to a running retriever service

```bash
retriever pipeline run data/pdfs/ \
  --run-mode service \
  --service-url http://localhost:7670 \
  --service-concurrency 8
```

Server-side LanceDB writes happen inside the service; client-side VDB
upload is skipped.

### Audio with recall eval

```bash
retriever pipeline run data/audio/ \
  --input-type audio \
  --segment-audio --audio-split-type size --audio-split-interval 500000 \
  --evaluation-mode recall \
  --query-csv data/audio_query_gt.csv \
  --recall-match-mode audio_segment
```

### BEIR evaluation after PDF ingest

```bash
retriever pipeline run data/pdfs/ \
  --evaluation-mode beir \
  --beir-loader hf \
  --beir-dataset-name BeIR/fiqa \
  --beir-split test \
  --beir-k 1 --beir-k 5 --beir-k 10
```

### QA sweep after ingest

```bash
retriever pipeline run data/pdfs/ \
  --evaluation-mode qa \
  --eval-config configs/eval_sweep.yaml
```

### Save extraction Parquet for downstream tools

```bash
retriever pipeline run data/pdfs/ \
  --save-intermediate ./extracted_parquet
```

(Then `retriever eval build-page-index --parquet-dir ./extracted_parquet
--output page_index.json` produces a full-page index.)

### Sidecar metadata join (same triplet as `nv-ingest-client`)

```bash
retriever pipeline run data/pdfs/ \
  --meta-dataframe ./meta.csv \
  --meta-source-field source \
  --meta-fields meta_a,meta_b \
  --meta-join-key auto
```

## Inputs

- **Positional `INPUT_PATH`** — file or directory of documents.
- **`--input-type`** — `pdf` (default), `doc`, `txt`, `html`, `image`,
  `audio`, `video`. Required to pick the right extractor.

## Key flags (by panel)

The CLI organises its flags into Rich `--help` panels. The most common ones:

### I/O and execution
| Flag | Default | Notes |
|---|---|---|
| `--run-mode` | `batch` | `batch` (Ray) / `inprocess` / `service`. |
| `--input-type` | `pdf` | See list above. |
| `--debug` / `--no-debug` | off | Debug-level logging. |
| `--log-file PATH` | — | Tee stdout/stderr to a file. |

### PDF extraction
| Flag | Default | Notes |
|---|---|---|
| `--method` | `pdfium` | `pdfium`, `pdfium_hybrid`, `ocr`, `nemotron_parse`, `tika`. |
| `--dpi` | `300` | Page rendering DPI. |
| `--extract-text/-tables/-charts/-infographics/-page-as-image` | varies | Toggle individual primitive types. |

### Remote NIM endpoints
| Flag | Default | Notes |
|---|---|---|
| `--api-key` | `$NVIDIA_API_KEY` | Bearer token for all NIMs. |
| `--page-elements-invoke-url` | local | YOLOX HTTP URL. |
| `--ocr-invoke-url` | local | OCR HTTP URL. |
| `--ocr-version` | `v2` | `v2` multilingual (default) or `v1` legacy. |
| `--graphic-elements-invoke-url` | local | Graphic elements NIM. |
| `--table-structure-invoke-url` | local | Table structure NIM. |
| `--embed-invoke-url` | local | OpenAI-compat embedder NIM. |

### Embedding
| Flag | Default | Notes |
|---|---|---|
| `--embed-model-name` | VL embed model | Override the embedder. |
| `--embed-modality` | `text` | `text` / `image` / `text+image`. |
| `--local-ingest-embed-backend` | `vllm` | `vllm` or `hf` when `--embed-invoke-url` is unset. VL models always use hf. |

### Storage / chunking
| Flag | Default | Notes |
|---|---|---|
| `--store-images-uri` | — | If set, write image blobs there. |
| `--text-chunk` / `--text-chunk-max-tokens` / `--text-chunk-overlap-tokens` | off | Tokenizer-based chunking. |

### VDB
| Flag | Default | Notes |
|---|---|---|
| `--vdb-op` | `lancedb` | `nv-ingest-client` VDB operator key. |
| `--vdb-kwargs-json` | — | JSON object of constructor kwargs. |
| `--no-vdb` | off | Skip in-graph upload (extract+embed only). |
| `--save-intermediate DIR` | — | Write `extraction.parquet`. |
| `--meta-dataframe / --meta-source-field / --meta-fields / --meta-join-key` | — | Sidecar metadata join. All three of dataframe/source/fields must be set or all omitted. |

### Evaluation
| Flag | Default | Notes |
|---|---|---|
| `--evaluation-mode` | `recall` | `none`, `recall`, `beir`, `qa`. Recall only runs with `--input-type=audio`. |
| `--query-csv` | `./data/bo767_query_gt.csv` | For recall mode. |
| `--reranker / --no-reranker` | off | Apply reranker NIM during eval. |
| `--beir-loader / --beir-dataset-name / --beir-split / --beir-k …` | — | BEIR-specific. |
| `--eval-config PATH` | — | Required when `--evaluation-mode=qa`. |

### Ray / batch tuning
Per-stage actor count + batch size + CPU/GPU fractions for `pdf-split`,
`pdf-extract`, `ocr`, `page-elements`, `embed`, `store`, `nemotron-parse`.
Defaults of `0` / `None` mean "use built-in heuristics".

### Service mode
| Flag | Default | Notes |
|---|---|---|
| `--service-url` | `http://localhost:7670` | Used only when `--run-mode=service`. |
| `--service-concurrency` | `8` | Max concurrent uploads. |
| `--service-api-token` | `$NEMO_RETRIEVER_API_TOKEN` | Bearer token. |

### Observability
| Flag | Default | Notes |
|---|---|---|
| `--runtime-metrics-dir DIR` | — | Write `*.runtime.summary.json`. |
| `--runtime-metrics-prefix STR` | `run` | Filename prefix for the above. |
| `--detection-summary-file PATH` | — | Per-primitive detection counts. |

## Outputs

- LanceDB table at `<--vdb-kwargs-json[uri]>/<table>.lance` (default
  `lancedb/nv-ingest.lance`) — unless `--no-vdb`.
- Extraction Parquet at `--save-intermediate/extraction.parquet` — if set.
- Detection summary JSON — if `--detection-summary-file` set.
- Runtime metrics JSON at `<--runtime-metrics-dir>/<prefix>.runtime.summary.json`
  — if set. Useful for sweep aggregation.
- Stdout: per-stage timings, recall/BEIR/QA metrics if eval ran.

## Common failure modes

- **`--evaluation-mode=recall is only supported with --input-type=audio`** —
  for PDF/non-audio recall, use [recall](recall.md) standalone.
- **`Sidecar metadata: pass all of … or omit all`** — `--meta-dataframe`,
  `--meta-source-field`, `--meta-fields` are an all-or-nothing triplet.
- **`No files found for input_type=…`** — wrong `--input-type` for the
  directory contents, or the glob didn't match. Check file extensions.
- **`--vdb-kwargs-json must be valid JSON`** — escape your shell properly:
  `--vdb-kwargs-json '{"uri":"./lancedb","table_name":"my-corpus"}'`.
- **Warnings about "Forcing X GPUs to 0.0 because --X-invoke-url is set"** —
  informational; remote NIMs replace the local model so GPU fraction is
  zeroed automatically.
- **Service mode "VDB writes are handled server-side"** — informational;
  with `--run-mode=service` the client doesn't write to LanceDB, the
  server does.

## Related

- [ingest](ingest.md) — convenience wrapper around the common PDF path.
- [eval](eval.md) — invoked when `--evaluation-mode=qa`.
- [recall](recall.md) — standalone recall@k against an existing table.
- [service](service.md) — start the server that `--run-mode=service` talks to.
- [vector-store](vector-store.md) — upload pre-computed embeddings.
