# retriever benchmark

Actor-stage throughput benchmarks. Each sub-app measures rows/sec for one
Ray actor: PDF split, PDF extract, OCR, page-elements, audio-extract, or
all of them in sequence.

Subcommands (each exposes `run`):

- `retriever benchmark split run` — `PDFSplitActor` rows/sec.
- `retriever benchmark extract run` — `PDFExtractionActor` rows/sec.
- `retriever benchmark ocr run` — `OCRActor` rows/sec.
- `retriever benchmark page-elements run` — `PageElementDetectionActor`
  rows/sec.
- `retriever benchmark audio-extract run` — `MediaChunkActor + ASRActor`
  rows/sec.
- `retriever benchmark all run` — runs every actor benchmark above in
  sequence and reports each.

If flags look stale, re-check `retriever benchmark <stage> run --help`.

## When to use this

- You changed an actor's batch size / GPU fraction and want to confirm the
  throughput impact in isolation, decoupled from the full pipeline.
- You're profiling which stage is the bottleneck on a new machine / GPU.
- You're comparing v1 vs v2 of an OCR / page-elements engine.

**Use a different command when:**

- You want end-to-end pipeline timing → [pipeline](pipeline.md) with
  `--runtime-metrics-dir`.
- You want answer quality, not throughput → [eval](eval.md).

## Canonical invocations

Run all stages on a default input set:

```bash
retriever benchmark all run
```

Just OCR with explicit GPU + batch:

```bash
retriever benchmark ocr run \
  --batch-size 32 \
  --gpus-per-actor 1.0
```

(Each `run` subcommand has its own flag set; use `--help` to see them.
Most take `--input-dir`, `--batch-size`, `--workers`, `--gpus-per-actor`,
and an output JSON path.)

## Outputs

- Per-stage rows/sec and total wall time to stdout.
- Many subcommands also write a JSON summary file for later aggregation;
  the exact path is per-subcommand. Check `--help`.

## Common failure modes

- **CUDA OOM on `--gpus-per-actor 1.0`** — drop to `0.5` or `0.25` (or
  point at a remote NIM via the corresponding `*-invoke-url` flag —
  benchmarks support remote NIMs the same way `pipeline run` does).
- **`No files found in --input-dir`** — wrong path or wrong file
  extension for that benchmark (e.g. PDFs for `extract`, audio for
  `audio-extract`).
- **`all` reports per-stage failures but doesn't abort** — by design, so
  one bad stage doesn't lose the data from the others. Inspect the
  per-stage stdout sections.

## Related

- [pipeline](pipeline.md) — runs all these actors together with realistic
  batching; the place to optimise after a benchmark identifies the
  bottleneck.
- `retriever harness run` — repeatable, history-tracked benchmark
  orchestration on top of these single-stage benchmarks.
