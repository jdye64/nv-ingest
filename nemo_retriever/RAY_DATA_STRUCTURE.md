# Ray Data Structure Guide (Batch Pipeline)

This document explains what a single Ray Dataset row looks like after each actor stage in the `nemo_retriever` batch pipeline.

Scope:
- Pipeline: `BatchIngestor` in `nemo_retriever/src/nemo_retriever/ingest_modes/batch.py`
- Focus: PDF-style pipeline (`.extract().embed().vdb_upload()`)
- Data model: Ray Data with `batch_format="pandas"` (rows are pandas row records)

## Why this exists

When developing a single actor, it is easy to assume the wrong input shape for that stage. This guide provides:
- Expected columns entering and leaving each actor
- Key nested fields used by downstream actors
- Common gotchas that break downstream compatibility

## Pipeline Order (PDF Path)

1. `read_binary_files` (Ray source)
2. `DocToPdfConversionActor`
3. `PDFSplitActor`
4. `PDFExtractionActor`
5. `PageElementDetectionActor` (or `NemotronParseActor` in parse-only mode)
6. `OCRActor` (normal mode)
7. `explode_content_to_rows` (inside `.embed()`)
8. `_BatchEmbedActor` / `embed_text_main_text_embed`
9. `_LanceDBWriteActor` (side-effect write; shape mostly unchanged)

## Stage-by-Stage Row Shapes

### 0) Ray Source: `read_binary_files`

Typical columns:
- `bytes`: original file bytes
- `path`: source file path

Notes:
- One row per file.

---

### 1) `DocToPdfConversionActor`

Input columns:
- `bytes`, `path`

Output columns:
- `bytes`, `path`

Behavior:
- `.pdf` passes through.
- `.docx` and `.pptx` are converted to PDF bytes and stored back in `bytes`.
- Unsupported extensions pass through.
- On conversion failure, error metadata may be attached.

---

### 2) `PDFSplitActor`

Input columns:
- `bytes`, `path`

Output columns:
- `bytes`: single-page PDF bytes
- `path`
- `page_number`: 1-indexed page
- `metadata`: includes at least `source_path`

Behavior:
- Explodes one file row into many rows (one row per page).

---

### 3) `PDFExtractionActor`

Input columns:
- `bytes`, `path`, `page_number`, `metadata`

Output columns:
- `path`
- `page_number`
- `text`
- `page_image`
- `images`
- `tables`
- `charts`
- `infographics`
- `metadata`

Important nested fields:
- `page_image` usually contains:
  - `image_b64`
  - `encoding`
  - `orig_shape_hw`
- `metadata` usually contains:
  - `has_text`
  - `needs_ocr_for_text`
  - `dpi`
  - `source_path`
  - `error` (or `None`)

Notes:
- This stage is where rows become page-centric semantic records.

---

### 4A) `PageElementDetectionActor` (normal mode)

Input columns:
- Page-centric columns from extraction stage

Added columns:
- `page_elements_v3`: dict payload containing `detections`, optional `timing`, optional `error`
- `page_elements_v3_num_detections`: int
- `page_elements_v3_counts_by_label`: dict of label -> count

Notes:
- Keeps all existing columns unchanged.
- `OCRActor` depends on `page_elements_v3.detections`.

---

### 4B) `NemotronParseActor` (parse-only mode)

Input columns:
- Page-centric columns from extraction stage

Added/updated columns:
- `table`: list of extracted table entries
- `chart`: list of extracted chart entries
- `infographic`: list of extracted infographic entries
- `table_parse`, `chart_parse`, `infographic_parse`: aliases
- `nemotron_parse_v1_2`: stage timing/error payload

Notes:
- In parse-only mode, this stage replaces the `PageElementDetectionActor + OCRActor` pair.

---

### 5) `OCRActor` (normal mode only)

Input columns:
- Includes `page_image` and `page_elements_v3`

Added/updated columns:
- `table`: list of objects, usually with `bbox_xyxy_norm`, `text`
- `chart`: list of objects, usually with `bbox_xyxy_norm`, `text`
- `infographic`: list of objects, usually with `bbox_xyxy_norm`, `text`
- `ocr_v1`: stage timing/error payload

Notes:
- Uses page elements to crop page image regions and run OCR.
- Keeps `text` from extraction stage intact.

---

### 6) `explode_content_to_rows` (inside `.embed()`)

Input columns:
- Page record with `text` and optional structured lists (`table`, `chart`, `infographic`)

Output shape:
- One input row may become multiple rows:
  - one row for page text
  - one row per structured item text

Added helper columns:
- `_embed_modality`
- `_image_b64` (for image or text-image modalities)

Behavior notes:
- Structured rows overwrite `text` with the element text.
- If a structured bbox exists, `_image_b64` may be cropped from page image.

---

### 7) `_BatchEmbedActor` via `embed_text_main_text_embed`

Input columns:
- Exploded rows from previous stage

Added/updated columns:
- `metadata.embedding`: vector (primary location used downstream)
- `text_embeddings_1b_v2`: payload dict
- `text_embeddings_1b_v2_dim`: int
- `text_embeddings_1b_v2_has_embedding`: bool
- `_contains_embeddings`: bool

Removed helper columns:
- `_embed_modality`
- `_image_b64`

Notes:
- Local or remote embedding path both converge to the same row shape.

---

### 8) `_LanceDBWriteActor`

Input columns:
- Embedded rows

Output columns:
- Same as input (writes are side-effect based)

Behavior:
- Streams embeddings to LanceDB row-by-row.
- Returns the batch unchanged so pipeline can continue.

## Minimal Shape Examples

These are intentionally simplified and omit many optional fields.

### After `PDFSplitActor`

```python
{
  "bytes": b"%PDF-1.7 ... single page ...",
  "path": "/data/file.pdf",
  "page_number": 3,
  "metadata": {"source_path": "/data/file.pdf"}
}
```

### After `PDFExtractionActor`

```python
{
  "path": "/data/file.pdf",
  "page_number": 3,
  "text": "Extracted page text ...",
  "page_image": {"image_b64": "...", "encoding": "png", "orig_shape_hw": (2200, 1700)},
  "images": [],
  "tables": [],
  "charts": [],
  "infographics": [],
  "metadata": {
    "has_text": True,
    "needs_ocr_for_text": False,
    "dpi": 200,
    "source_path": "/data/file.pdf",
    "error": None
  }
}
```

### After `OCRActor`

```python
{
  "...": "...",
  "table": [{"bbox_xyxy_norm": [0.1, 0.2, 0.8, 0.5], "text": "| A | B |"}],
  "chart": [],
  "infographic": [],
  "ocr_v1": {"timing": {"seconds": 0.4}, "error": None}
}
```

### After embedding

```python
{
  "...": "...",
  "metadata": {"source_path": "/data/file.pdf", "embedding": [0.01, -0.02, "..."]},
  "text_embeddings_1b_v2": {"embedding": [0.01, -0.02, "..."], "info_msg": None},
  "text_embeddings_1b_v2_dim": 2048,
  "text_embeddings_1b_v2_has_embedding": True,
  "_contains_embeddings": True
}
```

## Development Guidelines

- Treat column additions as API contracts for downstream stages.
- Avoid renaming core columns (`text`, `metadata`, `page_image`, `table`, `chart`, `infographic`) without updating all consumers.
- If adding nested metadata fields, keep old keys intact when possible to avoid downstream breakage.
- Fail-soft behavior is preferred: attach stage-level error payloads instead of raising when possible.
- For new actors, document input and output row shape in this file as part of the change.

## Related Code

- `nemo_retriever/src/nemo_retriever/ingest_modes/batch.py`
- `nemo_retriever/src/nemo_retriever/utils/convert/to_pdf.py`
- `nemo_retriever/src/nemo_retriever/pdf/split.py`
- `nemo_retriever/src/nemo_retriever/pdf/extract.py`
- `nemo_retriever/src/nemo_retriever/page_elements/page_elements.py`
- `nemo_retriever/src/nemo_retriever/ocr/ocr.py`
- `nemo_retriever/src/nemo_retriever/ingest_modes/inprocess.py` (shared explode/embed helpers used by batch mode)
