import time
from pathlib import Path
from nemo_retriever.graph_ingestor import GraphIngestor
from nemo_retriever.params import ExtractParams, EmbedParams

docs = [str(Path("/datasets/nv-ingest/bo767").resolve())]

NGC_MODELS = Path("/datasets/nv-ingest/models")

# --- HuggingFace models (no TRT engines) ---
extract = ExtractParams(
    method="pdfium",
    use_fused_vision=False,
    use_table_structure=True,
    use_graphic_elements=True,
    inference_batch_size=32,
)

embed = EmbedParams()

ing = GraphIngestor(
    run_mode="batch",
    ray_address="auto",
    node_overrides={
        "PDFExtractionActor": {"concurrency": 48, "batch_size": 16},
        "PageElementDetectionActor": {"concurrency": 4, "num_gpus": 0.1, "num_cpus": 0, "batch_size": 32},
        "TableStructureActor": {"concurrency": 3, "num_gpus": 0.1, "num_cpus": 0, "batch_size": 32},
        "GraphicElementsActor": {"concurrency": 3, "num_gpus": 0.1, "num_cpus": 0, "batch_size": 32},
        "OCRActor": {"concurrency": 3, "num_gpus": 0.1, "num_cpus": 0, "batch_size": 32},
        "_BatchEmbedActor": {"concurrency": 1, "num_gpus": 0.1, "batch_size": 256},
    },
)

# --- TRT engine config (uncomment to switch back) ---
# extract = ExtractParams(
#     method="pdfium",
#     use_fused_vision=True,
#     page_elements_trt_engine_path=str(NGC_MODELS / "page-elements/ngc/hub/models--nim--nvidia--nemotron-page-elements-v3"),
#     table_structure_trt_engine_path=str(NGC_MODELS / "table-structure/ngc/hub/models--nim--nvidia--nemotron-table-structure-v1"),
#     graphic_elements_trt_engine_path=str(NGC_MODELS / "graphic-elements/ngc/hub/models--nim--nvidia--nemotron-graphic-elements-v1"),
#     ocr_trt_engine_path=str(NGC_MODELS / "ocr/ngc/hub/models--nim--nvidia--nemoretriever-ocr-v1"),
#     use_table_structure=True,
#     use_graphic_elements=True,
#     inference_batch_size=32,
# )
#
# embed = EmbedParams(
#     embed_trt_engine_path=str(NGC_MODELS / "embedding/ngc/hub/models--nim--nvidia--llama-nemotron-embed-1b-v2"),
# )
#
# ing = GraphIngestor(
#     run_mode="batch",
#     ray_address="local",
#     node_overrides={
#         "PDFExtractionActor": {"concurrency": 48, "batch_size": 16},
#         "FusedVisionActor": {"concurrency": 6, "num_gpus": 0.10, "num_cpus": 2, "batch_size": 32},
#         "_BatchEmbedActor": {"concurrency": 2, "num_gpus": 0.10, "batch_size": 256},
#     },
# )
#ing = ing.files(docs).extract(extract).embed(embed)

# Disable embedding for a test
ing = ing.files(docs).extract(extract)
t0 = time.perf_counter()
ray_ds = ing.ingest()
raw_num_rows = ray_ds.count()
elapsed = time.perf_counter() - t0
print(f"\n{'='*60}")
print(f"Ingestion complete: {raw_num_rows} rows in {elapsed:.2f}s")
print(f"Throughput: {54730 / elapsed:.2f} rows/sec")
print(f"{'='*60}\n")

# ── Write embeddings to LanceDB ──────────────────────────────────────
import lancedb
from nemo_retriever.vector_store.lancedb_utils import (
    build_lancedb_rows,
    lancedb_schema,
    infer_vector_dim,
    create_or_append_lancedb_table,
)
from nemo_retriever.vector_store.lancedb_store import LanceDBConfig, create_lancedb_index

LANCEDB_URI = "/tmp/nemo_retriever_lancedb"
LANCEDB_TABLE = "bo767"

import json
import numpy as _np
import csv
import shutil
import time as _time

# Collect all output lines, print them in one block at the very end
# so they aren't lost in Ray worker logs.
_out: list[str] = []
def _log(msg: str = ""):
    _out.append(msg)

_diag_file = Path("/tmp/nemo_retriever_debug/trt_embed_tensors.txt")
if _diag_file.exists():
    _log("TRT Embed Engine tensor info:")
    _log(_diag_file.read_text())

_log("Materialising Ray Dataset to pandas …")
t1 = time.perf_counter()
df = ray_ds.to_pandas()
_log(f"  → {len(df)} rows in {time.perf_counter() - t1:.2f}s")

DUMP_DIR = Path("/tmp/nemo_retriever_debug")
DUMP_DIR.mkdir(parents=True, exist_ok=True)

# ── Count detections per model and add as columns ─────────────────────
def _count_pe_detections(row):
    pe = row.get("page_elements_v3")
    if isinstance(pe, dict):
        dets = pe.get("detections")
        return len(dets) if isinstance(dets, list) else 0
    return 0

def _count_list_col(row, col):
    val = row.get(col)
    return len(val) if isinstance(val, list) else 0

df["n_page_element_detections"] = df.apply(_count_pe_detections, axis=1)
df["n_table_crops"] = df.apply(lambda r: _count_list_col(r, "table"), axis=1)
df["n_chart_crops"] = df.apply(lambda r: _count_list_col(r, "chart"), axis=1)

_log(f"  Detection totals across {len(df)} rows:")
_log(f"    page_elements: {df['n_page_element_detections'].sum()}")
_log(f"    table crops:   {df['n_table_crops'].sum()}")
_log(f"    chart crops:   {df['n_chart_crops'].sum()}")

# ── Count per-label breakdown from page_elements_v3 ───────────────────
label_counts: dict[str, int] = {}
for pe in df["page_elements_v3"]:
    if isinstance(pe, dict):
        for det in (pe.get("detections") or []):
            if isinstance(det, dict):
                lbl = det.get("label_name", "unknown")
                label_counts[lbl] = label_counts.get(lbl, 0) + 1
_log(f"    per-label breakdown: {json.dumps(label_counts, indent=6)}")

# ── Strip image data and write full DataFrame to JSON ─────────────────
def _strip_images(val):
    if isinstance(val, dict):
        return {k: v for k, v in val.items()
                if k not in ("jpeg_bytes", "pixels", "image_b64", "orig_shape_hw")}
    return val

def _strip_embeddings(val):
    """Replace large embedding vectors with their length for readability."""
    if isinstance(val, dict) and "embedding" in val:
        emb = val["embedding"]
        out = dict(val)
        if hasattr(emb, "__len__"):
            out["embedding"] = f"<vector len={len(emb)}>"
        return out
    return val

df_export = df.copy()
if "page_image" in df_export.columns:
    df_export["page_image"] = df_export["page_image"].apply(_strip_images)

for col in df_export.columns:
    if "embed" in col.lower():
        df_export[col] = df_export[col].apply(_strip_embeddings)
    sample = df_export[col].iloc[0] if len(df_export) > 0 else None
    if isinstance(sample, (bytes, _np.ndarray)):
        df_export.drop(columns=[col], inplace=True)

EXPORT_PATH = DUMP_DIR / "full_dataframe.json"
df_export.to_json(EXPORT_PATH, orient="records", indent=2, default_handler=str)
_log(f"  → wrote full dataframe ({len(df_export)} rows, {len(df_export.columns)} cols) to {EXPORT_PATH}")

for i, row in df.head(3).iterrows():
    meta = row.get("metadata")
    embed_col = row.get("text_embeddings_1b_v2")
    meta_emb = meta.get("embedding") if isinstance(meta, dict) else None
    col_emb = embed_col.get("embedding") if isinstance(embed_col, dict) else None
    meta_len = len(meta_emb) if hasattr(meta_emb, '__len__') else "N/A"
    col_len = len(col_emb) if hasattr(col_emb, '__len__') else "N/A"
    _log(f"  row {i}: metadata.embedding len={meta_len} | "
         f"text_embeddings_1b_v2.embedding len={col_len}")

_log("Building LanceDB rows …")
lance_rows = build_lancedb_rows(df)
_log(f"  → {len(lance_rows)} rows with embeddings (from {len(df)} total rows)")

if lance_rows:
    sample_vec = lance_rows[0].get("vector", [])
    _log(f"  → first vector: len={len(sample_vec)}, first_5={sample_vec[:5]}")
    vecs_sample = [r["vector"] for r in lance_rows[:10] if "vector" in r]
    if len(vecs_sample) >= 2:
        v0 = _np.array(vecs_sample[0])
        all_same = all(_np.allclose(v0, _np.array(v)) for v in vecs_sample[1:])
        all_zero = _np.allclose(v0, 0.0)
        _log(f"  → first 10 vectors all identical: {all_same}")
        _log(f"  → first vector all zeros: {all_zero}")
        _log(f"  → first vector norm: {_np.linalg.norm(v0):.6f}")
    with open(DUMP_DIR / "lance_rows_sample.json", "w") as f:
        json.dump(lance_rows[:3], f, indent=2, default=str)
    _log(f"  → wrote {DUMP_DIR / 'lance_rows_sample.json'}")
    pages = set(r.get("pdf_page", "") for r in lance_rows[:20])
    _log(f"  → sample pdf_page values: {sorted(pages)[:5]}")

if not lance_rows:
    _log("ERROR: No embeddings found in the output. Skipping LanceDB + recall.")
else:
    dim = infer_vector_dim(lance_rows)
    _log(f"  → vector dim = {dim}")
    schema = lancedb_schema(vector_dim=dim)
    shutil.rmtree(LANCEDB_URI, ignore_errors=True)
    db = lancedb.connect(uri=LANCEDB_URI)
    _log(f"Writing {len(lance_rows)} rows to LanceDB at {LANCEDB_URI}/{LANCEDB_TABLE} …")
    t2 = time.perf_counter()
    table = create_or_append_lancedb_table(db, LANCEDB_TABLE, lance_rows, schema, overwrite=True)
    lance_cfg = LanceDBConfig(
        uri=LANCEDB_URI,
        table_name=LANCEDB_TABLE,
        overwrite=True,
        create_index=True,
    )
    create_lancedb_index(table, cfg=lance_cfg)
    _log(f"  → done in {time.perf_counter() - t2:.2f}s")

    from nemo_retriever.recall.core import RecallConfig, retrieve_and_score

    QUERY_CSV = Path(__file__).resolve().parent.parent / "data" / "bo767_query_gt.csv"

    recall_cfg = RecallConfig(
        lancedb_uri=LANCEDB_URI,
        lancedb_table=LANCEDB_TABLE,
        embedding_model="nvidia/llama-nemotron-embed-1b-v2",
        top_k=10,
        ks=(1, 5, 10),
        nprobes=0,
        refine_factor=10,
        match_mode="pdf_page",
    )

    with open(QUERY_CSV) as f:
        gt_rows = list(csv.DictReader(f))
    gt_pages = sorted(set(r.get("pdf_page", "") for r in gt_rows[:20]))
    _log(f"  Ground truth sample pdf_page values: {gt_pages[:5]}")

    _log(f"\nRunning recall evaluation on {QUERY_CSV} …")
    t3 = time.perf_counter()
    df_query, gold, raw_hits, retrieved_keys, metrics = retrieve_and_score(
        QUERY_CSV, cfg=recall_cfg,
    )
    recall_elapsed = time.perf_counter() - t3

    _log(f"\n{'='*60}")
    _log("Per-query results (first 20):")
    _log(f"{'='*60}")
    for i, (q, g, rk) in enumerate(zip(
        df_query["query"].astype(str).tolist(),
        gold,
        retrieved_keys,
    )):
        if i >= 20:
            break
        hit = "HIT" if g in rk[:10] else "MISS"
        _log(f"\n  [{i}] {hit} | query: {q[:80]}…" if len(q) > 80 else f"\n  [{i}] {hit} | query: {q}")
        _log(f"       expected: {g}")
        _log(f"       top-10:   {rk[:10]}")

    _log(f"\n{'='*60}")
    _log(f"Recall Results  ({len(df_query)} queries, top_k={recall_cfg.top_k})")
    _log(f"{'='*60}")
    for metric, value in metrics.items():
        _log(f"  {metric}: {value:.4f}")
    _log(f"\nRecall evaluation took {recall_elapsed:.2f}s")
    _log(f"Total pipeline time: {time.perf_counter() - t0:.2f}s")

_log(f"\n{'='*60}")
_log("Detection Summary")
_log(f"{'='*60}")
_log(f"  page_elements: {df['n_page_element_detections'].sum()}")
_log(f"  table crops:   {df['n_table_crops'].sum()}")
_log(f"  chart crops:   {df['n_chart_crops'].sum()}")
_log(f"  per-label:     {json.dumps(label_counts)}")

# Wait for Ray worker logs to flush, then print everything
_time.sleep(3)
print("\n" * 5)
print("#" * 80)
print("#  FINAL RESULTS (delayed to appear after Ray worker logs)")
print("#" * 80)
for line in _out:
    print(line)
print("#" * 80)
