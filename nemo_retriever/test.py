import time
from pathlib import Path
from nemo_retriever.graph_ingestor import GraphIngestor
from nemo_retriever.params import ExtractParams, EmbedParams

docs = [str(Path("/datasets/nv-ingest/bo767").resolve())]

NGC_MODELS = Path("/datasets/nv-ingest/models")

extract = ExtractParams(
    method="pdfium",
    page_elements_trt_engine_path=str(NGC_MODELS / "page-elements/ngc/hub/models--nim--nvidia--nemotron-page-elements-v3"),
    table_structure_trt_engine_path=str(NGC_MODELS / "table-structure/ngc/hub/models--nim--nvidia--nemotron-table-structure-v1"),
    graphic_elements_trt_engine_path=str(NGC_MODELS / "graphic-elements/ngc/hub/models--nim--nvidia--nemotron-graphic-elements-v1"),
    ocr_trt_engine_path=str(NGC_MODELS / "ocr"),
    use_table_structure=True,
    use_graphic_elements=True,
    inference_batch_size=32,
)

embed = EmbedParams(
    embed_trt_engine_path=str(NGC_MODELS / "embedding/ngc/hub/models--nim--nvidia--llama-nemotron-embed-1b-v2"),
)

ing = GraphIngestor(
    run_mode="batch",
    ray_address="auto",
    node_overrides={
        "PDFExtractionActor": {"concurrency": 48, "batch_size": 16},
        # GPU actors: large batch_size feeds more rows per invocation,
        # higher concurrency keeps multiple actors queued on the GPU.
        #   6×0.05 + 4×0.05 + 4×0.05 + 8×0.05 + 8×0.05 = 1.5
        #   B200 handles this easily with fractional scheduling.
        "PageElementDetectionActor": {"concurrency": 6, "num_gpus": 0.05, "batch_size": 64},
        "TableStructureActor": {"concurrency": 4, "num_gpus": 0.05, "batch_size": 32},
        "GraphicElementsActor": {"concurrency": 4, "num_gpus": 0.05, "batch_size": 32},
        "OCRActor": {"concurrency": 8, "num_gpus": 0.05, "batch_size": 64},
        "_BatchEmbedActor": {"concurrency": 8, "num_gpus": 0.05, "batch_size": 512},
    },
)
ing = ing.files(docs).extract(extract).embed(embed)
t0 = time.perf_counter()
ray_ds = ing.ingest()
raw_num_rows = ray_ds.count()
elapsed = time.perf_counter() - t0
print(f"\n{'='*60}")
print(f"Ingestion complete: {raw_num_rows} rows in {elapsed:.2f}s")
print(f"Throughput: {raw_num_rows / elapsed:.2f} rows/sec")
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

print("Materialising Ray Dataset to pandas …")
t1 = time.perf_counter()
df = ray_ds.to_pandas()
print(f"  → {len(df)} rows in {time.perf_counter() - t1:.2f}s")

print("Building LanceDB rows …")
lance_rows = build_lancedb_rows(df)
print(f"  → {len(lance_rows)} rows with embeddings (from {len(df)} total rows)")

if not lance_rows:
    print("ERROR: No embeddings found in the output. Skipping LanceDB + recall.")
else:
    dim = infer_vector_dim(lance_rows)
    schema = lancedb_schema(vector_dim=dim)
    db = lancedb.connect(uri=LANCEDB_URI)
    print(f"Writing {len(lance_rows)} rows to LanceDB at {LANCEDB_URI}/{LANCEDB_TABLE} …")
    t2 = time.perf_counter()
    table = create_or_append_lancedb_table(db, LANCEDB_TABLE, lance_rows, schema, overwrite=True)
    lance_cfg = LanceDBConfig(
        uri=LANCEDB_URI,
        table_name=LANCEDB_TABLE,
        overwrite=True,
        create_index=True,
    )
    create_lancedb_index(table, cfg=lance_cfg)
    print(f"  → done in {time.perf_counter() - t2:.2f}s")

    # ── Recall evaluation ────────────────────────────────────────────
    from nemo_retriever.recall.core import RecallConfig, evaluate_recall

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

    print(f"\nRunning recall evaluation on {QUERY_CSV} …")
    t3 = time.perf_counter()
    result = evaluate_recall(QUERY_CSV, cfg=recall_cfg)
    recall_elapsed = time.perf_counter() - t3

    print(f"\n{'='*60}")
    print(f"Recall Results  ({result['n_queries']} queries, top_k={result['top_k']})")
    print(f"{'='*60}")
    for metric, value in result["metrics"].items():
        print(f"  {metric}: {value:.4f}")
    print(f"\nRecall evaluation took {recall_elapsed:.2f}s")
    print(f"Total pipeline time: {time.perf_counter() - t0:.2f}s")
