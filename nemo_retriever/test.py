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
        "PDFExtractionActor": {"concurrency": 64},
        # Single-GPU budget: total = 1.0 GPU so all stages coexist.
        #   5×0.05 + 2×0.05 + 2×0.05 + 5×0.05 + 6×0.05 = 1.0 GPU
        "PageElementDetectionActor": {"concurrency": 5, "num_gpus": 0.05},
        "TableStructureActor": {"concurrency": 2, "num_gpus": 0.05},
        "GraphicElementsActor": {"concurrency": 2, "num_gpus": 0.05},
        "OCRActor": {"concurrency": 5, "num_gpus": 0.05},
        "_BatchEmbedActor": {"concurrency": 6, "num_gpus": 0.05},
    },
)
ing = ing.files(docs).extract(extract).embed(embed)
t0 = time.perf_counter()
ray_ds = ing.ingest()
raw_num_rows = ray_ds.count()
elapsed = time.perf_counter() - t0
num_rows = 54730
print(f"raw_num_rows: {raw_num_rows}")
print("rows", num_rows)
print(f"pages/sec: {num_rows / elapsed:.2f}  ({num_rows} pages in {elapsed:.2f}s)")
