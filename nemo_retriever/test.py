import time
from pathlib import Path
from nemo_retriever.graph_ingestor import GraphIngestor
from nemo_retriever.params import ExtractParams, EmbedParams

docs = [str(Path("/datasets/nv-ingest/bo767").resolve())]

TRT_MODEL_DIR = Path("/models/trt")

extract = ExtractParams(
    method="pdfium",
    page_elements_trt_engine_path=str(TRT_MODEL_DIR / "page_elements.engine"),
    table_structure_trt_engine_path=str(TRT_MODEL_DIR / "table_structure.engine"),
    graphic_elements_trt_engine_path=str(TRT_MODEL_DIR / "graphic_elements.engine"),
    ocr_trt_engine_path=str(TRT_MODEL_DIR / "ocr_detector.engine"),
    use_table_structure=True,
    use_graphic_elements=True,
)

embed = EmbedParams(
    embed_invoke_url="http://localhost:8012/v1/embeddings",
    nim_http_max_concurrent=16,
)

ing = GraphIngestor(
    run_mode="batch",
    ray_address="auto",
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
