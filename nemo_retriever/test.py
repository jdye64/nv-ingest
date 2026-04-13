import time
from pathlib import Path
from nemo_retriever import create_ingestor
from nemo_retriever.params import ExtractParams, EmbedParams, RemoteRetryParams

docs = [str(Path("/datasets/nv-ingest/bo767").resolve())]

extract = ExtractParams(
    method="pdfium",
    page_elements_invoke_url="http://localhost:8000/v1/infer",
    ocr_invoke_url="http://localhost:8009/v1/infer",
    table_structure_invoke_url="http://localhost:8006/v1/infer",
    graphic_elements_invoke_url="http://localhost:8003/v1/infer",
    use_table_structure=True,
    use_graphic_elements=True,
    remote_retry=RemoteRetryParams(remote_max_pool_workers=48),
)

embed = EmbedParams(
    embed_invoke_url="http://localhost:8012/v1/embed",
    nim_http_max_concurrent=48,
)

ing = create_ingestor(run_mode="batch", ray_address="auto")
ing = ing.files(docs).extract(extract).embed(embed)
t0 = time.perf_counter()
ray_ds = ing.ingest()
raw_num_rows = ray_ds.count()
elapsed = time.perf_counter() - t0
num_rows = 54730
print(f"raw_num_rows: {raw_num_rows}")
print("rows", num_rows)
print(f"pages/sec: {num_rows / elapsed:.2f}  ({num_rows} pages in {elapsed:.2f}s)")
