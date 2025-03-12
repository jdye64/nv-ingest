from nv_ingest_client.client import Ingestor
from nv_ingest_client.util.milvus import create_nvingest_schema, create_nvingest_index_params, create_collection, write_records_minio, bulk_insert_milvus, dense_retrieval, hybrid_retrieval, create_bm25_model, write_to_nvingest_collection, create_nvingest_collection, nvingest_retrieval

import time, os, json, glob, logging
from tqdm import tqdm
from collections import defaultdict
from datetime import datetime
from itertools import chain

import numpy as np
import pandas as pd

from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.embeddings.nvidia import NVIDIAEmbedding

now = datetime.now()
dt_hour  = now.strftime("%m-%d-%y_%I")

log_fn = "test_results/bo767_full.json"
logs = []
def log(key, val):
  dt_min = now.strftime("%m-%d-%y_%I:%M")
  print(f"{dt_min}: {key}: {val}")
  logs.append({key: val})
  with open(log_fn, "w") as fp:
    fp.write(json.dumps(logs))

t0 = time.time()

hostname = "localhost"

ingestion_start = time.time()
ingestor = (
    Ingestor(message_client_hostname=hostname)
    .files("/raid/jdyer/bo767/*.pdf")
    .extract(
        extract_text=True,
        extract_tables=True,
        extract_charts=True,
        extract_images=False,
        text_depth="page",
        paddle_output_format="markdown",
        #extract_infographics=True,
        #extract_method="nemoretriever_parse"
    ).embed()
)
results = ingestor.ingest(show_progress=True)
ingestion_end = time.time()
ingestion_time = ingestion_end - ingestion_start
log("ingestion_time", ingestion_time)
log("ingestion_pages_per_sec", 54730/ingestion_time)
log("ingestion_files_per_sec", 767/ingestion_time)
log("record_count", len(results))

indexing_start = time.time()
sparse = False
dense_dim = 2048

schema_start = time.time()
create_nvingest_collection("text", f"http://{hostname}:19530", sparse=sparse)
create_nvingest_collection("tables", f"http://{hostname}:19530", sparse=sparse)
create_nvingest_collection("charts", f"http://{hostname}:19530", sparse=sparse)
schema_end = time.time()
log("schema_creation", schema_end-schema_start)

text_results = [[element for element in results if element['document_type'] == 'text'] for results in results]
table_results = [[element for element in results if element['metadata']['content_metadata']['subtype'] == 'table'] for results in results]
chart_results = [[element for element in results if element['metadata']['content_metadata']['subtype'] == 'chart'] for results in results]
log("text_record_count", len(text_results))
log("table_record_count", len(table_results))
log("chart_record_count", len(chart_results))
write_to_nvingest_collection(text_results, "text", sparse=sparse, milvus_uri=f"http://{hostname}:19530", minio_endpoint="localhost:9000")
write_to_nvingest_collection(table_results, "tables", sparse=sparse, milvus_uri=f"http://{hostname}:19530", minio_endpoint="localhost:9000")
write_to_nvingest_collection(chart_results, "charts", sparse=sparse, milvus_uri=f"http://{hostname}:19530", minio_endpoint="localhost:9000")

multimodal_index_start = time.time()
create_nvingest_collection("multimodal", f"http://{hostname}:19530", sparse=sparse, gpu_search=True)
write_to_nvingest_collection(results, "multimodal", sparse=sparse, milvus_uri=f"http://{hostname}:19530", minio_endpoint="localhost:9000")
multimodal_index_end = time.time()
multimodal_indexing_time = multimodal_index_end - multimodal_index_start
log("multimodal_indexing_time", multimodal_indexing_time)
log("total_indexing_time", time.time()-indexing_start)
log("e2e_runtime", ingestion_time + multimodal_indexing_time)
log("e2e_pages_per_sec", 54730/(ingestion_time + multimodal_indexing_time))

schema = create_nvingest_schema(sparse=sparse, dense_dim=dense_dim)

def get_recall_scores(query_df, collection_name):
    hits = defaultdict(list)
    all_answers = nvingest_retrieval(
        df_query["query"].to_list(),
        collection_name,
        hybrid=sparse,
        embedding_endpoint="http://localhost:8012/v1",
        model_name="nvidia/llama-3.2-nv-embedqa-1b-v2",
        top_k=10,
        #nv_ranker_endpoint=f"http://{hostname}:8015/v1/ranking",
        #nv_ranker_model_name="nvidia/llama-3.2-nv-rerankqa-1b-v2",
        #nv_ranker=True,
        #gpu_search=gpu_search,
    )

    for i in range(len(df_query)):
        expected_pdf_page = query_df['pdf_page'][i]
        retrieved_answers = all_answers[i]
        retrieved_pdfs = [os.path.basename(result['entity']['source']['source_id']).split('.')[0] for result in retrieved_answers]
        retrieved_pages = [str(result['entity']['content_metadata']['page_number']) for result in retrieved_answers]
        retrieved_pdf_pages = [f"{pdf}_{page}" for pdf, page in zip(retrieved_pdfs, retrieved_pages)]    

        for k in [1, 3, 5, 10]:
            hits[k].append(expected_pdf_page in retrieved_pdf_pages[:k])
    
    for k in hits:
        print(f'  - Recall @{k}: {np.mean(hits[k]) :.3f}')

    return hits


total_queries = 0
# table recall
hits = defaultdict(list)
table_start = time.time()
df_query = pd.read_csv('data/table_queries_cleaned_235.csv')[['query','pdf','page','table']]
df_query['pdf_page'] = df_query.apply(lambda x: f"{x.pdf}_{x.page}", axis=1)
total_queries += len(df_query)
hits = get_recall_scores(df_query, "tables")
table_end = time.time()
table_time = table_end-table_start
log("table_recall", table_time)
log("table_qps", len(df_query)/table_time)
table_recalls = [{"recall @"+str(k): np.mean(hits[k])} for k in hits]
with open(f"test_results/{dt_hour}_table_recall.json", "w") as fp:
  fp.write(json.dumps(table_recalls))

# chart recall
hits = defaultdict(list)
chart_start = time.time()
df_query = pd.read_csv('data/charts_with_page_num_fixed.csv')[['query','pdf','page']]
df_query['page'] = df_query['page']-1 # page -1 because the page number starts with 1 in that csv
df_query['pdf_page'] = df_query.apply(lambda x: f"{x.pdf}_{x.page}", axis=1) 
total_queries += len(df_query)
hits = get_recall_scores(df_query, "charts")
chart_end = time.time()
chart_time = chart_end-chart_start
log("chart_recall", chart_time)
log("chart_qps", len(df_query)/chart_time)
chart_recalls = [{"recall @"+str(k): np.mean(hits[k])} for k in hits]
with open(f"test_results/{dt_hour}_chart_recall.json", "w") as fp:
  fp.write(json.dumps(chart_recalls))

# text recall
hits = defaultdict(list)
text_start = time.time()
df_query = pd.read_csv('data/text_query_answer_gt_page.csv')
df_query.pdf = df_query.pdf.apply(lambda x: x.replace('.pdf',''))
df_query['pdf_page'] = df_query.apply(lambda x: f"{x.pdf}_{x.gt_page}", axis=1)
total_queries += len(df_query)
hits = get_recall_scores(df_query, "text")
text_end = time.time()
text_time = text_end-text_start
log("text_recall", text_time)
log("text_qps", len(df_query)/text_time)
text_recalls = [{"recall @"+str(k): np.mean(hits[k])} for k in hits]
with open(f"test_results/{dt_hour}_text_recall.json", "w") as fp:
  fp.write(json.dumps(text_recalls))

#multimodal recall
hits = defaultdict(list)
multimodal_start = time.time()
df_query = pd.read_csv('data/text_query_answer_gt_page.csv').rename(columns={'gt_page':'page'})[['query','pdf','page']]
df_query.pdf = df_query.pdf.apply(lambda x: x.replace('.pdf',''))
df_query['modality'] = 'text'
df_query2 = pd.read_csv('data/table_queries_cleaned_235.csv')[['query','pdf','page']]
df_query2['modality'] = 'table'

df_query3 = pd.read_csv('data/charts_with_page_num_fixed.csv')[['query','pdf','page']]
df_query3['page'] = df_query3['page']-1 # page -1 because the page number starts with 1 in that csv
df_query3['modality'] = 'chart'

df_query = pd.concat([df_query, df_query2, df_query3]).reset_index(drop=True)
df_query['pdf_page'] = df_query.apply(lambda x: f"{x.pdf}_{x.page}", axis=1) 
total_queries += len(df_query)
hits = get_recall_scores(df_query, "multimodal")
multimodal_end = time.time()
multimodal_time = multimodal_end-multimodal_start
log("multimodal_recall", multimodal_time)
log("multimodal_qps", len(df_query)/multimodal_time)
multimodal_recalls = [{"recall @"+str(k): np.mean(hits[k])} for k in hits]
log("total_qps", total_queries/(multimodal_time + chart_time + table_time + text_time))
with open(f"test_results/{dt_hour}_multimodal_recall.json", "w") as fp:
  fp.write(json.dumps(multimodal_recalls))
