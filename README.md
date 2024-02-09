# Quickstart

We need to build the morpheus-ms here that gets loaded by the Nemo Retriever pipeline

# Build Morpheus 24.03 release container from source

### Clone the Morpheus repository and checkout the 24.03 release tag.

```bash
git clone https://github.com/nv-morpheus/Morpheus.git
git checkout branch-24.03
git submodule update --init --recursive
```

### Build the Morpheus-ms-base container

```bash
DOCKER_IMAGE_NAME=morpheus-ms-base \
DOCKER_IMAGE_TAG=24.03 \
bash docker/build_container_release.sh
```

Wait for a while.

# Build the Morpheus-ms container

```bash
docker build --tag morpheus-ms:24.03 .
```

### Check if the morpheus-ms is running and responding correctly - This will submit a PDF directly to the morpheus-ms service, bypassing the Nemo Retriever pipeline

```bash
# In a second terminal, start the morpheus-ms service
python src/pipeline.py

# In another terminal, submit a PDF to the morpheus-ms service

# Nothing formal in this yet, add whatever PDFs you want to the data/pdf_ingest_testing directory
# Useful: https://www.gutenberg.org/browse/scores/top
python ./src/util/submit_to_morpheus_ms.py \
  --file_source ./data/pdf_ingest_testing/[pdf_name].pdf \
  --enable_pdf_extract \
  --enable_split
```

# Launch the Nemo Retriever pipeline with the morpheus-ms service

Note: as of 9 Feb, 2024 you will need to pull the Devin's experimental Nemo Retrieval pipeline branch.

### Create a Nemo Retriever pipeline with the morpheus-ms service

You should see three new services `morpheus_ms`, `redis`, and `triton` in the docker compose ps output.

```bash
docker compose -f docker-compose.yaml -f docker-compose.override.gpu.yaml --build up

docker compose ps
Every 2.0s: docker compose ps                                                                                                                                                                                                                                              drobison-mint: Fri Feb  9 14:39:34 2024

NAME                                                         IMAGE                                                                            COMMAND                  SERVICE          CREATED        STATUS                    PORTS
devin-nemo-retrieval-microservice-private-elasticsearch-1    docker.elastic.co/elasticsearch/elasticsearch:8.12.0                             "/bin/tini -- /usr/l…"   elasticsearch    4 days ago     Up 2 days (healthy)       0.0.0.0:9200->9200/tcp, :::9200->9200/tcp, 9300/tcp
devin-nemo-retrieval-microservice-private-embedding-ms-1     nvcr.io/nvidian/nemo-llm/embedding-ms:f0cfd7168aac4cccd605e95355c895a158fee2b3   "/opt/docker_entrypo…"   embedding-ms     16 hours ago   Up 13 hours (healthy)     0.0.0.0:8080->8080/tcp, :::8080->8080/tcp
devin-nemo-retrieval-microservice-private-etcd-1             quay.io/coreos/etcd:v3.5.11                                                      "etcd -advertise-cli…"   etcd             4 days ago     Up 38 hours (healthy)     2379-2380/tcp
devin-nemo-retrieval-microservice-private-milvus-1           milvusdb/milvus:v2.3.5                                                           "/tini -- milvus run…"   milvus           4 days ago     Up 38 hours (healthy)
devin-nemo-retrieval-microservice-private-minio-1            minio/minio:RELEASE.2023-03-20T20-16-18Z                                         "/usr/bin/docker-ent…"   minio            4 days ago     Up 38 hours (healthy)     9000/tcp
devin-nemo-retrieval-microservice-private-morpheus-ms-1      morpheus-ms:24.03                                                                "/opt/conda/bin/tini…"   morpheus-ms      13 hours ago   Up 13 hours
devin-nemo-retrieval-microservice-private-otel-collector-1   otel/opentelemetry-collector-contrib:0.91.0                                      "/otelcol-contrib --…"   otel-collector   4 days ago     Up 38 hours               0.0.0.0:4317->4317/tcp, :::4317->4317/tcp, 0.0.0.0:13133->13133/tcp, :::13133->13
133/tcp, 0.0.0.0:55679->55679/tcp, :::55679->55679/tcp, 55678/tcp
devin-nemo-retrieval-microservice-private-pgvector-1         ankane/pgvector                                                                  "docker-entrypoint.s…"   pgvector         4 days ago     Up 2 days                 0.0.0.0:5433->5432/tcp, :::5433->5432/tcp
devin-nemo-retrieval-microservice-private-postgres-1         postgres:16.1                                                                    "docker-entrypoint.s…"   postgres         4 days ago     Up 2 days                 0.0.0.0:5432->5432/tcp, :::5432->5432/tcp
devin-nemo-retrieval-microservice-private-redis-1            redis:latest                                                                     "docker-entrypoint.s…"   redis            4 days ago     Up 38 hours               0.0.0.0:6379->6379/tcp, :::6379->6379/tcp
devin-nemo-retrieval-microservice-private-retrieval-ms-1     devin-nemo-retrieval-microservice-private-retrieval-ms                           "/bin/sh -c 'opentel…"   retrieval-ms     16 hours ago   Up 16 hours (unhealthy)   0.0.0.0:1984->8000/tcp, :::1984->8000/tcp
devin-nemo-retrieval-microservice-private-tika-1             apache/tika:latest                                                               "/bin/sh -c 'exec ja…"   tika             4 days ago     Up 16 hours               0.0.0.0:9998->9998/tcp, :::9998->9998/tcp
devin-nemo-retrieval-microservice-private-triton-1           nvcr.io/nvidia/tritonserver:23.12-py3                                            "/opt/nvidia/nvidia_…"   triton           18 hours ago   Up 16 hours               0.0.0.0:8000-8002->8000-8002/tcp, :::8000-8002->8000-8002/tcp
devin-nemo-retrieval-microservice-private-zipkin-1           openzipkin/zipkin                                                                "start-zipkin"           zipkin           4 days ago     Up 38 hours (healthy)     9410/tcp, 0.0.0.0:9411->9411/tcp, :::9411->9411/tcp

```

### Create collections for performance comparison, and export the collection ID's

Note: the collection ID will be associated with the pipeline specified in the request, and subsequent index and 
query calls will use the associated pipeline. 

The Haystack pipeline will run the pipeline defined in `[nemo_retriever]/pipelines/dense_milvus.mustache`, where 
initial pdf text extraction will be done by the Tika service, and the text will then be chunked, embedded, and 
uploaded to Milvus by nemo defined haystack components.

The Morpheus pipeline will run the pipeline defined in `[nemo_retriever]/pipelines/test_writer_only.mustache`, this 
will call out to the morpheus-ms to do pdf text extraction, chunking, and embedding, then return the results to the 
Haystack DocumentWriter component for upload.

See: `[nemo_retriever]/src/v1/document_indexing.py` for ingest API modifications.
See: `[nemo_retriever]/src/components/nemo*` for Nemo Haystack component definitions.

```bash
curl http://localhost:1984/v1/collections?pretty=true \
> -H 'Content-Type: application/json' \
> -d '{"name": "test_collection", "pipeline": "dense_milvus"}'

{
  "collection": {
    "pipeline": "dense_milvus",
    "name": "test_collection",
    "id": "a962acb5-f1e7-4632-add7-1ca601063287"
  }
}

curl http://localhost:1984/v1/collections?pretty=true \
> -H 'Content-Type: application/json' \
> -d '{"name": "test_collection", "pipeline": "test_writer_only"}'

{
  "collection": {
    "pipeline": "test_writer_only",
    "name": "test_collection",
    "id": "33e0b745-585d-44c6-8db0-b03b841ea50b"
  }
}

export HAYSTACK_PATH_COLLECTION_ID=a962acb5-f1e7-4632-add7-1ca601063287
export MORPHEUS_PATH_COLLECTION_ID=33e0b745-585d-44c6-8db0-b03b841ea50b
```

### Upload documents to the indexing endpoint

Note: `[nemo_retriever]/script/quickstart/upload.py` has also been modified to support new options `--silent` and 
`--debug_pdf_extract_method=[tika|morpheus]` 

Note: You may want to run the upload command below multiple times as a warm up; the first time the Triton service is 
hit, it tends to take an unusually long time to respond.

```bash
time python script/quickstart/upload.py -c --debug_pdf_extract_method=tika --silent ${HAYSTACK_PATH_COLLECTION_ID} data/
[pdf_name].pdf
time python script/quickstart/upload.py -c --debug_pdf_extract_method=morpheus --silent ${MORPHEUS_PATH_COLLECTION_ID} 
data/ [pdf_name].pdf
```

At present stat collection is a manual process from trace logs.

### For the Haystack pipeline:
```bash
retrieval-ms-1  | INFO:     Using Tika to process PDFs.
retrieval-ms-1  | INFO:     TIKA ELAPSED: 7236.22189 ms
retrieval-ms-1  | INFO:     Nemo Document split: 160.444888 ms.
retrieval-ms-1  | INFO:     Nemo Document embedding: 13578.185228 ms.
```

### For the Morpheus pipeline:
```bash
morpheus-ms-1  | DEBUG:morpheus_pdf_ingest.modules.redis_task_source:latency::redis_source_retrieve: 33.609341 msec.
morpheus-ms-1  | DEBUG:morpheus_pdf_ingest.modules.redis_task_source:throughput::redis_source_retrieve: -2.9753633074805006e-05 MB/sec.
morpheus-ms-1  | DEBUG:root:pdf_text_extractor since ts_send: 15.98558 msec.
morpheus-ms-1  | DEBUG:root:pdf_text_extractor elapsed time 213.460169 msec.
morpheus-ms-1  | DEBUG:root:nemo_document_splitter since ts_send: 8.10594 msec.
morpheus-ms-1  | DEBUG:root:nemo_document_splitter elapsed time 34.773735 msec.
```
