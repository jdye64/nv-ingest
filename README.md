# Quickstart

## Links

- [Morpheus-Core](https://github.com/nv-morpheus/Morpheus)
- [Morpheus-MS](https://gitlab-master.nvidia.com/drobison/morpheus-pdf-ingest-ms)
- [Nemo Retriever](https://gitlab-master.nvidia.com/drobison/devin-nemo-retrieval-microservice-private)
- [Tracking Doc + Notes on Nemo Retriever](https://docs.google.com/document/d/12krzO82T-nhtyvTNh6Pbc8mqBOrBp4JZ9wWwM3OOR2Q/edit#heading=h.gya3si3lass1)
- [Ryan Angilly's Engineering Overview of Nemo Retriever](https://nvidia-my.sharepoint.com/personal/rangilly_nvidia_com/_layouts/15/stream.aspx?id=%2Fpersonal%2Frangilly%5Fnvidia%5Fcom%2FDocuments%2FRecordings%2FRyan%20Angilly%20presents%20how%20Retriever%20Services%20work%2D20240124%5F120001%2DMeeting%20Recording%2Emp4&referrer=StreamWebApp%2EWeb&referrerScenario=AddressBarCopied%2Eview&ga=1)

## Big picture Concepts

- We need to build the Morpheus-ms container so it can be deployed into the Nemo Retriever's docker compose cluster.
    - We'll build the Morpheus-ms container from the Morpheus runtime container, which is built from the Morpheus-Core
      repo.
- The Morpheus-ms container is, right now, basically the dependencies + entrypoint to run src/pipeline.py
    - We will build this from the Morpheus-ms repo, which consists of a library morpheus_pdf_ingest, and a pipeline
      defined in src/pipeline.py
- We can test the src/pipeline.py code outside the Nemo Retriever cluster without having to use/build the Morpheus-ms
  container, which is nice for development
    - The pipeline.py script expects to read from a Redis service, do various things, including run an inference model
      on
      Triton, and then return via Redis
    - Redis and Triton need to be deployed
    - Triton needs models to run.
    - The pipeline.py script starts a Morpheus pipeline that monitors a Redis list for new jobs, and then processes
      them, and returns a result.
    - We can use the src/util/submit_to_morpheus_ms.py script to submit a PDF to the Morpheus-ms service, good for
      fast iteration and experimentation.
- Once we've independently tested the Morpheus-ms, we can deploy it into the Nemo Retriever cluster and test it there.
    - The modified Nemo Retriever cluster and insertion hooks for Morpheus-ms can be found in Devin's private branch of
      the Nemo Retriever repo.
    - In the context of the Nemo Retriever, the Morpheus-ms service is used to process PDFs and return the results to
      the
      Nemo Retriever pipeline for indexing. Currently we insert into the document_indexing.py file, and forward new
      documents to the Morpheus-ms service for processing.

We need to build the morpheus-ms here that gets loaded by the Nemo Retriever pipeline

## Build Morpheus 24.03 (morpheus-ms-base:24.03) release container from source

### Clone the Morpheus repository and checkout the 24.03 release tag.

```bash
git clone https://github.com/nv-morpheus/Morpheus.git
git checkout branch-24.03
git submodule update --init --recursive
```

### Build Morpheus 24.03 (morpheus-ms-base:24.03) release container

```bash
DOCKER_IMAGE_NAME=morpheus-ms-base \
DOCKER_IMAGE_TAG=24.03 \
bash docker/build_container_release.sh
```

Wait for a while.

## Build the Morpheus-ms container (morpheus-ms:24.03) from the source

### Clone the Morpheus-ms repository

```bash
git clone https://gitlab-master.nvidia.com/drobison/morpheus-pdf-ingest-ms
```

```bash
docker build --tag morpheus-ms:24.03 .
```

## Verify src/pipeline.py is working as expected - This will submit a PDF directly to the morpheus-ms service, bypassing the Nemo Retriever pipeline

```bash
# In MORPHEUS_ROOT
## Make a new Conda environment from the Morpheus Requirements file
mamba env create -n morpheus -f docker/environments/conda/environments/all_cuda-121_arch-x86_64.yaml
conda activate morpheus

python examples/llm/main.py vdb_upload export-triton-model --model_name intfloat/e5-small-v2 \
  --triton_repo ./models/triton-model-repo
  
Created Triton Model at ./models/triton-model-repo/intfloat/e5-small-v2
Total time: 4.36 sec

export MORPHEUS_ROOT=[path to your morpheus core installation]
export MORPHEUS_PDF_INGEST_ROOT=[path to morpheus pdf ingest]

# In MORPHEUS_PDF_INGEST_ROOT
ln -s ${MORPHEUS_ROOT}/models:${MORPHEUS_PDF_INGEST_ROOT}/models_symlinks   # map in the pre-built models from morpheus. We don't have to, but if we don't we have to use our own.

docker run -p 6379:6379 redis  # Start Redis and make sure that we map its ports into our host network.
docker run --rm -ti --gpus=all -p8000:8000 -p8001:8001 -p8002:8002 -v $PWD/models_symlinks:/models nvcr.
io/nvidia/tritonserver:23.12-py3 tritonserver --model-repository=/models/triton-model-repo --exit-on-error=false 
--model-control-mode=explicit --load-model intfloat/e5-small-v2 --load-model all-MiniLM-L6-v2

## Make sure the triton server is running and the models are loaded with no errors reported ##

docker ps
CONTAINER ID   IMAGE                                   COMMAND                  CREATED          STATUS          PORTS                                                           NAMES
84a18da41101   nvcr.io/nvidia/tritonserver:23.12-py3   "/opt/nvidia/nvidia_…"   16 seconds ago   Up 15 seconds   0.0.0.0:8000-8002->8000-8002/tcp, :::8000-8002->8000-8002/tcp   eloquent_austin
10357419c96a   redis                                   "docker-entrypoint.s…"   28 seconds ago   Up 27 seconds   0.0.0.0:6379->6379/tcp, :::6379->6379/tcp                       romantic_eu

# In a second terminal, start the morpheus-ms service
python src/pipeline.py
DEBUG:morpheus.utils.module_utils:Module 'nemo_document_splitter' was successfully registered with 'morpheus_pdf_ingest' namespace.
DEBUG:morpheus.utils.module_utils:Module 'pdf_text_extractor' was successfully registered with 'morpheus_pdf_ingest' namespace.
DEBUG:morpheus.utils.module_utils:Module 'redis_task_sink' was successfully registered with 'morpheus_pdf_ingest' namespace.
DEBUG:morpheus.utils.module_utils:Module 'redis_task_source' was successfully registered with 'morpheus_pdf_ingest' namespace.
INFO:root:Starting pipeline setup
INFO:root:Pipeline setup completed in 0.00 seconds
INFO:root:Running pipeline
DEBUG:asyncio:Using selector: EpollSelector
INFO:morpheus.pipeline.pipeline:====Pipeline Pre-build====
INFO:morpheus.pipeline.pipeline:====Pre-Building Segment: main====
INFO:morpheus.pipeline.pipeline:====Pre-Building Segment Complete!====
INFO:morpheus.pipeline.pipeline:====Pipeline Pre-build Complete!====
INFO:morpheus.pipeline.pipeline:====Registering Pipeline====
INFO:morpheus.pipeline.pipeline:====Building Pipeline====
INFO:morpheus.pipeline.pipeline:====Building Pipeline Complete!====
INFO:morpheus.pipeline.pipeline:====Registering Pipeline Complete!====
INFO:morpheus.pipeline.pipeline:====Starting Pipeline====
INFO:morpheus.pipeline.pipeline:====Pipeline Started====
INFO:morpheus.pipeline.pipeline:====Building Segment: main====
DEBUG:morpheus.utils.module_utils:Module 'redis_task_source' with namespace 'morpheus_pdf_ingest' is successfully loaded.
INFO:morpheus.pipeline.single_output_source:Added source: <redis_listener-0; LinearModuleSourceStage(module_config=<morpheus.utils.module_utils.ModuleLoader object at 0x7f9550635660>, output_port_name=output, output_type=<class 'morpheus._lib.messages.ControlMessage'>)>
  └─> morpheus.ControlMessage
DEBUG:morpheus.utils.module_utils:Module 'pdf_text_extractor' with namespace 'morpheus_pdf_ingest' is successfully loaded.
INFO:morpheus.pipeline.single_port_stage:Added stage: <pdf_extractor-1; LinearModulesStage(module_config=<morpheus.utils.module_utils.ModuleLoader object at 0x7f95506358d0>, input_port_name=input, output_port_name=output, input_type=<class 'morpheus._lib.messages.ControlMessage'>, output_type=<class 'morpheus._lib.messages.ControlMessage'>)>
  └─ morpheus.ControlMessage -> morpheus.ControlMessage
DEBUG:morpheus.utils.module_utils:Module 'nemo_document_splitter' with namespace 'morpheus_pdf_ingest' is successfully loaded.
INFO:morpheus.pipeline.single_port_stage:Added stage: <nemo_doc_splitter-2; LinearModulesStage(module_config=<morpheus.utils.module_utils.ModuleLoader object at 0x7f95506359c0>, input_port_name=input, output_port_name=output, input_type=<class 'morpheus._lib.messages.ControlMessage'>, output_type=<class 'morpheus._lib.messages.ControlMessage'>)>
  └─ morpheus.ControlMessage -> morpheus.ControlMessage
INFO:morpheus.pipeline.single_port_stage:Added stage: <preprocess-nlp-3; PreprocessNLPStage(vocab_hash_file=data/bert-base-uncased-hash.txt, truncation=True, do_lower_case=True, add_special_tokens=False, stride=-1, column=content)>
  └─ morpheus.ControlMessage -> morpheus.MultiInferenceMessage
INFO:morpheus.pipeline.single_port_stage:Added stage: <inference-4; TritonInferenceStage(model_name=all-MiniLM-L6-v2, server_url=triton:8001, force_convert_inputs=True, use_shared_memory=True, needs_logits=None, inout_mapping=None)>
  └─ morpheus.MultiInferenceMessage -> morpheus.MultiResponseMessage
DEBUG:morpheus.utils.module_utils:Module 'redis_task_sink' with namespace 'morpheus_pdf_ingest' is successfully loaded.
INFO:morpheus.pipeline.single_port_stage:Added stage: <redis_task_sink-5; LinearModulesStage(module_config=<morpheus.utils.module_utils.ModuleLoader object at 0x7f9550635cc0>, input_port_name=input, output_port_name=output, input_type=typing.Any, output_type=<class 'morpheus._lib.messages.ControlMessage'>)>
  └─ morpheus.MultiResponseMessage -> morpheus.ControlMessage
INFO:morpheus.pipeline.pipeline:====Building Segment Complete!====

# In another terminal, submit a PDF to the morpheus-ms service

# Nothing formal in this yet, add whatever PDFs you want to the data/pdf_ingest_testing directory
# Useful: https://www.gutenberg.org/browse/scores/top
python ./src/util/submit_to_morpheus_ms.py \
  --file_source ./data/pdf_ingest_testing/[pdf_name].pdf \
  --enable_pdf_extract \
  --enable_split
```

## Launch the Nemo Retriever pipeline with the morpheus-ms service

Note: As of 9 Feb, 2024 you will need to pull the Devin's experimental Nemo Retrieval pipeline branch.
Note: Currently, only the 'index' side of the Nemo pipeline has hooks into the morpheus-ms, 'query' side still needs
work.

### Clone the experimental Nemo Retrieval pipeline branch

```bash
git clone https://gitlab-master.nvidia.com/drobison/devin-nemo-retrieval-microservice-private
git checkout devin_morpheus_pdf_ingest
```

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
