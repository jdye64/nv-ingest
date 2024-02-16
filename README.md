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
    - We can use the src/util/upload_to_ingest_ms.py script to submit a PDF to the Morpheus-ms service, good for
      fast iteration and experimentation.
- Once we've independently tested the Morpheus-ms, we can deploy it into the Nemo Retriever cluster and test it there.
    - The modified Nemo Retriever cluster and insertion hooks for Morpheus-ms can be found in Devin's private branch of
      the Nemo Retriever repo.
    - In the context of the Nemo Retriever, the Morpheus-ms service is used to process PDFs and return the results to
      the
      Nemo Retriever pipeline for indexing. Currently we insert into the document_indexing.py file, and forward new
      documents to the Morpheus-ms service for processing.

We need to build the morpheus-ms here that gets loaded by the Nemo Retriever pipeline

## Running the pipeline

### Clone the Morpheus repository and checkout the 24.03 release tag.

```bash
git clone https://github.com/nv-morpheus/Morpheus.git
git checkout branch-24.03
git submodule update --init --recursive

./scripts/fetch_data.py fetch all # pull down all the LFS artifacts for Morpheus, including pre-built models
```

Note the path to the Morpheus repository; we will need it in the next step.

Go to the `morpheus-pdf-ingest-ms` directory and open `.env` with your favorite editor, we will add some envrionment
variables.

`.env`

```bash
MORPHEUS_PDF_INGEST_ROOT=[PATH TO MORPHEUS PDF INGEST MS ROOT]
MORPHEUS_ROOT=[PATH TO MORPHEUS ROOT]
MODEL_NAME=intfloat/e5-small-v2
```

### Clone the Morpheus-ms repository

```bash
git clone https://gitlab-master.nvidia.com/drobison/morpheus-pdf-ingest-ms
```

### Build Morpheus 24.03 (morpheus-ms-base:24.03) release container

```bash
docker compose build morpheus-ms-base
```

### Create Triton model

By default, the model is `intfloat/e5-small-v2`. You can override the model by editing `MODEL_NAME` in `.env`.

```bash
docker compose run morpheus-ms-base
```

```
Created Triton Model at /models/triton-model-repo/intfloat/e5-small-v2
Total time: 8.81 sec
```

### Start supporting services

You can start Redis and Triton Inference Server using the provided docker-compose file. Use `docker compose up` to start
one or both of them. Triton is optional at the moment -- unless you are testing ingestion embedding creation.

```bash
docker compose pull redis triton
docker compose up -d redis triton
```

The `-d` option will start the containers in "detached" mode in the background.

Make sure the triton server is running and the models are loaded with no errors reported.

```bash
$ docker ps
CONTAINER ID   IMAGE                                   COMMAND                  CREATED              STATUS              PORTS                                                           NAMES
4e051d750bdc   nvcr.io/nvidia/tritonserver:23.12-py3   "/opt/nvidia/nvidia_…"   About a minute ago   Up About a minute   0.0.0.0:8000-8002->8000-8002/tcp, :::8000-8002->8000-8002/tcp   morpheus-pdf-ingest-ms-triton-1
de13d0d34d57   redis/redis-stack                       "/entrypoint.sh"         About a minute ago   Up About a minute   0.0.0.0:6379->6379/tcp, :::6379->6379/tcp, 8001/tcp             morpheus-pdf-ingest-ms-redis-1
```

```bash
$ docker logs morpheus-pdf-ingest-ms-triton-1
```

### Build the Morpheus-ms container (morpheus-ms:24.03) from the source

```bash
docker compose build morpheus-ms
$ docker compose run -d morpheus-ms
```

Verify `pipeline.py` is working as expected.

```bash
$ docker ps                                                                                                                                    
CONTAINER ID   IMAGE                                   COMMAND                  CREATED              STATUS              PORTS                                                           NAMES   
e72d9908d7ff   morpheus-ms:24.03                       "/opt/conda/bin/tini…"   About a minute ago   Up About a minute                                                                   morpheus
-pdf-ingest-ms-morpheus-ms-run-e8210add8358
$ docker logs morpheus-pdf-ingest-ms-morpheus-ms-run-e8210add8358
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
```

## Testing with the upload_to_ingest_ms.py script

### Command-line Interface Options

- `--file_source`: Specifies the list of file sources or paths to be processed. This option can be repeated to add
  multiple file sources.
    - **Type**: String
    - **Multiple**: Yes
    - **Default**: Empty list
    - **Example Usage**: `--file_source path/to/file1.txt --file_source path/to/file2.pdf`

- `--redis-host`: Sets the DNS name for the Redis server.
    - **Type**: String
    - **Default**: `localhost`
    - **Example Usage**: `--redis-host my.redis.server`

- `--redis-port`: Specifies the port on which the Redis server is running.
    - **Type**: String
    - **Default**: `6379`
    - **Example Usage**: `--redis-port 6379`

- `--extract`: Enables the PDF text extraction task. This is a flag option.
    - **Type**: Flag (Boolean)
    - **Default**: False (not set)
    - **Example Usage**: `--extract`

- `--split`: Enables the text splitting task. This is a flag option.
    - **Type**: Flag (Boolean)
    - **Default**: False (not set)
    - **Example Usage**: `--split`

- `--extract_method`: Specifies the type(s) of extraction method to use. This option can be repeated to specify multiple
  extraction methods.
    - **Choices**: `pymupdf`, `haystack`, `unstructured_io`, `unstructured_service`
    - **Case Sensitive**: No
    - **Multiple**: Yes
    - **Example Usage**: `--extract_method pymupdf --extract_method haystack`

### Example document submission to the morpheus-ms service

Each of the following can be run from the host machine or from within the morpheus-ms container.

- Host: `python ./src/util/upload_to_ingest_ms.py ...`
- Container: `python upload_to_ingest_ms.py ...`

Submit a text file, with no splitting.

**Note:** You will receive a response containing a single document, which is the entire text file -- NO-OP.

```bash
python upload_to_ingest_ms.py --file_source ./data/plain_text/pg84.txt
2024-02-16 01:10:29,112 - __main__ - INFO - Read plain text content: The Project Gutenberg eBook of Frankenstein; Or, The Modern Prometheus
    
This ebook is for the u... (truncated)
2024-02-16 01:10:29,112 - __main__ - INFO - Content length: 438839
2024-02-16 01:10:29,118 - __main__ - INFO - Waiting for response on channel 'response_5536050b-388d-4ad1-b021-602940d3d9fd'...
2024-02-16 01:10:29,202 - __main__ - INFO - Received 1 documents from source: ./data/plain_text/pg84.txt
```

Submit a text file with a splitting task.

**Note:** You will receive back a response with multiple documents, each containing a chunk of the original text file.

```bash
python ./src/util/upload_to_ingest_ms.py --file_source ./data/plain_text/pg84.txt --split
2024-02-16 01:12:49,721 - __main__ - INFO - Read plain text content: The Project Gutenberg eBook of Frankenstein; Or, The Modern Prometheus
    
This ebook is for the u... (truncated)
2024-02-16 01:12:49,721 - __main__ - INFO - Content length: 438839
2024-02-16 01:12:49,727 - __main__ - INFO - Waiting for response on channel 'response_6e3ef2a6-6c22-4e48-b3c0-9b9275e12050'...
2024-02-16 01:12:49,888 - __main__ - INFO - Received 1445 documents from source: ./data/plain_text/pg84.txt
```

Submit a PDF file with a splitting task.

**Note:** This will not currently produce what you want as it will try to split one giant base64 encoded string. This
should be updated to decode the base64 string and then split it into chunks.

```bash
python ./src/util/upload_to_ingest_ms.py --file_source ./data/pdf_ingest_testing/the-un-security-council-handbook-by-scr-
1.pdf --split
2024-02-16 01:14:50,497 - __main__ - INFO - Encoded PDF content: JVBERi0xLjcNJeLjz9MNCjEwNTIgMCBvYmoNPDwvTGluZWFyaXplZCAxL0wgNzIyNzI2L08gMTA1NS9FIDI1MDk1L04gMTE2L1Qg... (truncated)
2024-02-16 01:14:50,497 - __main__ - INFO - Content length: 3213532
2024-02-16 01:14:50,522 - __main__ - INFO - Waiting for response on channel 'response_51818492-a539-480b-87ad-f1b26980c1c5'...
2024-02-16 01:14:51,037 - __main__ - INFO - Received 7142 documents from source: ./data/pdf_ingest_testing/the-un-security-council-handbook-by-scr-1.pdf
```

Submit a PDF file with splitting and extraction tasks.

**Note:** This works for pymupdf, haystack, and unstructured, if you have an API key.

**Note:** You can specify multiple extraction methods, but we currently won't do the right thing and produce two
sets of outputs. TODO.

```bash
python ./src/util/upload_to_ingest_ms.py --file_source ./data/pdf_ingest_testing/the-un-security-council-handbook-by-scr-1.pdf --split --extract --extract_method=haystack
2024-02-16 01:19:40,301 - __main__ - INFO - Encoded PDF content: JVBERi0xLjcNJeLjz9MNCjEwNTIgMCBvYmoNPDwvTGluZWFyaXplZCAxL0wgNzIyNzI2L08gMTA1NS9FIDI1MDk1L04gMTE2L1Qg... (truncated)
2024-02-16 01:19:40,301 - __main__ - INFO - Content length: 3213532
2024-02-16 01:19:40,324 - __main__ - INFO - Waiting for response on channel 'response_52dcd900-349a-4f2e-a886-1e1fcaf5f094'...
2024-02-16 01:19:41,906 - __main__ - INFO - Received 847 documents from source: ./data/pdf_ingest_testing/the-un-security-council-handbook-by-scr-1.pdf

python ./src/util/upload_to_ingest_ms.py --file_source ./data/pdf_ingest_testing/the-un-security-council-handbook-by-scr-1.pdf --split --extract --extract_method=pymupdf
2024-02-16 01:20:17,233 - __main__ - INFO - Encoded PDF content: JVBERi0xLjcNJeLjz9MNCjEwNTIgMCBvYmoNPDwvTGluZWFyaXplZCAxL0wgNzIyNzI2L08gMTA1NS9FIDI1MDk1L04gMTE2L1Qg... (truncated)
2024-02-16 01:20:17,233 - __main__ - INFO - Content length: 3213532
2024-02-16 01:20:17,256 - __main__ - INFO - Waiting for response on channel 'response_21e69092-14a7-4f2f-a531-7ee7b4e62ed8'...
2024-02-16 01:20:17,603 - __main__ - INFO - Received 819 documents from source: ./data/pdf_ingest_testing/the-un-security-council-handbook-by-scr-1.pdf
```

## Launch the Nemo Retriever pipeline with the morpheus-ms service

**Note:** if you have deployed morpheus-ms, redis, or triton services, you will need to stop them before starting the
Nemo Retriever pipeline.

**Note:** As of 9 Feb, 2024 you will need to pull the Devin's experimental Nemo Retrieval pipeline branch.
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

The Morpheus pipeline will run the pipeline defined in `[nemo_retriever]/pipelines/morpheus_extract_split.mustache`,
this will call out to the morpheus-ms to do pdf text extraction, chunking, and embedding, then return the results to the
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
> -d '{"name": "test_collection", "pipeline": "morpheus_extract_split"}'

{
  "collection": {
    "pipeline": "morpheus_extract_split",
    "name": "test_collection",
    "id": "33e0b745-585d-44c6-8db0-b03b841ea50b"
  }
}

export HAYSTACK_PATH_COLLECTION_ID=a962acb5-f1e7-4632-add7-1ca601063287
export MORPHEUS_EXTRACT_SPLIT_COLLECTION=33e0b745-585d-44c6-8db0-b03b841ea50b
```

## Example document upload to the indexing endpoint using the Nemo Retrieval upload script

### Command line interface options

- `-c`, `--collection`: Specifies the Collection ID to which files will be uploaded. This option is required.
    - **Type**: String
    - **Required**: Yes
    - **Example Usage**: `--collection "123ABC"`

- `-m`, `--metadata`: Allows specifying metadata for the upload in the format `key=value`. This option can be repeated
  to specify multiple metadata entries.
    - **Type**: Action (Append)
    - **Required**: No
    - **Example Usage**: `--metadata "author=John Doe" --metadata "year=2024"`

- `--filenames`: Specifies the filenames to upload. Multiple filenames can be provided.
    - **Type**: Nargs (Accepts one or more arguments)
    - **Required**: No
    - **Example Usage**: `--filenames file1.txt file2.pdf`

- `-s`, `--silent`: Enables silent mode, which minimizes the output of the script.
    - **Type**: Store True (Flag/Boolean)
    - **Required**: No
    - **Example Usage**: `--silent`

- `-n`, `--n_parallel`: Sets the number of parallel uploads. If not specified, defaults to 5.
    - **Type**: Integer
    - **Default**: 5
    - **Required**: No
    - **Example Usage**: `--n_parallel 10`

**Note:** You may want to run the upload command below multiple times as a warm-up; the first time the Triton service is
hit, it tends to take an unusually long time to respond.

Extract using pymupdf

```bash
time python script/quickstart/upload.py -c ${MORPHEUS_EXTRACT_SPLIT} \
  --metadata ingest::service=morpheus \
  --metadata ingest::extract_methods=pymupdf \
  --metadata ingest::extract=true \
  --metadata ingest::split=true \
  --silent --n_parallel=15 \
  --filenames '../morpheus-pdf-ingest-ms/data/pdf_ingest_testing/the-un-security-council
  -handbook-by-scr-1.pdf'
  
Processing 1 files...

real    0m2.576s
user    0m0.080s
sys     0m0.013s
```

Extract using haystack

```bash
time python script/quickstart/upload.py -c ${MORPHEUS_EXTRACT_SPLIT} \
  --metadata ingest::service=morpheus \
  --metadata ingest::extract_methods=haystack \
  --metadata ingest::extract=true \
  --metadata ingest::split=true \
  --silent --n_parallel=15 \
  --filenames '../morpheus-pdf-ingest-ms/data/pdf_ingest_testing/the-un-security-council
  -handbook-by-scr-1.pdf'
  
Processing 1 files...

real    0m3.804s
user    0m0.074s
sys     0m0.016s
```

Extract using Tika

```bash
time python script/quickstart/upload.py -c ${TIKA_TEST} --metadata 
ingest::service=tika --silent --n_parallel=15 --filenames '../morpheus-pdf-ingest-ms/data/pdf_ingest_testing/the-un-security-council-handbook-by-scr-1.pdf'
Processing 1 files...

real    0m3.070s
user    0m0.079s
sys     0m0.017s
```


