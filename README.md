<!--
SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: LicenseRef-NvidiaProprietary

NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
property and proprietary rights in and to this material, related
documentation and any modifications thereto. Any use, reproduction,
disclosure or distribution of this material and related documentation
without an express license agreement from NVIDIA CORPORATION or
its affiliates is strictly prohibited.
-->

# Quickstart

## Important Documents

- [LICENSE](./LICENSE)
- [CONTRIBUTING.md](./CONTRIBUTING.md)
- [Continuous Integration](./ci/README.md)
  - [Building Ingest Service and Client Packages](./ci/README.md#building-and-publishing-packages)

## Links

- [NV Ingest Repository](https://gitlab-master.nvidia.com/dl/ai-services/microservices/nv-ingest)
- [Morpheus Repository](https://github.com/nv-morpheus/Morpheus)
- [Nemo Retriever Repository](https://gitlab-master.nvidia.com/drobison/devin-nemo-retrieval-microservice-private)
  - [Ryan Angilly's Engineering Overview of Nemo Retriever](https://nvidia-my.sharepoint.com/personal/rangilly_nvidia_com/_layouts/15/stream.aspx?id=%2Fpersonal%2Frangilly%5Fnvidia%5Fcom%2FDocuments%2FRecordings%2FRyan%20Angilly%20presents%20how%20Retriever%20Services%20work%2D20240124%5F120001%2DMeeting%20Recording%2Emp4&referrer=StreamWebApp%2EWeb&referrerScenario=AddressBarCopied%2Eview&ga=1)

# Table of Contents

- [NV-Ingest: what it is and what it is not.](#nv-ingest-what-it-is-and-what-it-is-not)
  - [What it is not](#what-it-is-not)
  - [What it is](#what-it-is)
- [Quickstart](#quickstart)
  - [Downloading nightly containers and pip packages](#downloading-nightly-containers-and-pip-packages)
  - [Submitting documents to an existing ingest-service via CLI tool.](#submitting-documents-to-an-existing-ingest-service-via-cli-tool)
- [Submitting documents to an existing ingest-service](#submitting-documents-to-an-existing-ingest-service)
- [Building the nv-ingest-ms-runtime container](#building-the-nv-ingest-ms-runtime-container)
- [Utilities](#utilities)
  - [nv-ingest-cli](./client/README.md)
  - [gen_dataset.py](#gen_datasetpy)
  - [image_viewer.py](#image_viewerpy)

## NV-Ingest: what it is and what it is not.

NV-Ingest is a microservice consisting of a container implementing the document ingest pipeline, a message passing
service container (currently Redis), and optionally a Triton inference service container.

NV-Ingest can be deployed as a stand-alone service or as a dependency of a larger deployment, such as the Nemo Retriever
cluster.

### What it is not

A service that:

- Runs a static pipeline or fixed set of operations on every submitted document.
- Acts as a wrapper for any specific document parsing library.

### What it is

A service that:

- Accepts a JSON Job description, containing a document payload, and a set of ingestion tasks to perform on that
  payload.
- Allows the results of a Job to be retrieved; the result is a JSON dictionary containing a list of Metadata describing
  objects extracted from the base document, as well as processing annotations and timing/trace data.
- Supports PDF, Docx, and images.
- Is in the process of supporting content extraction from a number of base document types, including pptx and other
  document types.
- Supports multiple methods of extraction for each document type in order to balance trade-offs between throughput and
  accuracy. For example, for PDF documents we support extraction via MuPDF, ECLAIR, and Unstructured.io; additional
  extraction engines can and will be added as necessary to support downstream consumer requirements.
- Supports or is in the process of supporting various types of pre and post processing operations, including text
  splitting and chunking; image captioning, transform, and filtering; embedding generation, and image offloading to
  storage.



## Quickstart

### Downloading nightly containers and pip packages

Download the latest nightly container: `nv-ingest:[YEAR].[MONTH].[DAY].dev0`

You will need a personl access token to pull containers and packages,
see [here](https://gitlab-master.nvidia.com/help/user/profile/personal_access_tokens)

[Download container](https://gitlab-master.nvidia.com/dl/ai-services/microservices/nv-ingest/container_registry/60799)

[Install pip package](https://gitlab-master.nvidia.com/dl/ai-services/microservices/nv-ingest/-/packages)

### **Environment Configuration Variables**

- **`MESSAGE_CLIENT_HOST`**:

  - **Description**: Specifies the hostname or IP address of the message broker used for communication between
    services.
  - **Example**: `localhost`, `192.168.1.10`

- **`MESSAGE_CLIENT_PORT`**:

  - **Description**: Specifies the port number on which the message broker is listening.
  - **Example**: `6379`, `5672`

- **`CAPTION_CLASSIFIER_GRPC_TRITON`**:

  - **Description**: The endpoint where the caption classifier model is hosted using gRPC for communication. This is
    used to send requests for caption classification.
  - **Example**: `triton:8001`

- **`CAPTION_CLASSIFIER_MODEL_NAME`**:

  - **Description**: The name of the caption classifier model.
  - **Example**: `deberta_large`

- **`REDIS_MORPHEUS_TASK_QUEUE`**:

  - **Description**: The name of the task queue in Redis where tasks are stored and processed.
  - **Example**: `morpheus_task_queue`

- **`ECLAIR_TRITON_HOST`**:

  - **Description**: The hostname or IP address of the ECLAIR model service.
  - **Example**: `triton-eclair`

- **`ECLAIR_TRITON_PORT`**:

  - **Description**: The port number on which the ECLAIR model service is listening.
  - **Example**: `8001`

- **`OTEL_EXPORTER_OTLP_ENDPOINT`**:

  - **Description**: The endpoint for the OpenTelemetry exporter, used for sending telemetry data.
  - **Example**: `http://otel-collector:4317`

- **`INGEST_LOG_LEVEL`**:
  - **Description**: The log level for the ingest service, which controls the verbosity of the logging output.
  - **Example**: `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`

### Launch nv-ingest micro-service(s)

```bash
# Redis is our message broker for the ingest service, always required.
docker compose up -d redis

# Optional (Telemetry services)
# TODO: Add examples for telemetry services
docker compose up -d otel-collector prometheus grafana zipkin

# Optional (Triton) See below for Triton setup we need Triton for any model inference
# This is only needed for captioning or ECLAIR based extraction.
docker compose up -d triton

# Ingest service
docker compose up -d nv-ingest-ms-runtime
```

You should see something like this:

```bash
CONTAINER ID   IMAGE                                        COMMAND                 CREATED        STATUS                PORTS                              NAMES
6065c12d6034   .../nv-ingest:2024.6.3.dev0                 "/opt/conda/bin/tini…"   6 hours ago    Up 6 hours                                               nv-ingest-ms-runtime-1
c1f1f6b9cc8c   nvcr.io/nvidia/tritonserver:24.05-py3       "/opt/nvidia/nvidia_…"   5 days ago     Up 8 hours            0.0.0.0:8000-8002->8000-8002/tcp   devin-nv-ingest-triton-1
d277cf2c2703   redis/redis-stack                           "/entrypoint.sh"         2 weeks ago    Up 8 hours            0.0.0.0:6379->6379/tcp, 8001/tcp   devin-nv-ingest-redis-1
```

### Submitting documents to an existing ingest-service via CLI tool.

If you have installed the nv-ingest client package, you will have the nv-ingest-cli tool available in your environment.

```bash
nv-ingest-cli --help
Usage: nv-ingest-cli [OPTIONS]

Options:
  --batch_size INTEGER            Batch size (must be >= 1).  [default: 10]
  --doc PATH                      Add a new document to be processed (supports
                                  multiple).
  --dataset PATH                  Path to a dataset definition file.
  --client [REST|REDIS|KAFKA]     Client type.  [default: REDIS]
  --client_host TEXT              DNS name or URL for the endpoint.
  --client_port INTEGER           Port for the client endpoint.
  --client_kwargs TEXT            Additional arguments to pass to the client.
  --concurrency_n INTEGER         Number of inflight jobs to maintain at one
                                  time.  [default: 10]
  --document_processing_timeout INTEGER
                                  Timeout when waiting for a document to be
                                  processed.  [default: 10]
  --dry_run                       Perform a dry run without executing actions.
  --output_directory PATH         Output directory for results.
  --log_level [DEBUG|INFO|WARNING|ERROR|CRITICAL]
                                  Log level.  [default: INFO]
  --shuffle_dataset               Shuffle the dataset before processing.
                                  [default: True]
  --task TEXT                     Task definitions in JSON format, allowing multiple tasks to be configured by repeating this option.
                                  Each task must be specified with its type and corresponding options in the '[task_id]:{json_options}' format.

                                  Example:
                                    --task 'split:{"split_by":"page", "split_length":10}'
                                    --task 'extract:{"document_type":"pdf", "extract_text":true}'
                                    --task 'extract:{"document_type":"pdf", "extract_method":"eclair"}'
                                    --task 'extract:{"document_type":"pdf", "extract_method":"unstructured_io"}'
                                    --task 'extract:{"document_type":"docx", "extract_text":true, "extract_images":true}'
                                    --task 'store:{"content_type":"image", "store_method":"minio", "endpoint":"minio:9000"}'
                                    --task 'caption:{}'

                                  Tasks and Options:
                                  - split: Divides documents according to specified criteria.
                                      Options:
                                      - split_by (str): Criteria ('page', 'size', 'word', 'sentence'). No default.
                                      - split_length (int): Segment length. No default.
                                      - split_overlap (int): Segment overlap. No default.
                                      - max_character_length (int): Maximum segment character count. No default.
                                      - sentence_window_size (int): Sentence window size. No default.

                                  - extract: Extracts content from documents, customizable per document type.
                                      Can be specified multiple times for different 'document_type' values.
                                      Options:
                                      - document_type (str): Document format ('pdf', 'docx', 'pptx', 'html', 'xml', 'excel', 'csv', 'parquet'). Required.
                                      - extract_method (str): Extraction technique. Defaults are smartly chosen based on 'document_type'.
                                      - extract_text (bool): Enables text extraction. Default: False.
                                      - extract_images (bool): Enables image extraction. Default: False.
                                      - extract_tables (bool): Enables table extraction. Default: False.

                                  - store: Stores any images extracted from documents.
                                      Options:
                                      - content_type (str): Content type ('image', ). Required.
                                      - store_method (str): Storage type ('minio', ). Required.

                                  - caption: Attempts to extract captions for images extracted from documents. Note: this is not generative, but rather a
                                      simple extraction.
                                      Options:
                                        N/A

                                  - dedup: Idenfities and optionally filters duplicate images in extraction.
                                      Options:
                                        - content_type (str): Content type to deduplicate ('image')
                                        - filter (bool): When set to True, duplicates will be filtered, otherwise, an info message will be added.

                                  - filter: Idenfities and optionally filters images above or below scale thresholds.
                                      Options:
                                        - content_type (str): Content type to deduplicate ('image')
                                        - min_size: (Union[float, int]): Minimum allowable size of extracted image.
                                        - max_aspect_ratio: (Union[float, int]): Maximum allowable aspect ratio of extracted image.
                                        - min_aspect_ratio: (Union[float, int]): Minimum allowable aspect ratio of extracted image.
                                        - filter (bool): When set to True, duplicates will be filtered, otherwise, an info message will be added.

                                  Note: The 'extract_method' automatically selects the optimal method based on 'document_type' if not explicitly stated.
  --version                       Show version.
  --help                          Show this message and exit.

```

`test.pdf` is a simple PDF document with text and images.

![Simple PDF with Text and Images](./docs/images/test.pdf.png)

```shell
Submit ./data/test.pdf to the ingest service, and extract text and images from it using the pdfium method.
=====================================================================================================================

nv-ingest-cli \
  --doc ./data/test.pdf \
  --output_directory ./processed_docs \
  --task='extract:{"document_type": "pdf", "extract_method": "pdfium"}' \
  --client_host=localhost \
  --client_port=6379


Check to see that your document has been processed.
===================================================

ls ./processed_docs/*
processed_docs/image:
test.pdf.metadata.json

processed_docs/text:
test.pdf.metadata.json

```

Example of a processed document dataset

```shell
cat ./processed_docs/text/test.pdf.metadata.json
[{
  "document_type": "text",
  "metadata": {
    "content": "Here is one line of text. Here is another line of text. Here is an image.",
    "content_metadata": {
      "description": "Unstructured text from PDF document.",
      "hierarchy": {
        "block": -1,
        "line": -1,
        "page": -1,
        "page_count": 1,
        "span": -1
      },
      "page_number": -1,
      "type": "text"
    },
    "error_metadata": null,
    "image_metadata": null,
    "source_metadata": {
      "access_level": 1,
      "collection_id": "",
      "date_created": "2024-03-11T14:56:40.125063",
      "last_modified": "2024-03-11T14:56:40.125054",
      "partition_id": -1,
      "source_id": "test.pdf",
      "source_location": "",
      "source_name": "",
      "source_type": "PDF 1.4",
      "summary": ""
    },
    "text_metadata": {
      "keywords": "",
      "language": "en",
      "summary": "",
      "text_type": "document"
    }
  }
]]

$ cat ./processed_docs/image/test.pdf.metadata.json
[{
  "document_type": "image",
  "metadata": {
    "content": "<--- Base64 encoded image data --->",
    "content_metadata": {
      "description": "Image extracted from PDF document.",
      "hierarchy": {
        "block": 3,
        "line": -1,
        "page": 0,
        "page_count": 1,
        "span": -1
      },
      "page_number": 0,
      "type": "image"
    },
    "error_metadata": null,
    "image_metadata": {
      "caption": "",
      "image_location": [
        73.5,
        160.7775878906,
        541.5,
        472.7775878906
      ],
      "image_type": "png",
      "structured_image_type": "image_type_1",
      "text": ""
    },
    "source_metadata": {
      "access_level": 1,
      "collection_id": "",
      "date_created": "2024-03-11T14:56:40.125063",
      "last_modified": "2024-03-11T14:56:40.125054",
      "partition_id": -1,
      "source_id": "test.pdf",
      "source_location": "",
      "source_name": "",
      "source_type": "PDF 1.4",
      "summary": ""
    },
    "text_metadata": null
  }
}]
```

Check extracted images: [image_viewer.py](#image_viewerpy)

```shell
python src/util/image_viewer.py --file_path ./processed_docs/image/test.pdf.metadata.json
```

![Simple image viewer utility](./docs/images/image_viewer_example.png)

### Create Triton model

By default, NV-Ingest does not require Triton, but if you are testing ingestion embedding creation (currently disabled),
image caption extraction, or other tasks that require Triton, you will need to create a Triton container and or model's
for the tasks you are testing.

### (Optional) Setting up Triton Inference Server with DeBerta caption extraction model.

Using `'--task="caption:{}"` requires that there is a Triton server running with the [DeBerta Caption Extraction
Model](./triton_models/README.md#deberta-caption-selection-model) loaded.

### (Optional) Setting up Triton Inference Server with Eclair model

Using `--task="extract:{'document_type': 'pdf', extract_method'='eclair}"` requires that there is a Triton server
running with the [ECLAIR Document OCR Model](./triton_models/README.md#eclair-document-ocr-model) loaded.

## Utilities

### Example document submission to the nv-ingest-ms-runtime service

Each of the following can be run from the host machine or from within the nv-ingest-ms-runtime container.

- Host: `nv-ingest-cli ...`
- Container: `nv-ingest-cli ...`

Submit a text file, with no splitting.

**Note:** You will receive a response containing a single document, which is the entire text file -- This is mostly
a NO-OP, but the returned data will be wrapped in the appropriate metadata structure.

```bash
nv-ingest-cli \
  --doc ./data/test.pdf \
  --client_host=localhost \
  --client_port=6379
```

Submit a PDF file with only a splitting task.

```bash
nv-ingest-cli \
  --doc ./data/test.pdf \
  --output_directory ./processed_docs \
  --task='split' \
  --client_host=localhost \
  --client_port=6379
```

Submit a PDF file with splitting and extraction tasks.

**Note: (TODO)** This currently only works for pdfium, eclair, and Unstructured.io; haystack, Adobe, and LlamaParse
have existing workflows but have not been fully converted to use our unified metadata schema.

```bash
nv-ingest-cli \
  --doc ./data/test.pdf \
  --output_directory ./processed_docs \
  --task='extract:{"document_type": "pdf", "extract_method": "pdfium"}' \
  --task='extract:{"document_type": "docx", "extract_method": "python_docx"}' \
  --task='split' \
  --client_host=localhost \
  --client_port=6379

```

Submit a [dataset](#command-line-dataset-creation-with-enumeration-and-sampling) for processing

```shell
nv-ingest-cli \
  --dataset dataset.json \
  --output_directory ./processed_docs \
  --task='extract:{"document_type": "pdf", "extract_method": "pdfium"}' \
  --client_host=localhost \
  --client_port=6379

```

Submit a PDF file with extraction tasks and upload extracted images to MinIO.

```bash
nv-ingest-cli \
  --doc ./data/test.pdf \
  --output_directory ./processed_docs \
  --task='extract:{"document_type": "pdf", "extract_method": "pdfium"}' \
  --task='store:{"endpoint":"minio:9000","access_key":"minioadmin","secret_key":"minioadmin"}' \
  --client_host=localhost \
  --client_port=6379

```

### Command line dataset creation with enumeration and sampling

#### gen_dataset.py

```shell
python ./src/util/gen_dataset.py --source_directory=./data --size=1GB --sample pdf=60 --sample txt=40 --output_file \
  dataset.json --validate-output
```

This script samples files from a specified source directory according to defined proportions and a total size target. It
offers options for caching the file list, outputting a sampled file list, and validating the output.

### Options

- `--source_directory`: Specifies the path to the source directory where files will be scanned for sampling.

  - **Type**: String
  - **Required**: Yes
  - **Example**: `--source_directory ./data`

- `--size`: Defines the total size of files to sample. You can use suffixes (KB, MB, GB).

  - **Type**: String
  - **Required**: Yes
  - **Example**: `--size 500MB`

- `--sample`: Specifies file types and their proportions of the total size. Can be used multiple times for different
  file types.

  - **Type**: String
  - **Required**: No
  - **Multiple**: Yes
  - **Example**: `--sample pdf=40 --sample txt=60`

- `--cache_file`: If provided, caches the scanned file list as JSON at this path.

  - **Type**: String
  - **Required**: No
  - **Example**: `--cache_file ./file_list_cache.json`

- `--output_file`: If provided, outputs the list of sampled files as JSON at this path.

  - **Type**: String
  - **Required**: No
  - **Example**: `--output_file ./sampled_files.json`

- `--validate-output`: If set, the script re-validates the `output_file` JSON and logs total bytes for each file type.

  - **Type**: Flag
  - **Required**: No

- `--log-level`: Sets the logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'). Default is 'INFO'.

  - **Type**: Choice
  - **Required**: No
  - **Example**: `--log-level DEBUG`

- `--with-replacement`: Sample with replacement. Files can be selected multiple times.
  - **Type**: Flag
  - **Default**: True (if omitted, sampling will be with replacement)
  - **Usage Example**: `--with-replacement` to enable sampling with replacement or omit for default behavior.
    Use `--no-with-replacement` to disable it and sample without replacement.

The script performs a sampling process that respects the specified size and type proportions, generates a detailed file
list, and provides options for caching and validation to facilitate efficient data handling and integrity checking.

### Command line interface for the Image Viewer application, displays paginated images from a JSON file

viewer. Each image is resized for uniform display, and users can navigate through the images using "Next" and "Previous"
buttons.

#### image_viewer.py

- `--file_path`: Specifies the path to the JSON file containing the images. The JSON file should contain a list of
  objects, each with an `"image"` field that includes a base64 encoded string of the image data.
  - **Type**: String
  - **Required**: Yes
  - **Example Usage**:
    ```
    --file_path "/path/to/your/images.json"
    ```


## Third Party License Notice:
If configured to do so, this project will download and install additional third-party open source software projects. Review the license terms of these open source projects before use:

https://pypi.org/project/pdfservices-sdk/


## Contributing

We require that all contributors "sign-off" on their commits. This certifies that the contribution is your original work, or you have rights to submit it under the same license, or a compatible license.

 Any contribution which contains commits that are not Signed-Off will not be accepted.

To sign off on a commit you simply use the --signoff (or -s) option when committing your changes:
```
$ git commit -s -m "Add cool feature."
```

This will append the following to your commit message:
```
Signed-off-by: Your Name <your@email.com>
```

### Full text of the DCO:
```
  Developer Certificate of Origin
  Version 1.1
  
  Copyright (C) 2004, 2006 The Linux Foundation and its contributors.
  1 Letterman Drive
  Suite D4700
  San Francisco, CA, 94129
  
  Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
```
```
  Developer's Certificate of Origin 1.1
  
  By making a contribution to this project, I certify that:
  
  (a) The contribution was created in whole or in part by me and I have the right to submit it under the open source license indicated in the file; or
  
  (b) The contribution is based upon previous work that, to the best of my knowledge, is covered under an appropriate open source license and I have the right under that license to submit that work with modifications, whether created in whole or in part by me, under the same open source license (unless I am permitted to submit under a different license), as indicated in the file; or
  
  (c) The contribution was provided directly to me by some other person who certified (a), (b) or (c) and I have not modified it.
  
  (d) I understand and agree that this project and the contribution are public and that a record of the contribution (including all personal information I submit with it, including my sign-off) is maintained indefinitely and may be redistributed consistent with this project or the open source license(s) involved.
```