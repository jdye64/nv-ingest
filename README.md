# Quickstart

We need to build the morpheus-ms here that gets loaded by the Nemo Retriever pipeline

# Build Morpheus 24.03 release container from source

Clone the Morpheus repository and checkout the 24.03 release tag.

```bash
git clone https://github.com/nv-morpheus/Morpheus.git
git checkout branch-24.03
git submodule update --init --recursive
```

Build the Morpheus-ms-base container

```bash
DOCKER_IMAGE_NAME=morpheus-ms-base \
DOCKER_IMAGE_TAG=24.03 \
bash docker/build_container_release.sh
```

Wait for a while.

Build the Morpheus-ms container -- This one copies in the morpheus_pdf_ingest library and sets up the entrypoint.

```bash
docker build --tag morpheus-ms:24.03 .
```

# Check if the morpheus-ms is running and responding correctly

```bash
# Nothing formal in this yet, add whatever PDFs you want to the data/pdf_ingest_testing directory
python ./src/util/submit_to_morpheus_ms.py --file_source ./data/pdf_ingest_testing/*
<--- Extracted output of your PDF will be printed here --->
```

# Create a collection associated with the morpheus-ms pipeline

curl http://localhost:1984/v1/collections \
> -H 'Content-Type: application/json' \
> -d '{"name": "devin_morpheus_collection", "pipeline": "test_morpheus_client"}'

export COLLECTION_ID=84cea217-bb06-4ea0-9871-0f9ead2ea882

python script/quickstart/upload.py -c ${COLLECTION_ID} data/small.pdf