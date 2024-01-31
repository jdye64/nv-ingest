# Quickstart

We need to build the morpheus-ms here that gets loaded by the Nemo Retriver pipeline

docker compose -f docker-compose.yaml up retrieval-ms --build

# TODO(Devin): this has to be built manually, and locally until vdb_upload goes in. 
docker build --tag devin-morpheus-ms:manual_0.02 .

# Check if the morpheus-ms is running and responding correctly
python ./src/util/submit_to_morpheus_ms.py


# Create a collection associated with the morpheus-ms pipeline
curl http://localhost:1984/v1/collections \
>   -H 'Content-Type: application/json' \
>   -d '{"name": "devin_morpheus_collection", "pipeline": "test_morpheus_client"}'

export COLLECTION_ID=84cea217-bb06-4ea0-9871-0f9ead2ea882

python script/quickstart/upload.py -c ${COLLECTION_ID} data/small.pdf