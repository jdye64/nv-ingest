#!/bin/bash

# Add model names and NGC paths here.
DOWNLOAD=(
    "deplot,nvidian/nemo-llm/nemo-retriever-deplot-image-to-text-triton-pytorch:11"
    "paddle,nvidian/nemo-llm/nemo-retriever-paddleocr-image-to-text-triton-pytorch:7"
    "cached,nvidian/nemo-llm/nemo-retriever-cached-image-to-text-triton-pytorch:15"
)
# Please also update the helm release for triton

update_triton_config() {
  local file="$1"

  # Use sed to replace the specific line
  sed -i '/kind: KIND_AUTO/c\
count: 2\
kind: KIND_GPU' "$file"
}

# Get script path: https://stackoverflow.com/a/4774063
SCRIPT_PATH="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
cd "$SCRIPT_PATH/../"

MD5_SCRIPT_PATH="scripts/compute_triton_models_md5sum.py"


# Create a temporary workspace.
MANIFEST_FILE="checksum_manifest.json"
TRITON_MODEL_DIR=model_repository
WORKSPACE=tmp
UNTAR_DIR=$WORKSPACE/untar
mkdir -p $UNTAR_DIR

# Download models from NGC.
for ROW in "${DOWNLOAD[@]}"; do
    IFS=',' read -r MODEL NGC_PATH <<< "$ROW"
    VERSION="${NGC_PATH##*:}"
    VERSION_DIR=$MODEL/$VERSION

    if [ -d "$VERSION_DIR" ]; then
        # Compute the md5sum of the local model version.
        COMPUTED_MD5SUM=$(python "$MD5_SCRIPT_PATH" --directory "$VERSION_DIR")
    else
        # This model does not exist locally so it must be downloaded.
        COMPUTED_MD5SUM="null"
    fi

    # Get the expected md5sum of the model version from the checksum manifest.
    EXPECTED_MD5SUM=$(jq -r --arg MODEL "$MODEL" --arg VERSION "$VERSION" '.[$MODEL][$VERSION]' "$MANIFEST_FILE")

    if [ "$COMPUTED_MD5SUM" == "$EXPECTED_MD5SUM" ]; then
        echo "Model $MODEL version $VERSION already exists, skipping download."
    else
        echo "Downloading $MODEL version $VERSION."
        ngc registry model --org ${NGC_ORG:-nvidian} download-version "${NGC_PATH}" --dest $WORKSPACE
    fi
done

# Untar models into a temporary directory and then move the version folder.
find $WORKSPACE -type f -name "*.tar.gz" | while read -r TAR_FILE; do
    tar -xzf "$TAR_FILE" -C $UNTAR_DIR

    NAME=$(ls $UNTAR_DIR | head -n 1)
    VERSION=$(ls $UNTAR_DIR/$NAME | head -n 1)
    NGC_VERSION_DIR="$UNTAR_DIR/$NAME/$VERSION"
    NGC_MODEL_CONFIG="$UNTAR_DIR/$NAME/config.pbtxt"
    CURRENT_MODEL_DIR="$NAME/"

    echo "Copying model $NAME version $VERSION."
    mkdir -p $CURRENT_MODEL_DIR
    cp -r "$NGC_VERSION_DIR" "$CURRENT_MODEL_DIR"
    cp -r "$NGC_MODEL_CONFIG" "$CURRENT_MODEL_DIR"
    update_triton_config "$CURRENT_MODEL_DIR/config.pbtxt"

    rm -rf $UNTAR_DIR/$NAME
done

# Final clean up.
rm -rf $WORKSPACE
