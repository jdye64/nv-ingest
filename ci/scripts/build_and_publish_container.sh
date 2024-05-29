#!/bin/bash

# Function to display usage
usage() {
    echo "Usage: $0 --type <dev|release>"
    exit 1
}

# Get the directory of the current script
SCRIPT_DIR=$(dirname "$(realpath "$0")")

# Parse options
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --type)
            TYPE="$2"
            shift
            ;;
        *)
            usage
            ;;
    esac
    shift
done

# Validate input
if [[ -z "$TYPE" ]]; then
    usage
fi

# Determine the version from the underlying packages
get_version() {
    CLIENT_WHEEL_PATH="$SCRIPT_DIR/../../client/dist/*.whl"
    SERVICE_WHEEL_PATH="$SCRIPT_DIR/../../dist/*.whl"

    local client_wheel=$(ls $CLIENT_WHEEL_PATH | head -n 1)
    local service_wheel=$(ls $SERVICE_WHEEL_PATH | head -n 1)

    if [[ -z "$client_wheel" || -z "$service_wheel" ]]; then
        echo "Error: No wheel found for client or service"
        exit 1
    fi

    # Extract the version from the client wheel filename
    local client_version=$(basename "$client_wheel" | sed -n 's/^[^-]*-\([0-9.]*\(\.dev[0-9]*\)*\(\.post[0-9]*\)*\)-py3-none-any\.whl/\1/p')
    # Extract the version from the service wheel filename
    local service_version=$(basename "$service_wheel" | sed -n 's/^[^-]*-\([0-9.]*\(\.dev[0-9]*\)*\(\.post[0-9]*\)*\)-py3-none-any\.whl/\1/p')

    # Ensure both versions match
    if [[ "$client_version" != "$service_version" ]]; then
        echo "Error: Mismatched versions for client and service"
        exit 1
    fi

    echo $client_version
}

VERSION=$(get_version)

# Construct the tag based on the type and version
if [[ "$TYPE" == "dev" ]]; then
    TAG="${VERSION}"
elif [[ "$TYPE" == "release" ]]; then
    TAG="${VERSION}"
else
    echo "Invalid type: $TYPE"
    usage
fi

# Build and push the Docker image using Kaniko
/kaniko/executor --context $SCRIPT_DIR/../.. \
  --dockerfile $SCRIPT_DIR/../../Dockerfile \
  --destination $CI_REGISTRY_IMAGE:$TAG \
  --build-arg VERSION=$VERSION \
  --build-arg TYPE=$TYPE \
  --target runtime
