#!/bin/bash

# Ensure the NGC CLI tool is installed and configured with your API key
# You can install the NGC CLI tool following NVIDIA's documentation

# NGC repository details
NGC_ORG="nvidian"
NGC_TEAM="nemo-llm"

# Default paths to the wheels
CLIENT_WHEEL_PATH="client/dist/*.whl"
SERVICE_WHEEL_PATH="dist/*.whl"

# Default resource names
CLIENT_RESOURCE_NAME="nv-ingest-python-client"
SERVICE_RESOURCE_NAME="nv-ingest-service"

# Parse arguments
DRY_RUN=false
USE_GITLAB=true
GITLAB_REPO_URL="https://gitlab-master.nvidia.com/api/v4/projects/${CI_PROJECT_ID}/packages/pypi"
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --dry-run) DRY_RUN=true ;;
        --use-gitlab) USE_GITLAB=true ;;
        --client-path) CLIENT_WHEEL_PATH="$2"; shift ;;
        --service-path) SERVICE_WHEEL_PATH="$2"; shift ;;
        --client-resource) CLIENT_RESOURCE_NAME="$2"; shift ;;
        --service-resource) SERVICE_RESOURCE_NAME="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# Function to extract version from wheel filename
extract_version() {
    local wheel_file=$1
    # Extract the version part of the filename, assuming format: package_name-version[-[dev|post]X]-py3-none-any.whl
    echo $(basename "$wheel_file" | sed -n 's/^[^-]*-\([0-9.]*\(\.dev[0-9]*\)*\(\.post[0-9]*\)*\)-py3-none-any\.whl/\1/p')
}

# Function to publish wheel to NGC
publish_wheel_ngc() {
    local wheel=$1
    local resource_name=$2
    local version=$(extract_version "$wheel")
    local command="ngc registry resource upload-version --source=\"$wheel\" --org=\"$NGC_ORG\" --team=\"$NGC_TEAM\" \"${NGC_ORG}/${NGC_TEAM}/${resource_name}:${version}\""

    if [ "$DRY_RUN" = true ]; then
        echo "[DRY RUN] $command"
    else
        eval $command
        if [ $? -ne 0 ]; then
            echo "Error: Failed to upload $wheel"
            exit 1
        fi
    fi
}

# Function to publish wheel to GitLab
publish_wheel_gitlab() {
    local wheel=$1
    local command="twine upload --repository-url $GITLAB_REPO_URL -u $CI_JOB_TOKEN -p $CI_JOB_TOKEN \"$wheel\""
    echo $command

    if [ "$DRY_RUN" = true ]; then
        echo "[DRY RUN] $command"
    else
        eval $command
        if [ $? -ne 0 ]; then
            echo "Error: Failed to upload $wheel to GitLab"
            exit 1
        fi
    fi
}

# Function to handle wheels
handle_wheels() {
    local wheel_path=$1
    local resource_name=$2
    local wheels=($wheel_path)

    if [ ${#wheels[@]} -eq 0 ]; then
        echo "Error: No wheels found at $wheel_path"
        exit 1
    elif [ ${#wheels[@]} -gt 1 ]; then
        echo "Error: More than one wheel found at $wheel_path. Upload request is ambiguous."
        exit 1
    else
        if [ "$USE_GITLAB" = true ]; then
            publish_wheel_gitlab "${wheels[0]}"
        else
            publish_wheel_ngc "${wheels[0]}" "$resource_name"
        fi
    fi
}

# Handle client wheels
handle_wheels "$CLIENT_WHEEL_PATH" "$CLIENT_RESOURCE_NAME"

# Handle service wheels
handle_wheels "$SERVICE_WHEEL_PATH" "$SERVICE_RESOURCE_NAME"
