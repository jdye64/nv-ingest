#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
source "${SCRIPT_DIR}/../scripts/common.sh"

declare -a ERRORS

# Function to add error messages
add_error() {
    ERRORS+=("$1")
}

# If IGNORE_GIT_DIFF is enabled, use all files
if [[ "${IGNORE_GIT_DIFF}" == "1" ]]; then
   PY_MODIFIED_FILES=$(find ./ -name '*' | grep -P "${PYTHON_FILE_REGEX}")
else
   # Get the list of modified files to check
   get_modified_files "${PYTHON_FILE_REGEX}" PY_MODIFIED_FILES
fi

# Run copyright fix
if [[ -z "${SKIP_COPYRIGHT}" ]]; then
   echo "Running copyright check..."

   if [[ "${IGNORE_GIT_DIFF}" == "1" ]]; then
      COPYRIGHT_OUTPUT=$(python3 ./ci/scripts/copyright.py --fix-all ./ 2>&1)
   else
      COPYRIGHT_OUTPUT=$(python3 ./ci/scripts/copyright.py --fix-all --git-modified-only ./ 2>&1)
   fi

   if [[ $? -ne 0 ]]; then
       add_error "Copyright check failed: ${COPYRIGHT_OUTPUT}"
   else
       echo "Copyright check passed successfully."
   fi
fi

# Run isort
if [[ -z "${SKIP_ISORT}" ]]; then
   echo "Running isort..."
   ISORT_OUTPUT=$(python3 -m isort --settings-file "${PY_CFG}" ${PY_MODIFIED_FILES[@]} 2>&1)
   if [[ $? -ne 0 ]]; then
       add_error "Isort failed: ${ISORT_OUTPUT}"
   else
      echo "Isort passed successfully."
   fi
fi

# Run black
if [[ -z "${SKIP_BLACK}" ]]; then
   echo "Running black..."
   BLACK_OUTPUT=$(python3 -m black ${PY_MODIFIED_FILES[@]} 2>&1)
   if [[ $? -ne 0 ]]; then
       add_error "Black formatting failed: ${BLACK_OUTPUT}"
   else
       echo "Black passed successfully."
   fi
fi

# Run yapf
# TODO(Devin): Decide on yapf vs black, yapf is just more configurable
SKIP_YAPF=1
if [[ "${SKIP_YAPF}" == "" ]]; then
   echo "Running yapf..."
   python3 -m yapf -i --style ${PY_CFG} -r ${PY_MODIFIED_FILES[@]}
fi

# Run flake8
if [[ -z "${SKIP_FLAKE8}" ]]; then
    echo "Running flake8..."
    FALKE8_OUTPUT=$(python3 -m flake8 ${PY_MODIFIED_FILES[@]} 2>&1)
    FLAKE8_EXIT_CODE=$?

    if [[ $FLAKE8_EXIT_CODE -ne 0 ]]; then
        echo "Flake8 found issues."
        ERRORS+=("Flake8 issues: ${FALKE8_OUTPUT}")
    else
        echo "Flake8 passed successfully."
    fi
fi

# Report errors
if [ ${#ERRORS[@]} -ne 0 ]; then
    echo "Errors occurred during the run:"
    for error in "${ERRORS[@]}"; do
        echo "${error}"
    done
    exit 1
else
    echo "All style cleanup tools finished successfully."
fi
