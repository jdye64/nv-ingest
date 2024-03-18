# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.



export SCRIPT_DIR=${SCRIPT_DIR:-"$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"}

# The root to the Morpheus repo
export NV_INGEST_ROOT=${NV_INGEST_ROOT:-"$(realpath ${SCRIPT_DIR}/../..)"}

export PY_ROOT="${NV_INGEST_ROOT}"
export PY_CFG="${PY_ROOT}/setup.cfg"
export PY_DIRS="${PY_ROOT} ci/scripts"

# Determine the commits to compare against. If running in CI, these will be set. Otherwise, diff with main
export BASE_SHA=${CHANGE_TARGET:-${BASE_SHA:-$(${SCRIPT_DIR}/gitutils.py get_merge_target)}}
export COMMIT_SHA=${GIT_COMMIT:-${COMMIT_SHA:-HEAD}}

export PYTHON_FILE_REGEX='^(\.\/)?(?!\.|build|external).*\.(py|pyx|pxd)$'

# Use these options to skip any of the checks
export SKIP_COPYRIGHT=${SKIP_COPYRIGHT:-""}
export SKIP_IWYU=${SKIP_IWYU:-""}
export SKIP_ISORT=${SKIP_ISORT:-""}
export SKIP_YAPF=${SKIP_YAPF:-""}
export SKIP_BLACK=${SKIP_BLACK:-""}

# Set BUILD_DIR to use a different build folder
export BUILD_DIR=${BUILD_DIR:-"${NV_INGEST_ROOT}/build"}

# Speficy the clang-tools version to use. Default 16
export CLANG_TOOLS_VERSION=${CLANG_TOOLS_VERSION:-16}

# Returns the `branch-YY.MM` that is used as the base for merging
function get_base_branch() {
   local major_minor_version=$(git describe --tags | grep -o -E '[0-9][0-9]\.[0-9][0-9]')

   echo "branch-${major_minor_version}"
}

# Determine the merge base as the root to compare against. Optionally pass in a
# result variable otherwise the output is printed to stdout
function get_merge_base() {
   local __resultvar=$1
   # Resolve BASE_SHA to the SHA of origin/main if it's empty
   local base_sha=${BASE_SHA:-$(git rev-parse origin/main)}
   local commit_sha=${COMMIT_SHA:-HEAD}
   local result=$(git merge-base "$base_sha" "$commit_sha")

   if [[ "$__resultvar" ]]; then
      eval $__resultvar="'$result'"
   else
      echo "$result"
   fi
}

# Determine the changed files. First argument is the (optional) regex filter on
# the results. Second argument is the (optional) variable with the returned
# results. Otherwise the output is printed to stdout. Result is an array
function get_modified_files() {
   local  __resultvar=$2

   local GIT_DIFF_ARGS=${GIT_DIFF_ARGS:-"--name-only"}
   local GIT_DIFF_BASE=${GIT_DIFF_BASE:-$(get_merge_base)}

   # If invoked by a git-commit-hook, this will be populated
   local result=( $(git diff ${GIT_DIFF_ARGS} $(get_merge_base) | grep -P ${1:-'.*'}) )

   local files=()

   for i in "${result[@]}"; do
      if [[ -e "${i}" ]]; then
         files+=(${i})
      fi
   done

   if [[ "$__resultvar" ]]; then
      eval $__resultvar="( ${files[@]} )"
   else
      echo "${files[@]}"
   fi
}

# Determine a unified diff useful for clang-XXX-diff commands. First arg is
# optional file regex. Second argument is the (optional) variable with the
# returned results. Otherwise the output is printed to stdout
function get_unified_diff() {
   local  __resultvar=$2

   local result=$(git diff --no-color --relative -U0 $(get_merge_base) -- $(get_modified_files $1))

   if [[ "$__resultvar" ]]; then
      eval $__resultvar="'${result}'"
   else
      echo "${result}"
   fi
}

function get_num_proc() {
   NPROC_TOOL=`which nproc`
   NUM_PROC=${NUM_PROC:-`${NPROC_TOOL}`}
   echo "${NUM_PROC}"
}

function cleanup {
   # Restore the original directory
   popd &> /dev/null
}

trap cleanup EXIT

# Change directory to the repo root
pushd "${NV_INGEST_ROOT}" &> /dev/null
