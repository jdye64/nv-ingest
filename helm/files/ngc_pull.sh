#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.


set -euo pipefail

# To ensure we actually have an NGC binary, switch to full path if default is used
if [ "$NGC_EXE" = "ngc" ]; then
  NGC_EXE=$(which ngc)
fi

# check if ngc cli is truly available at this point
if [ ! -x "$NGC_EXE" ]; then
  echo "ngc cli is not installed or available!"
  exit 1
fi

# download the model
directory="${STORE_MOUNT_PATH}/${NGC_MODEL_NAME}_v${NGC_MODEL_VERSION}"
echo "Directory is $directory"
ready_file="$directory/.ready"
lock_file="$directory/.lock"
mkdir -p "$directory"
touch "$lock_file"
{
  if flock -xn 200; then
    trap 'rm -f $lock_file' EXIT
    if [ ! -e "$ready_file" ]; then
      $NGC_EXE registry model download-version --dest "$STORE_MOUNT_PATH" "${NGC_CLI_ORG}/${NGC_CLI_TEAM}/${NGC_MODEL_NAME}:${NGC_MODEL_VERSION}"
      # decrypt the model - if needed (conditions met)
      if [ -n "${NGC_DECRYPT_KEY:+''}" ] && [ -f "$directory/${MODEL_NAME}.enc" ]; then
        echo "Decrypting $directory/${MODEL_NAME}.enc"
        # untar if necessary
        if [ -n "${TARFILE:+''}" ]; then
          echo "TARFILE enabled, unarchiving..."
          openssl enc -aes-256-cbc -d -pbkdf2 -in "$directory/${MODEL_NAME}.enc" -out "$directory/${MODEL_NAME}.tar" -k "${NGC_DECRYPT_KEY}"
          tar -xvf "$directory/${MODEL_NAME}.tar" -C "$STORE_MOUNT_PATH"
          rm "$directory/${MODEL_NAME}.tar"
        else
          openssl enc -aes-256-cbc -d -pbkdf2 -in "$directory/${MODEL_NAME}.enc" -out "$directory/${MODEL_NAME}" -k "${NGC_DECRYPT_KEY}"
        fi
        rm "$directory/${MODEL_NAME}.enc"
      else
        echo "No decryption key provided, or encrypted file found. Skipping decryption.";
        if [ -n "${TARFILE:+''}" ]; then
          echo "TARFILE enabled, unarchiving..."
          tar -xvf "$directory/${NGC_MODEL_VERSION}.tar.gz" -C "$STORE_MOUNT_PATH"
          rm "$directory/${NGC_MODEL_VERSION}.tar.gz"
        fi
      fi
      touch "$ready_file"
      echo "Done dowloading"
      flock -u 200
    else
      echo "Download was already complete"
    fi;
  else
    while [ ! -e "$ready_file" ]
    do
      echo "Did not get the download lock. Waiting for the pod holding the lock to download the files."
      sleep 1
    done;
    echo "Done waiting"
  fi
} 200>"$lock_file"
ls -la "$directory"
