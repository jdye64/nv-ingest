# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0


# pylint: disable=broad-except

class JobSubmissionException(Exception):
    """Indicates a problem with submitting the job to the nv-ingest service."""
    
    def __init__(self, message="Something went wrong"):
        self.message = message
        super().__init__(self.message)
