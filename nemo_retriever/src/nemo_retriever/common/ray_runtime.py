# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os

LOCAL_RAY_OBJECT_STORE_MEMORY_FRACTION = "0.55"
_OBJECT_STORE_FRACTION_ENV = "RAY_DEFAULT_OBJECT_STORE_MEMORY_PROPORTION"


def configure_local_ray_defaults(ray_address: str | None = None) -> None:
    """Set conservative Ray defaults before starting a local Ray runtime."""
    address = str(ray_address or os.environ.get("RAY_ADDRESS", "")).strip()
    if address and address not in {"auto", "local"}:
        return
    os.environ.setdefault(_OBJECT_STORE_FRACTION_ENV, LOCAL_RAY_OBJECT_STORE_MEMORY_FRACTION)
