# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os

LOCAL_RAY_OBJECT_STORE_MEMORY_FRACTION = 0.35
_OBJECT_STORE_FRACTION_ENV = "RAY_DEFAULT_OBJECT_STORE_MEMORY_PROPORTION"
_MIN_OBJECT_STORE_MEMORY_BYTES = 80 * 1024 * 1024
_DEV_SHM_SAFETY_FRACTION = 0.70


def _effective_ray_address(ray_address: str | None) -> str:
    address = str(ray_address or os.environ.get("RAY_ADDRESS", "")).strip()
    return address


def _starts_local_ray(ray_address: str | None, *, starts_local: bool | None = None) -> bool:
    if starts_local is not None:
        return starts_local
    address = _effective_ray_address(ray_address)
    return not address or address == "local"


def _object_store_fraction() -> float:
    raw = os.environ.get(_OBJECT_STORE_FRACTION_ENV)
    if raw:
        try:
            fraction = float(raw)
        except ValueError:
            fraction = LOCAL_RAY_OBJECT_STORE_MEMORY_FRACTION
    else:
        fraction = LOCAL_RAY_OBJECT_STORE_MEMORY_FRACTION
    return min(max(fraction, 0.01), 0.95)


def _system_memory_bytes() -> int | None:
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        pages = os.sysconf("SC_PHYS_PAGES")
    except (AttributeError, OSError, ValueError):
        return None
    return int(page_size) * int(pages)


def _dev_shm_available_bytes() -> int | None:
    try:
        stats = os.statvfs("/dev/shm")
    except OSError:
        return None
    return int(stats.f_bavail) * int(stats.f_frsize)


def _local_object_store_memory_bytes() -> int | None:
    system_memory = _system_memory_bytes()
    if system_memory is None:
        return None
    target = int(system_memory * _object_store_fraction())
    if dev_shm_available := _dev_shm_available_bytes():
        target = min(target, int(dev_shm_available * _DEV_SHM_SAFETY_FRACTION))
    if target < _MIN_OBJECT_STORE_MEMORY_BYTES:
        return None
    return target


def configure_local_ray_defaults(ray_address: str | None = None, *, starts_local: bool | None = None) -> None:
    """Set conservative Ray defaults before starting a local Ray runtime."""
    if not _starts_local_ray(ray_address, starts_local=starts_local):
        return
    os.environ.setdefault(_OBJECT_STORE_FRACTION_ENV, str(LOCAL_RAY_OBJECT_STORE_MEMORY_FRACTION))


def local_ray_init_kwargs(ray_address: str | None = None, *, starts_local: bool | None = None) -> dict[str, int]:
    """Return explicit ``ray.init`` kwargs for local Ray runtimes."""
    if not _starts_local_ray(ray_address, starts_local=starts_local):
        return {}
    configure_local_ray_defaults(ray_address, starts_local=True)
    object_store_memory = _local_object_store_memory_bytes()
    if object_store_memory is None:
        return {}
    return {"object_store_memory": object_store_memory}
