# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared helpers for unit tests."""

from __future__ import annotations

import asyncio


def _run(coro_or_result):
    """Run a coroutine synchronously in tests; pass through plain values."""
    if not asyncio.iscoroutine(coro_or_result):
        return coro_or_result
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro_or_result)
    finally:
        loop.close()
