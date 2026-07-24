# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Process-local configuration context for ``@configured`` injection."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import TYPE_CHECKING, Iterator

if TYPE_CHECKING:
    from nemo_retriever.config.models import RetrieverServiceConfig

_CONFIG: ContextVar["RetrieverServiceConfig | None"] = ContextVar("nemo_retriever_config", default=None)


def get_config() -> "RetrieverServiceConfig":
    """Return the active :class:`RetrieverServiceConfig`.

    Raises:
        RuntimeError: When no configuration has been loaded into the context.
    """
    config = _CONFIG.get()
    if config is None:
        raise RuntimeError(
            "No RetrieverServiceConfig is active. Call load_config() / set_config() "
            "or use config_context() before invoking @configured functions."
        )
    return config


def try_get_config() -> "RetrieverServiceConfig | None":
    """Return the active config, or ``None`` when unset."""
    return _CONFIG.get()


def set_config(config: "RetrieverServiceConfig") -> None:
    """Install *config* as the process-local active configuration."""
    _CONFIG.set(config)


@contextmanager
def config_context(config: "RetrieverServiceConfig") -> Iterator["RetrieverServiceConfig"]:
    """Temporarily set the active configuration (for tests and scripts)."""
    token = _CONFIG.set(config)
    try:
        yield config
    finally:
        _CONFIG.reset(token)
