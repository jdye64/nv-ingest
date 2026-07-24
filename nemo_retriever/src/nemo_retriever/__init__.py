# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Retriever application package."""

from __future__ import annotations

import sys
import warnings

# Suppress the pynvml deprecation FutureWarning before any submodule imports torch.
# The pynvml 13.x shim installs a PynvmlFinder via .pth at Python startup that warns
# on every `import pynvml`.  We both add a warnings filter and mark the finder as
# already warned so that spawned subprocesses (which get a fresh finder) stay quiet.
warnings.filterwarnings("ignore", message="The pynvml package is deprecated", category=FutureWarning)
for _finder in sys.meta_path:
    if type(_finder).__name__ == "PynvmlFinder":
        _finder.has_warned_pynvml = True
        break

from nemo_retriever.graph.retriever import retriever as _retriever_cls

__all__ = [
    "__version__",
    "ConfigJustification",
    "RetrieverServiceConfig",
    "create_ingestor",
    "get_config",
    "get_version",
    "get_version_info",
    "GraphIngestionError",
    "ingestor",
    "list_configured",
    "load_config",
    "print_config",
    "retriever",
    "RetrieverServiceCompatibilityError",
    "save_config",
]

retriever = _retriever_cls()


def __getattr__(name: str):
    if name == "create_ingestor":
        from nemo_retriever.ingestor import create_ingestor

        return create_ingestor
    if name in {"__version__", "get_version", "get_version_info"}:
        from nemo_retriever.version import __version__, get_version, get_version_info

        return {
            "__version__": __version__,
            "get_version": get_version,
            "get_version_info": get_version_info,
        }[name]
    if name == "ingestor":
        from nemo_retriever.ingestor import ingestor

        return ingestor
    if name == "GraphIngestionError":
        from nemo_retriever.ingestor.graph_ingestor import GraphIngestionError

        return GraphIngestionError
    if name == "RetrieverServiceCompatibilityError":
        from nemo_retriever.service.client import RetrieverServiceCompatibilityError

        return RetrieverServiceCompatibilityError
    if name == "ConfigJustification":
        from nemo_retriever.config import ConfigJustification

        return ConfigJustification
    if name == "RetrieverServiceConfig":
        from nemo_retriever.config import RetrieverServiceConfig

        return RetrieverServiceConfig
    if name == "get_config":
        from nemo_retriever.config import get_config

        return get_config
    if name == "load_config":
        from nemo_retriever.config import load_config

        return load_config
    if name == "print_config":
        from nemo_retriever.config import print_config

        return print_config
    if name == "save_config":
        from nemo_retriever.config import save_config

        return save_config
    if name == "list_configured":
        from nemo_retriever.config import list_configured

        return list_configured
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
