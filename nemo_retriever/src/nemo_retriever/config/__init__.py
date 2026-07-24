# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unified configuration for all nemo-retriever run modes."""

from __future__ import annotations

__all__ = [
    "AuthConfig",
    "ConfigJustification",
    "ConfiguredEntry",
    "GatewayConfig",
    "LLMConfig",
    "LocalAsrConfig",
    "LocalEmbedConfig",
    "LocalExtractConfig",
    "LocalModelsConfig",
    "LoggingConfig",
    "MCPConfig",
    "NimEndpointsConfig",
    "PipelineDefaultsConfig",
    "PipelineOverridesConfig",
    "PipelinePoolConfig",
    "ResourceLimitsConfig",
    "RetrieverServiceConfig",
    "ServerConfig",
    "ServiceConfig",
    "ServiceMode",
    "SinksConfig",
    "VectorDbConfig",
    "WorkQueueConfig",
    "config_context",
    "configured",
    "discover_config_path",
    "field_justification_metadata",
    "get_config",
    "get_config_dump",
    "get_config_section",
    "justified_field",
    "list_configured",
    "load_config",
    "print_config",
    "print_config_tree",
    "save_config",
    "set_config",
    "try_get_config",
]


def __getattr__(name: str):
    if name in {
        "AuthConfig",
        "GatewayConfig",
        "LLMConfig",
        "LocalAsrConfig",
        "LocalEmbedConfig",
        "LocalExtractConfig",
        "LocalModelsConfig",
        "LoggingConfig",
        "MCPConfig",
        "NimEndpointsConfig",
        "PipelineOverridesConfig",
        "PipelinePoolConfig",
        "ResourceLimitsConfig",
        "ServerConfig",
        "ServiceMode",
        "SinksConfig",
        "VectorDbConfig",
        "WorkQueueConfig",
    }:
        from nemo_retriever.config import service_models as sm

        return getattr(sm, name)
    if name in {"RetrieverServiceConfig", "ServiceConfig"}:
        from nemo_retriever.config.models import RetrieverServiceConfig, ServiceConfig

        return RetrieverServiceConfig if name == "RetrieverServiceConfig" else ServiceConfig
    if name == "PipelineDefaultsConfig":
        from nemo_retriever.config.pipeline_defaults import PipelineDefaultsConfig

        return PipelineDefaultsConfig
    if name in {"ConfigJustification", "justified_field", "field_justification_metadata"}:
        from nemo_retriever.config import justification as j

        return getattr(j, name)
    if name in {"ConfiguredEntry", "list_configured"}:
        from nemo_retriever.config import registry as r

        return getattr(r, name)
    if name in {"config_context", "get_config", "set_config", "try_get_config"}:
        from nemo_retriever.config import context as c

        return getattr(c, name)
    if name in {"configured", "get_config_section"}:
        from nemo_retriever.config import decorator as d

        return getattr(d, name)
    if name in {
        "discover_config_path",
        "get_config_dump",
        "load_config",
        "print_config",
        "print_config_tree",
        "save_config",
    }:
        from nemo_retriever.config import loader as loader_mod

        return getattr(loader_mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
