# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Service-mode configuration backed by ``retriever-service.yaml``.

This module re-exports the unified :mod:`nemo_retriever.config` package for
backward compatibility with existing service imports.
"""

from nemo_retriever.config import (
    AuthConfig,
    GatewayConfig,
    LLMConfig,
    LocalAsrConfig,
    LocalEmbedConfig,
    LocalExtractConfig,
    LocalModelsConfig,
    LoggingConfig,
    MCPConfig,
    NimEndpointsConfig,
    PipelineOverridesConfig,
    PipelinePoolConfig,
    ResourceLimitsConfig,
    RetrieverServiceConfig,
    ServerConfig,
    ServiceConfig,
    ServiceMode,
    SinksConfig,
    VectorDbConfig,
    WorkQueueConfig,
    load_config,
)

__all__ = [
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
    "RetrieverServiceConfig",
    "ServerConfig",
    "ServiceConfig",
    "ServiceMode",
    "SinksConfig",
    "VectorDbConfig",
    "WorkQueueConfig",
    "load_config",
]
