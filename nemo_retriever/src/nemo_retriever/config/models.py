# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Top-level RetrieverServiceConfig — service sections plus pipeline defaults."""

from __future__ import annotations

import logging

from pydantic import ConfigDict, Field, model_validator

from nemo_retriever.common.schemas.base import RichModel
from nemo_retriever.config.pipeline_defaults import PipelineDefaultsConfig
from nemo_retriever.config.service_models import (
    AuthConfig,
    GatewayConfig,
    LLMConfig,
    LocalModelsConfig,
    LoggingConfig,
    MCPConfig,
    NimEndpointsConfig,
    PipelineOverridesConfig,
    PipelinePoolConfig,
    ResourceLimitsConfig,
    ServerConfig,
    ServiceMode,
    VectorDbConfig,
    WorkQueueConfig,
)


class RetrieverServiceConfig(RichModel):
    """Unified configuration for all retriever run modes.

    Service topology is selected via ``mode``; SDK pipeline defaults live
    under ``pipeline_defaults`` and apply to inprocess, batch, and service
    unless overridden at call time or via ``PipelineSpec``.
    """

    model_config = ConfigDict(extra="ignore")

    mode: ServiceMode = "standalone"
    server: ServerConfig = Field(default_factory=ServerConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    nim_endpoints: NimEndpointsConfig = Field(default_factory=NimEndpointsConfig)
    local_models: LocalModelsConfig = Field(default_factory=LocalModelsConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    resources: ResourceLimitsConfig = Field(default_factory=ResourceLimitsConfig)
    auth: AuthConfig = Field(default_factory=AuthConfig)
    mcp: MCPConfig = Field(default_factory=MCPConfig)
    gateway: GatewayConfig = Field(default_factory=GatewayConfig)
    pipeline: PipelinePoolConfig = Field(default_factory=PipelinePoolConfig)
    work_queue: WorkQueueConfig = Field(default_factory=WorkQueueConfig)
    vectordb: VectorDbConfig = Field(default_factory=VectorDbConfig)
    pipeline_overrides: PipelineOverridesConfig = Field(default_factory=PipelineOverridesConfig)
    pipeline_defaults: PipelineDefaultsConfig = Field(default_factory=PipelineDefaultsConfig)

    @model_validator(mode="after")
    def _cap_process_pool_workers_for_local_models(self) -> "RetrieverServiceConfig":
        """Each process-pool worker loads the full HF stack — cap pool size."""
        if not self.local_models.enabled:
            return self
        cap = self.local_models.max_process_pool_workers
        updates: dict[str, int] = {}
        if self.pipeline.realtime_workers > cap:
            updates["realtime_workers"] = cap
        if self.pipeline.batch_workers > cap:
            updates["batch_workers"] = cap
        if updates:
            logging.getLogger(__name__).warning(
                "local_models.enabled: capping pipeline workers to %d per pool "
                "(was realtime=%d batch=%d). Raise local_models.max_process_pool_workers "
                "only if GPU memory allows num_workers × model stack.",
                cap,
                self.pipeline.realtime_workers,
                self.pipeline.batch_workers,
            )
            self.pipeline = self.pipeline.model_copy(update=updates)
        return self


# Backward-compatible alias used throughout service mode.
ServiceConfig = RetrieverServiceConfig
