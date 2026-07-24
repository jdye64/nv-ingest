# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Default pipeline stage params loaded from YAML for all run modes."""

from __future__ import annotations

from pydantic import ConfigDict, Field

from nemo_retriever.common.params.models import (
    ASRParams,
    CaptionParams,
    DedupParams,
    EmbedParams,
    ExtractParams,
    StoreParams,
)
from nemo_retriever.common.schemas.base import RichModel


class PipelineDefaultsConfig(RichModel):
    """SDK pipeline defaults merged with service NIM/local resolution.

    Values here apply to inprocess, batch, and service modes unless
    overridden at call time or via per-request ``PipelineSpec``.
    """

    model_config = ConfigDict(extra="forbid")

    extract: ExtractParams = Field(default_factory=ExtractParams)
    embed: EmbedParams = Field(default_factory=EmbedParams)
    asr: ASRParams = Field(default_factory=ASRParams)
    caption: CaptionParams = Field(default_factory=CaptionParams)
    store: StoreParams = Field(default_factory=StoreParams)
    dedup: DedupParams = Field(default_factory=DedupParams)


def get_pipeline_default_extract() -> ExtractParams:
    """Return configured extract defaults, or bare :class:`ExtractParams`."""
    from nemo_retriever.config.context import try_get_config

    config = try_get_config()
    if config is None:
        return ExtractParams()
    return config.pipeline_defaults.extract.model_copy()


def get_pipeline_default_embed() -> EmbedParams:
    """Return configured embed defaults, or bare :class:`EmbedParams`."""
    from nemo_retriever.config.context import try_get_config

    config = try_get_config()
    if config is None:
        return EmbedParams()
    return config.pipeline_defaults.embed.model_copy()
