# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The single Pydantic source of truth for NeMo Retriever configuration.

:class:`NeMoRetrieverConfig` aggregates the two long-standing configuration
surfaces without rewriting them:

* ``service`` — the existing :class:`~nemo_retriever.service.config.ServiceConfig`
  (deployment topology, NIM endpoints, worker pools, auth, ...).
* ``ingestion`` — default ingest pipeline parameters (extraction, embedding,
  chunking, deduplication, vector store) drawn from the existing ``*Params``
  models.

The root is a ``pydantic-settings`` :class:`BaseSettings` so environment
variables layer in natively, and it wires up a source ordering that implements
the documented precedence:

``explicit overrides > remote cluster config > environment > file > defaults``

Loading/serialization/persistence is orchestrated by
:class:`nemo_retriever.config.service.ConfigService`; this module only defines
the shape and the source precedence.
"""

from __future__ import annotations

from contextvars import ContextVar
from typing import Any

from pydantic import Field
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)

from nemo_retriever.common.params.models import (
    DedupParams,
    EmbedParams,
    ExtractParams,
    LanceDbParams,
    TextChunkParams,
)
from nemo_retriever.common.schemas.base import RichModel
from nemo_retriever.config.categories import ConfigCategory, config_section
from nemo_retriever.service.config import ServiceConfig

ENV_PREFIX = "NEMO_RETRIEVER_"
ENV_NESTED_DELIMITER = "__"

# Data supplied per-load by ConfigService (file contents + remote payload +
# whether environment variables should participate). Stored in a ContextVar so
# the custom settings sources below can read it without global mutation.
_LOAD_CONTEXT: ContextVar[dict[str, Any]] = ContextVar("_config_load_context", default={})


@config_section(
    category=ConfigCategory.ACCURACY,
    title="Ingestion Defaults",
    description=(
        "Default ingest pipeline parameters applied when a request does not "
        "override them. These are the knobs that most affect extraction and "
        "retrieval quality."
    ),
)
class IngestionDefaultsConfig(RichModel):
    """Default pipeline parameters wrapping the existing ``*Params`` models."""

    extract: ExtractParams = Field(default_factory=ExtractParams)
    embed: EmbedParams = Field(default_factory=EmbedParams)
    text_chunk: TextChunkParams = Field(default_factory=TextChunkParams)
    dedup: DedupParams = Field(default_factory=DedupParams)
    vector_store: LanceDbParams = Field(default_factory=LanceDbParams)


class _MappingSource(PydanticBaseSettingsSource):
    """Settings source that yields a whole nested dict pulled from the load context."""

    def __init__(self, settings_cls, context_key: str) -> None:
        super().__init__(settings_cls)
        self._context_key = context_key

    def get_field_value(self, field, field_name):  # pragma: no cover - not used
        raise NotImplementedError

    def __call__(self) -> dict[str, Any]:
        data = _LOAD_CONTEXT.get().get(self._context_key)
        return dict(data) if isinstance(data, dict) else {}


@config_section(
    category=ConfigCategory.GENERAL,
    title="NeMo Retriever Configuration",
    description="Central source of truth for the entire NeMo Retriever project.",
)
class NeMoRetrieverConfig(BaseSettings):
    """Root configuration object for NeMo Retriever."""

    model_config = SettingsConfigDict(
        env_prefix=ENV_PREFIX,
        env_nested_delimiter=ENV_NESTED_DELIMITER,
        extra="ignore",
        validate_default=False,
    )

    service: ServiceConfig = Field(default_factory=ServiceConfig)
    ingestion: IngestionDefaultsConfig = Field(default_factory=IngestionDefaultsConfig)

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls,
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """Define layer precedence (first source wins).

        explicit overrides (init) > remote cluster > environment > file > defaults
        """
        use_env = _LOAD_CONTEXT.get().get("use_env", True)
        remote_source = _MappingSource(settings_cls, "remote_data")
        file_source = _MappingSource(settings_cls, "file_data")
        sources: list[PydanticBaseSettingsSource] = [init_settings, remote_source]
        if use_env:
            sources.append(env_settings)
        sources.append(file_source)
        return tuple(sources)


def build_config(
    *,
    file_data: dict[str, Any] | None = None,
    remote_data: dict[str, Any] | None = None,
    overrides: dict[str, Any] | None = None,
    use_env: bool = True,
) -> NeMoRetrieverConfig:
    """Construct a :class:`NeMoRetrieverConfig` from layered inputs.

    This is the single place that populates the load context and applies the
    documented precedence. :class:`ConfigService` is the public entry point;
    this helper keeps the source wiring in one testable function.
    """
    token = _LOAD_CONTEXT.set(
        {
            "file_data": file_data or {},
            "remote_data": remote_data or {},
            "use_env": use_env,
        }
    )
    try:
        return NeMoRetrieverConfig(**(overrides or {}))
    finally:
        _LOAD_CONTEXT.reset(token)


__all__ = [
    "NeMoRetrieverConfig",
    "IngestionDefaultsConfig",
    "build_config",
    "ENV_PREFIX",
    "ENV_NESTED_DELIMITER",
]
