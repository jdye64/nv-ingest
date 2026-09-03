# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Central configuration system for NeMo Retriever.

This package is the single point of truth for configuring NeMo Retriever from
harnesses, external tools, web pages, and the service itself.

Public surface:

* :class:`NeMoRetrieverConfig` — the single Pydantic object aggregating all
  configuration.
* :class:`ConfigService` — load / serialize / persist / sync the configuration
  (local files, environment, or a remote cluster endpoint).
* :func:`config_field` / :func:`config_section` / :class:`ConfigCategory` —
  self-labeling primitives that tag settings by what they change.
* :func:`generate_config_docs` — generate the extensive Fern docs page.
"""

from __future__ import annotations

from nemo_retriever.config.annotations import apply_annotations
from nemo_retriever.config.catalog import (
    FieldDoc,
    SectionDoc,
    build_catalog,
    fields_by_category,
    iter_fields,
)
from nemo_retriever.config.categories import (
    ConfigCategory,
    ConfigMeta,
    SectionMeta,
    annotate_fields,
    config_field,
    config_section,
    get_field_meta,
    get_section_meta,
    registered_sections,
)
from nemo_retriever.config.docs import (
    catalog_as_dict,
    generate_config_docs,
    generate_markdown,
)
from nemo_retriever.config.schema import (
    ENV_NESTED_DELIMITER,
    ENV_PREFIX,
    IngestionDefaultsConfig,
    NeMoRetrieverConfig,
    build_config,
)
from nemo_retriever.config.service import (
    CONFIG_FILE_ENV,
    DEFAULT_CONFIG_FILENAME,
    REMOTE_CONFIG_PATH,
    ConfigService,
)

# Attach category metadata to externally-defined models (ServiceConfig, *Params)
# exactly once, when this package is first imported.
apply_annotations()

__all__ = [
    # Root object + loader
    "NeMoRetrieverConfig",
    "IngestionDefaultsConfig",
    "build_config",
    "ConfigService",
    # Self-labeling primitives
    "ConfigCategory",
    "ConfigMeta",
    "SectionMeta",
    "config_field",
    "config_section",
    "annotate_fields",
    "get_field_meta",
    "get_section_meta",
    "registered_sections",
    # Introspection
    "FieldDoc",
    "SectionDoc",
    "build_catalog",
    "fields_by_category",
    "iter_fields",
    # Docs
    "generate_config_docs",
    "generate_markdown",
    "catalog_as_dict",
    # Constants
    "ENV_PREFIX",
    "ENV_NESTED_DELIMITER",
    "CONFIG_FILE_ENV",
    "DEFAULT_CONFIG_FILENAME",
    "REMOTE_CONFIG_PATH",
]
