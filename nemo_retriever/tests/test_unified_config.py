# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the unified RetrieverServiceConfig system."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from nemo_retriever.common.params import ExtractParams
from nemo_retriever.config import (
    ConfigJustification,
    RetrieverServiceConfig,
    configured,
    get_config,
    list_configured,
    load_config,
    save_config,
    set_config,
)
from nemo_retriever.config.context import _CONFIG
from nemo_retriever.config.registry import clear_configured_registry
from nemo_retriever.config.service_models import LocalModelsConfig, NimEndpointsConfig
from nemo_retriever.service.services import pipeline_executor


@pytest.fixture(autouse=True)
def _reset_config_context():
    token = _CONFIG.set(None)
    clear_configured_registry()
    yield
    _CONFIG.reset(token)
    clear_configured_registry()
    _ = pipeline_executor.build_extract_params
    _ = pipeline_executor.build_embed_params
    _ = pipeline_executor.build_asr_params
    _ = pipeline_executor.build_caption_params


def test_load_config_sets_context_and_pipeline_defaults(tmp_path: Path):
    yaml_path = tmp_path / "retriever-config.yaml"
    yaml_path.write_text(
        yaml.safe_dump(
            {
                "mode": "standalone",
                "pipeline_defaults": {"extract": {"ocr_version": "v1"}},
            }
        )
    )

    config = load_config(config_path=str(yaml_path), print_tree=False)
    assert config.pipeline_defaults.extract.ocr_version == "v1"
    assert get_config() is config


def test_save_config_redacts_secrets(tmp_path: Path):
    config = RetrieverServiceConfig()
    config.nim_endpoints.api_key = "secret-token"
    set_config(config)

    out = tmp_path / "out.yaml"
    save_config(out)
    saved = yaml.safe_load(out.read_text())
    assert saved["nim_endpoints"]["api_key"] == "****"


def test_configured_injects_pipeline_defaults():
    config = RetrieverServiceConfig()
    config.pipeline_defaults.extract.ocr_version = "v1"
    set_config(config)

    params = pipeline_executor.build_extract_params(NimEndpointsConfig(), LocalModelsConfig())
    assert params.ocr_version == "v1"


def test_configured_explicit_arg_wins_over_yaml():
    config = RetrieverServiceConfig()
    config.pipeline_defaults.extract.ocr_version = "v1"
    set_config(config)

    explicit = ExtractParams(ocr_version="v2")
    params = pipeline_executor.build_extract_params(
        NimEndpointsConfig(),
        LocalModelsConfig(),
        extract=explicit,
    )
    assert params.ocr_version == "v2"


def test_configured_registry_metadata():
    entries = list_configured()
    extract_entry = next(entry for entry in entries if entry.section == "pipeline_defaults.extract")
    assert ConfigJustification.ACCURACY in extract_entry.justification
    assert extract_entry.rationale


def test_configured_requires_rationale_with_justification():
    with pytest.raises(ValueError, match="rationale"):

        @configured(
            section="pipeline_defaults.extract",
            model=ExtractParams,
            justification=ConfigJustification.LATENCY,
            rationale="",
        )
        def _demo(extract: ExtractParams | None = None) -> ExtractParams:
            return extract or ExtractParams()
