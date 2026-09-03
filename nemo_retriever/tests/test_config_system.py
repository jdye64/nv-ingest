# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the central configuration system (nemo_retriever.config)."""

from __future__ import annotations

import json

import pytest
import yaml

from nemo_retriever.config import (
    ConfigCategory,
    ConfigService,
    NeMoRetrieverConfig,
    build_catalog,
    catalog_as_dict,
    config_field,
    config_section,
    fields_by_category,
    generate_markdown,
)
from nemo_retriever.config.categories import (
    ConfigMeta,
    _reset_registry_for_tests,
    get_field_meta,
    get_section_meta,
    registered_sections,
)


# --------------------------------------------------------------------------
# Self-labeling primitives
# --------------------------------------------------------------------------


def test_config_field_records_metadata():
    from pydantic import BaseModel

    class Model(BaseModel):
        knob: int = config_field(
            5,
            category=ConfigCategory.THROUGHPUT,
            impact="Raises parallelism.",
            tuning_hint="Match CPU cores.",
            ge=1,
        )

    meta = get_field_meta(Model, "knob", Model.model_fields["knob"])
    assert meta.categories == (ConfigCategory.THROUGHPUT,)
    assert meta.impact == "Raises parallelism."
    assert meta.tuning_hint == "Match CPU cores."
    # Metadata is visible in JSON schema.
    schema = Model.model_json_schema()
    assert "nemo_config" in schema["properties"]["knob"]


def test_config_field_supports_multiple_categories():
    from pydantic import BaseModel

    class Model(BaseModel):
        knob: int = config_field(
            1, category=[ConfigCategory.ACCURACY, ConfigCategory.PERFORMANCE], impact="x"
        )

    meta = get_field_meta(Model, "knob", Model.model_fields["knob"])
    assert set(meta.categories) == {ConfigCategory.ACCURACY, ConfigCategory.PERFORMANCE}


def test_config_section_sets_default_category_and_registers():
    from pydantic import BaseModel

    before = len(registered_sections())

    @config_section(category=ConfigCategory.SECURITY, title="My Section")
    class Section(BaseModel):
        plain: str = "x"

    assert len(registered_sections()) == before + 1
    section = get_section_meta(Section)
    assert section is not None and section.category is ConfigCategory.SECURITY
    # A field without config_field inherits the section's default category.
    meta = get_field_meta(Section, "plain", Section.model_fields["plain"])
    assert meta.categories == (ConfigCategory.SECURITY,)


def test_annotate_fields_on_external_model():
    from pydantic import BaseModel

    from nemo_retriever.config.categories import annotate_fields

    class External(BaseModel):
        threads: int = 4

    annotate_fields(External, {"threads": ConfigMeta((ConfigCategory.THROUGHPUT,), "worker count")})
    meta = get_field_meta(External, "threads", External.model_fields["threads"])
    assert meta.categories == (ConfigCategory.THROUGHPUT,)
    assert meta.impact == "worker count"


# --------------------------------------------------------------------------
# Catalog / introspection
# --------------------------------------------------------------------------


def test_catalog_covers_service_and_ingestion():
    sections = build_catalog(NeMoRetrieverConfig)
    paths = {s.path for s in sections}
    assert "service.server" in paths
    assert "ingestion.text_chunk" in paths
    total = sum(len(s.fields) for s in sections)
    assert total > 100


def test_catalog_env_var_naming():
    sections = build_catalog(NeMoRetrieverConfig)
    port = next(f for s in sections for f in s.fields if f.path == "service.server.port")
    assert port.env_var == "NEMO_RETRIEVER_SERVICE__SERVER__PORT"


def test_every_category_has_settings():
    grouped = fields_by_category(build_catalog(NeMoRetrieverConfig))
    for category in ConfigCategory:
        assert grouped[category], f"no settings tagged {category.value}"


# --------------------------------------------------------------------------
# Layered loading precedence
# --------------------------------------------------------------------------


@pytest.fixture()
def config_file(tmp_path):
    path = tmp_path / "nemo-retriever.yaml"
    path.write_text(
        yaml.safe_dump(
            {"service": {"server": {"port": 9000}, "auth": {"api_token": "filesecret"}}}
        )
    )
    return path


def test_defaults_only():
    service = ConfigService.load(use_env=False)
    assert service.config.service.server.port == 7670


def test_file_layer(config_file):
    service = ConfigService.load(config_file, use_env=False)
    assert service.config.service.server.port == 9000
    assert service.config.service.auth.api_token == "filesecret"


def test_env_beats_file(config_file, monkeypatch):
    monkeypatch.setenv("NEMO_RETRIEVER_SERVICE__SERVER__PORT", "9100")
    service = ConfigService.load(config_file, use_env=True)
    assert service.config.service.server.port == 9100


def test_overrides_beat_env(config_file, monkeypatch):
    monkeypatch.setenv("NEMO_RETRIEVER_SERVICE__SERVER__PORT", "9100")
    service = ConfigService.load(config_file, use_env=True, overrides={"service": {"server": {"port": 9200}}})
    assert service.config.service.server.port == 9200


def test_remote_beats_env(config_file, monkeypatch):
    monkeypatch.setenv("NEMO_RETRIEVER_SERVICE__SERVER__PORT", "9100")
    from nemo_retriever.config.schema import build_config

    cfg = build_config(
        file_data={"service": {"server": {"port": 9000}}},
        remote_data={"service": {"server": {"port": 9300}}},
        use_env=True,
    )
    assert cfg.service.server.port == 9300


def test_discovery_via_env(tmp_path, monkeypatch):
    path = tmp_path / "custom.yaml"
    path.write_text(yaml.safe_dump({"service": {"server": {"port": 8123}}}))
    monkeypatch.setenv("NEMO_RETRIEVER_CONFIG_FILE", str(path))
    service = ConfigService.load(use_env=False)
    assert service.config.service.server.port == 8123


def test_missing_explicit_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        ConfigService.load(tmp_path / "does-not-exist.yaml")


# --------------------------------------------------------------------------
# Serialization / persistence
# --------------------------------------------------------------------------


def test_to_yaml_and_json_roundtrip():
    service = ConfigService.load(use_env=False)
    parsed_yaml = yaml.safe_load(service.to_yaml())
    parsed_json = json.loads(service.to_json())
    assert parsed_yaml == parsed_json
    assert parsed_yaml["service"]["server"]["port"] == 7670


def test_secret_redaction(config_file):
    service = ConfigService.load(config_file, use_env=False)
    redacted = service.to_dict(redact_secrets=True)
    assert redacted["service"]["auth"]["api_token"] == "***"
    clear = service.to_dict(redact_secrets=False)
    assert clear["service"]["auth"]["api_token"] == "filesecret"


def test_save_roundtrip(tmp_path, config_file):
    service = ConfigService.load(config_file, use_env=False)
    out = service.save(tmp_path / "out.yaml")
    assert out.is_file()
    reloaded = ConfigService.load(out, use_env=False)
    assert reloaded.config.service.server.port == 9000


def test_from_service_yaml(tmp_path):
    svc_yaml = tmp_path / "retriever-service.yaml"
    svc_yaml.write_text(yaml.safe_dump({"server": {"port": 7777}}))
    service = ConfigService.from_service_yaml(svc_yaml)
    assert service.config.service.server.port == 7777


# --------------------------------------------------------------------------
# Docs generation
# --------------------------------------------------------------------------


def test_generate_fern_docs():
    md = generate_markdown(fmt="fern")
    assert md.startswith("---\ntitle: Configuration Reference")
    assert "<CardGroup" in md
    # categories and env vars appear
    for category in ConfigCategory:
        assert category.title in md
    assert "NEMO_RETRIEVER_SERVICE__SERVER__PORT" in md


def test_generate_markdown_plain():
    md = generate_markdown(fmt="markdown")
    assert md.startswith("# Configuration Reference")
    assert "<CardGroup" not in md


def test_catalog_as_dict_shape():
    data = catalog_as_dict()
    assert "sections" in data
    assert any(f["path"] == "service.server.port" for s in data["sections"] for f in s["fields"])


# --------------------------------------------------------------------------
# Cluster config router
# --------------------------------------------------------------------------


@pytest.fixture()
def config_client():
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from nemo_retriever.service.config import ServiceConfig
    from nemo_retriever.service.routers import config as config_router

    # Reset the process-local store between tests.
    config_router._STORE = config_router._ClusterConfigStore()

    app = FastAPI()
    app.state.config = ServiceConfig()
    app.include_router(config_router.router, prefix="/v1")
    return TestClient(app)


def test_router_get_derives_from_service(config_client):
    resp = config_client.get("/v1/config")
    assert resp.status_code == 200
    body = resp.json()
    assert body["version"] == 0
    assert body["source"] == "running-service"
    assert body["config"]["service"]["server"]["port"] == 7670


def test_router_put_then_get(config_client):
    put = config_client.put("/v1/config", json={"config": {"service": {"server": {"port": 8500}}}})
    assert put.status_code == 200
    assert put.json()["version"] == 1

    get = config_client.get("/v1/config")
    body = get.json()
    assert body["version"] == 1
    assert body["source"] == "cluster-store"
    assert body["config"]["service"]["server"]["port"] == 8500


def test_router_put_rejects_invalid(config_client):
    resp = config_client.put("/v1/config", json={"config": {"service": {"server": {"port": "not-an-int"}}}})
    assert resp.status_code == 422


def test_router_catalog_and_schema(config_client):
    catalog = config_client.get("/v1/config/catalog")
    assert catalog.status_code == 200
    assert "sections" in catalog.json()

    schema = config_client.get("/v1/config/schema")
    assert schema.status_code == 200
    assert schema.json()["title"] == "NeMoRetrieverConfig"
