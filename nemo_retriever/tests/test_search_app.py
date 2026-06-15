# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from nemo_retriever.search.app import create_app
from nemo_retriever.search.config import SearchConfig
from nemo_retriever.search.services.hits import HitCache, build_agent_hit


@pytest.fixture
def config() -> SearchConfig:
    return SearchConfig(
        service_url="http://service.test",
        vectordb_url="http://vectordb.test",
        hit_cache_ttl_s=3600,
        hit_cache_max_searches=10,
    )


@pytest.fixture
def client(config: SearchConfig) -> TestClient:
    app = create_app(config)
    with TestClient(app) as c:
        yield c


def test_build_agent_hit_shapes_export_urls() -> None:
    hit = build_agent_hit(
        {
            "text": "Revenue grew 12%",
            "source": "report.pdf",
            "page_number": 3,
            "metadata": {"type": "text"},
            "_distance": 0.42,
        },
        search_id="abc123",
        rank=1,
    )
    assert hit.hit_id == "abc123:1"
    assert hit.export.text_url.endswith("format=text")
    assert hit.export.document_url.endswith("/document?download=1")
    assert hit.content_type == "text"
    assert hit.distance == pytest.approx(0.42)


def test_status_service_unreachable(client: TestClient) -> None:
    from nemo_retriever.search.services.service_client import ServiceUnavailableError

    with patch(
        "nemo_retriever.search.routers.search.ServiceClient.get_service_health",
        new=AsyncMock(side_effect=ServiceUnavailableError("offline")),
    ):
        resp = client.get("/api/v1/status")
    assert resp.status_code == 503
    assert "offline" in resp.json()["detail"].lower()


def test_status_ok(client: TestClient) -> None:
    with (
        patch(
            "nemo_retriever.search.routers.search.ServiceClient.get_service_health",
            new=AsyncMock(return_value={"status": "ok", "mode": "standalone"}),
        ),
        patch(
            "nemo_retriever.search.routers.search.ServiceClient.get_vectordb_health",
            new=AsyncMock(return_value={"total_rows": 42, "table_exists": True, "table": "nemo_retriever"}),
        ),
    ):
        resp = client.get("/api/v1/status")
    assert resp.status_code == 200
    body = resp.json()
    assert body["service_reachable"] is True
    assert body["total_rows"] == 42


def test_search_happy_path(client: TestClient) -> None:
    raw = {"results": [{"hits": [{"text": "hello world", "source": "a.pdf", "page_number": 1, "_distance": 0.1}]}]}
    with patch(
        "nemo_retriever.search.routers.search.ServiceClient.query",
        new=AsyncMock(return_value=raw),
    ):
        resp = client.post("/api/v1/search", json={"query": "hello", "top_k": 5})
    assert resp.status_code == 200
    body = resp.json()
    assert body["hit_count"] == 1
    assert body["hits"][0]["text"] == "hello world"


def test_search_empty_index(client: TestClient) -> None:
    with patch(
        "nemo_retriever.search.routers.search.ServiceClient.query",
        new=AsyncMock(side_effect=RuntimeError("No data has been ingested yet.")),
    ):
        resp = client.post("/api/v1/search", json={"query": "hello", "top_k": 5})
    assert resp.status_code == 422


def test_export_hit_text(client: TestClient) -> None:
    cache = HitCache()
    hit = build_agent_hit({"text": "chunk body", "source": "x.pdf"}, search_id="s1", rank=1)
    cache.store("s1", [hit])
    app = create_app(SearchConfig())
    app.state.hit_cache = cache
    with TestClient(app) as c:
        resp = c.get("/api/v1/hits/s1:1/export?format=text")
    assert resp.status_code == 200
    assert resp.text == "chunk body"


def test_export_hit_summary(client: TestClient) -> None:
    cache = HitCache()
    hit = build_agent_hit({"text": "chunk body", "source": "x.pdf", "page_number": 2}, search_id="s1", rank=1)
    cache.store("s1", [hit])
    app = create_app(SearchConfig())
    app.state.hit_cache = cache
    with TestClient(app) as c:
        resp = c.get("/api/v1/hits/s1:1/export?format=summary")
    assert resp.status_code == 200
    assert "chunk body" in resp.text
    assert "x.pdf" in resp.text


def test_export_hit_json(client: TestClient) -> None:
    cache = HitCache()
    hit = build_agent_hit({"text": "chunk body", "source": "x.pdf"}, search_id="s1", rank=1)
    cache.store("s1", [hit])
    app = create_app(SearchConfig())
    app.state.hit_cache = cache
    with TestClient(app) as c:
        resp = c.get("/api/v1/hits/s1:1/export?format=json")
    assert resp.status_code == 200
    assert resp.json()["hit_id"] == "s1:1"


def test_export_hit_missing(client: TestClient) -> None:
    resp = client.get("/api/v1/hits/missing:1/export?format=text")
    assert resp.status_code == 404


def test_download_hit_document_from_store(client: TestClient) -> None:
    from nemo_retriever.search.services.documents import DocumentStore

    cache = HitCache()
    store = DocumentStore()
    store.register_file("report.pdf", b"%PDF-1.4 sample", display_name="report.pdf")
    hit = build_agent_hit({"text": "chunk body", "source": "report.pdf"}, search_id="s1", rank=1)
    cache.store("s1", [hit])
    app = create_app(SearchConfig())
    app.state.hit_cache = cache
    app.state.document_store = store
    with TestClient(app) as c:
        resp = c.get("/api/v1/hits/s1:1/document?download=1")
    assert resp.status_code == 200
    assert resp.content == b"%PDF-1.4 sample"
    assert "report.pdf" in resp.headers.get("content-disposition", "")


def test_download_hit_document_local_path(client: TestClient, tmp_path) -> None:
    doc = tmp_path / "local.txt"
    doc.write_text("local file contents")
    cache = HitCache()
    hit = build_agent_hit({"text": "chunk", "source": str(doc)}, search_id="s1", rank=1)
    cache.store("s1", [hit])
    app = create_app(SearchConfig())
    app.state.hit_cache = cache
    with TestClient(app) as c:
        resp = c.get("/api/v1/hits/s1:1/document")
    assert resp.status_code == 200
    assert resp.text == "local file contents"


def test_download_hit_document_not_found(client: TestClient) -> None:
    cache = HitCache()
    hit = build_agent_hit({"text": "chunk", "source": "missing.pdf"}, search_id="s1", rank=1)
    cache.store("s1", [hit])
    app = create_app(SearchConfig())
    app.state.hit_cache = cache
    with TestClient(app) as c:
        resp = c.get("/api/v1/hits/s1:1/document")
    assert resp.status_code == 404


def test_settings_get_and_update(client: TestClient) -> None:
    resp = client.get("/api/v1/settings")
    assert resp.status_code == 200
    body = resp.json()
    assert body["service_url"] == "http://service.test"
    assert "placeholders" in body
    assert len(body["placeholders"]) >= 1

    resp = client.put(
        "/api/v1/settings",
        json={"service_url": "http://custom:9000", "default_top_k": 25},
    )
    assert resp.status_code == 200
    updated = resp.json()
    assert updated["service_url"] == "http://custom:9000"
    assert updated["default_top_k"] == 25


def test_settings_reset(client: TestClient) -> None:
    client.put("/api/v1/settings", json={"service_url": "http://changed:1"})
    resp = client.post("/api/v1/settings/reset")
    assert resp.status_code == 200
    assert resp.json()["service_url"] == "http://service.test"


def test_ingest_no_files(client: TestClient) -> None:
    with patch(
        "nemo_retriever.search.routers.ingest.ServiceClient.get_service_health",
        new=AsyncMock(return_value={"status": "ok"}),
    ):
        resp = client.post("/api/v1/ingest", files={})
    assert resp.status_code == 422


def test_ingest_proxy(client: TestClient, tmp_path) -> None:
    doc = tmp_path / "sample.txt"
    doc.write_text("hello")
    ingest_result = __import__(
        "nemo_retriever.search.models.responses", fromlist=["IngestResponse"]
    ).IngestResponse(
        job_id="job-1",
        documents_submitted=1,
        documents_succeeded=0,
        documents_failed=0,
        elapsed_s=0.5,
        status="processing",
    )
    with (
        patch(
            "nemo_retriever.search.routers.ingest.ServiceClient.get_service_health",
            new=AsyncMock(return_value={"status": "ok"}),
        ),
        patch(
            "nemo_retriever.search.routers.ingest.ingest_uploaded_bytes",
            new=AsyncMock(return_value=ingest_result),
        ),
    ):
        resp = client.post(
            "/api/v1/ingest",
            files={"files": ("sample.txt", doc.read_bytes(), "text/plain")},
        )
    assert resp.status_code == 200
    body = resp.json()
    assert body["job_id"] == "job-1"
    assert body["status"] == "processing"


def test_get_ingest_job_status(client: TestClient) -> None:
    job_status = __import__(
        "nemo_retriever.search.models.responses", fromlist=["IngestJobStatus"]
    ).IngestJobStatus(
        job_id="job-1",
        status="completed",
        expected_documents=1,
        counts={"completed": 1, "failed": 0},
    )
    with patch(
        "nemo_retriever.search.routers.ingest.get_ingest_job_status",
        new=AsyncMock(return_value=job_status),
    ):
        resp = client.get("/api/v1/ingest/jobs/job-1")
    assert resp.status_code == 200
    assert resp.json()["status"] == "completed"


def test_ingest_submit_only(client: TestClient, tmp_path) -> None:
    doc = tmp_path / "sample.pdf"
    doc.write_bytes(b"%PDF-1.4\n%%EOF\n")
    with (
        patch(
            "nemo_retriever.search.routers.ingest.ServiceClient.get_service_health",
            new=AsyncMock(return_value={"status": "ok"}),
        ),
        patch(
            "nemo_retriever.search.services.ingest.RetrieverServiceClient._create_job",
            new=AsyncMock(return_value="job-99"),
        ),
        patch(
            "nemo_retriever.search.services.ingest.RetrieverServiceClient._upload_one",
            new=AsyncMock(return_value={"document_id": "doc-1"}),
        ),
    ):
        resp = client.post(
            "/api/v1/ingest",
            files={"files": ("sample.pdf", doc.read_bytes(), "application/pdf")},
        )
    assert resp.status_code == 200
    body = resp.json()
    assert body["job_id"] == "job-99"
    assert body["status"] == "processing"
    assert body["documents_submitted"] == 1


def test_cli_module_imports() -> None:
    from nemo_retriever.search.cli import app as search_app

    assert search_app is not None


def test_lazy_subapp_registration() -> None:
    import importlib

    cli_main = importlib.import_module("nemo_retriever.adapters.cli.main")
    registered = {name for name, _, _ in cli_main._LAZY_SUBAPPS}
    assert "search" in registered
