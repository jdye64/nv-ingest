# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Agent memory: schema, filter compilation, and the AgenticIngestor surface."""

from __future__ import annotations

import hashlib
from datetime import datetime, timedelta, timezone

import pytest

from nemo_retriever.common.schemas.memory import (
    MemoryFilter,
    MemoryForgetRequest,
    MemoryRecord,
    compile_memory_filter,
)
from nemo_retriever.common.vdb.memory_schema import (
    MEMORY_RESULT_COLUMNS,
    build_memory_row,
    memory_schema,
    parse_memory_row,
)
from nemo_retriever.ingestor import create_ingestor
from nemo_retriever.ingestor.agentic_ingestor import AgenticIngestor
from nemo_retriever.memory.backends import LocalMemoryBackend, rank_memory_rows

_VECTOR_DIM = 16


class _HashEmbedder:
    """Deterministic embedder so recall is testable without a model."""

    def embed(self, texts, *, input_type: str = "passage") -> list[list[float]]:
        return [self._vector(text) for text in texts]

    def embed_one(self, text: str, *, input_type: str = "query") -> list[float]:
        return self._vector(text)

    @staticmethod
    def _vector(text: str) -> list[float]:
        digest = hashlib.sha256(text.lower().encode("utf-8")).digest()
        return [digest[index] / 255.0 for index in range(_VECTOR_DIM)]


def _backend(tmp_path) -> LocalMemoryBackend:
    backend = LocalMemoryBackend.__new__(LocalMemoryBackend)
    from nemo_retriever.common.vdb.memory_store import MemoryStore

    backend.namespace = "default"
    backend._embedder = _HashEmbedder()
    backend._store = MemoryStore(uri=str(tmp_path / "memory"), create_fts_index=False)
    return backend


@pytest.fixture()
def memory(tmp_path) -> AgenticIngestor:
    return AgenticIngestor(backend="local", memory_backend=_backend(tmp_path))


# ----------------------------------------------------------------------
# Schema
# ----------------------------------------------------------------------


def test_memory_schema_promotes_filterable_columns_out_of_json() -> None:
    schema = memory_schema(_VECTOR_DIM)
    names = set(schema.names)

    assert {"session_id", "memory_type", "importance", "tags"} <= names
    # Time filtering only works in the storage engine if these are real
    # timestamps rather than the ISO strings the document schema uses.
    for column in ("occurred_at", "ingested_at", "valid_from", "valid_to"):
        assert str(schema.field(column).type).startswith("timestamp")


def test_build_memory_row_defaults_occurred_at_to_ingest_time() -> None:
    row = build_memory_row(text="hello", embedding=[0.1] * _VECTOR_DIM)

    assert row["occurred_at"] == row["ingested_at"]
    assert row["valid_from"] == row["occurred_at"]
    assert row["valid_to"] is None
    assert row["memory_id"].startswith("mem_")


def test_build_memory_row_rejects_unknown_enum_values() -> None:
    with pytest.raises(ValueError, match="memory_type"):
        build_memory_row(text="x", embedding=[0.1], memory_type="procedural-ish")


def test_parse_memory_row_decodes_metadata_and_timestamps() -> None:
    row = build_memory_row(text="hello", embedding=[0.1], metadata={"k": "v"})

    parsed = parse_memory_row(row)

    assert parsed["metadata"] == {"k": "v"}
    assert isinstance(parsed["occurred_at"], str)
    assert "vector" not in parsed
    assert set(parsed) <= set(MEMORY_RESULT_COLUMNS)


# ----------------------------------------------------------------------
# Filter compilation
# ----------------------------------------------------------------------


def test_compile_memory_filter_escapes_quotes_in_values() -> None:
    where = compile_memory_filter(MemoryFilter(session_id="o'brien"), namespace="ns")

    assert "session_id = 'o''brien'" in where


def test_compile_memory_filter_always_scopes_to_a_namespace() -> None:
    where = compile_memory_filter(MemoryFilter(), namespace="tenant-a")

    assert "namespace = 'tenant-a'" in where


def test_compile_memory_filter_hides_superseded_by_default() -> None:
    assert "valid_to IS NULL" in compile_memory_filter(MemoryFilter(), namespace="ns")
    assert "valid_to IS NULL" not in compile_memory_filter(MemoryFilter(include_superseded=True), namespace="ns")


def test_compile_memory_filter_returns_none_when_unconstrained() -> None:
    assert compile_memory_filter(MemoryFilter(include_superseded=True)) is None


def test_memory_filter_rejects_an_inverted_time_window() -> None:
    now = datetime.now(timezone.utc)
    with pytest.raises(ValueError, match="occurred_after"):
        MemoryFilter(occurred_after=now, occurred_before=now - timedelta(hours=1))


# ----------------------------------------------------------------------
# Ranking
# ----------------------------------------------------------------------


def test_rank_memory_rows_prefers_recent_when_decay_is_enabled() -> None:
    now = datetime.now(timezone.utc)
    rows = [
        {
            "memory_id": "old",
            "text": "a",
            "memory_type": "episodic",
            "_distance": 0.10,
            "occurred_at": (now - timedelta(days=30)).isoformat(),
        },
        {"memory_id": "new", "text": "b", "memory_type": "episodic", "_distance": 0.12, "occurred_at": now.isoformat()},
    ]

    without_decay = rank_memory_rows(rows, top_k=2)
    with_decay = rank_memory_rows(rows, top_k=2, recency_halflife_seconds=86_400)

    assert [hit.memory_id for hit in without_decay] == ["old", "new"]
    assert [hit.memory_id for hit in with_decay] == ["new", "old"]


# ----------------------------------------------------------------------
# Ingestor surface
# ----------------------------------------------------------------------


def test_create_ingestor_builds_an_agentic_ingestor() -> None:
    ingestor = create_ingestor(run_mode="agentic", memory_backend="local", namespace="agents")

    assert isinstance(ingestor, AgenticIngestor)
    assert ingestor.RUN_MODE == "agentic"
    assert ingestor.namespace == "agents"


def test_capabilities_excludes_verbs_that_only_raise() -> None:
    capabilities = create_ingestor(run_mode="agentic").capabilities()

    assert {"files", "texts", "extract", "embed", "ingest", "metadata", "tags"} <= capabilities
    assert capabilities.isdisjoint({"dedup", "caption", "vdb_upload", "webhook", "store"})


def test_metadata_and_tags_are_stamped_on_every_record(memory: AgenticIngestor) -> None:
    memory.metadata(project="atlas").tags("triage")

    record = memory.remember("The build fails on the arm64 runner.", tags=["ci"])

    assert record.metadata == {"project": "atlas"}
    assert record.tags == ["triage", "ci"]


def test_metadata_raises_on_run_modes_that_would_drop_it() -> None:
    ingestor = create_ingestor(run_mode="inprocess")

    with pytest.raises(NotImplementedError, match="metadata"):
        ingestor.metadata(project="atlas")


def test_remember_queues_until_flush(memory: AgenticIngestor) -> None:
    memory.remember("The user prefers pytest over unittest.")

    assert memory.pending == 1
    result = memory.flush()
    assert result.written == 1
    assert memory.pending == 0


def test_session_stamps_defaults_and_flushes_on_exit(memory: AgenticIngestor) -> None:
    with memory.session("thread-1", user_id="ada") as session:
        record = session.remember("Deployed the release candidate to staging.")
        assert record.session_id == "thread-1"
        assert record.user_id == "ada"

    assert memory.pending == 0
    # Defaults are scoped to the block, not the ingestor.
    assert memory.remember("Unrelated note about the parser.").session_id is None


def test_recall_finds_a_stored_memory(memory: AgenticIngestor) -> None:
    memory.remember("The user prefers pytest over unittest.", memory_type="semantic")
    memory.remember("Deployed the release candidate to staging.")
    memory.flush()

    hits = memory.recall("The user prefers pytest over unittest.", top_k=1, strategy="dense")

    assert len(hits) == 1
    assert "pytest" in hits[0].text
    assert hits[0].memory_type == "semantic"


def test_recall_scoped_to_a_session_excludes_other_sessions(memory: AgenticIngestor) -> None:
    with memory.session("thread-a") as session:
        session.remember("Alpha deployment finished without incident.")
    with memory.session("thread-b") as session:
        session.remember("Beta deployment finished without incident.")

    hits = memory.recall("deployment finished", top_k=10, session_id="thread-a", strategy="dense")

    assert hits
    assert {hit.session_id for hit in hits} == {"thread-a"}


def test_timeline_replays_a_session_in_order(memory: AgenticIngestor) -> None:
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    memory.remember_many(
        [
            MemoryRecord(
                text=f"Step number {index} of the run.",
                session_id="thread-1",
                occurred_at=start + timedelta(minutes=index),
            )
            for index in reversed(range(3))
        ]
    )
    memory.flush()

    replayed = memory.timeline("thread-1")

    assert [hit.text for hit in replayed] == [
        "Step number 0 of the run.",
        "Step number 1 of the run.",
        "Step number 2 of the run.",
    ]


def test_timeline_requires_a_session(memory: AgenticIngestor) -> None:
    with pytest.raises(ValueError, match="session_id"):
        memory.timeline()


def test_soft_forget_hides_a_memory_but_keeps_the_row(memory: AgenticIngestor) -> None:
    record = memory.remember("An obsolete fact about the deploy target.", memory_type="semantic")
    memory.flush()
    memory_id = memory.recall(record.text, top_k=1, strategy="dense")[0].memory_id

    result = memory.forget(memory_id=memory_id)

    assert result.forgotten == 1
    assert result.hard is False
    assert memory.recall(record.text, top_k=5, strategy="dense") == []
    assert memory.stats().total == 1


def test_hard_forget_removes_the_row(memory: AgenticIngestor) -> None:
    memory.remember("A fact that must be erased entirely.", memory_type="semantic")
    memory.flush()
    memory_id = memory.recall("erased entirely", top_k=1, strategy="dense")[0].memory_id

    result = memory.forget(memory_id=memory_id, hard=True)

    assert result.hard is True
    assert memory.stats().total == 0


def test_forget_refuses_an_unconstrained_filter(memory: AgenticIngestor) -> None:
    memory.remember("Something worth keeping around for a while.")
    memory.flush()

    with pytest.raises(ValueError, match="refusing to match every memory"):
        memory.forget(filter=MemoryFilter(namespace="default"))


def test_forget_requires_a_target() -> None:
    with pytest.raises(ValueError, match="memory_id or filter"):
        MemoryForgetRequest()


def test_stats_counts_by_memory_type(memory: AgenticIngestor) -> None:
    memory.remember("An event that happened during the run.")
    memory.remember("A durable fact about the user.", memory_type="semantic")
    memory.flush()

    stats = memory.stats()

    assert stats.total == 2
    assert stats.by_type == {"episodic": 1, "semantic": 1}


# ----------------------------------------------------------------------
# Salience
# ----------------------------------------------------------------------


@pytest.mark.parametrize("text", ["ok", "Sure!", "Let me check the logs", "   "])
def test_default_salience_drops_boilerplate(text: str) -> None:
    from nemo_retriever.memory.salience import default_salience

    assert default_salience(text).keep is False


def test_default_salience_keeps_short_statements_of_preference() -> None:
    from nemo_retriever.memory.salience import default_salience

    verdict = default_salience("I prefer tabs", role="user")

    assert verdict.keep is True
    assert verdict.importance == pytest.approx(0.9)


def test_default_salience_ranks_tool_output_below_user_speech() -> None:
    from nemo_retriever.memory.salience import default_salience

    text = "The deployment pipeline completed all twelve stages successfully."
    user = default_salience(text, role="user")
    observation = default_salience(text, role="tool", event_type="observation")

    assert user.importance > observation.importance


def test_default_salience_truncates_a_long_dump() -> None:
    from nemo_retriever.memory.salience import MAX_SALIENT_CHARS, default_salience

    verdict = default_salience("x" * (MAX_SALIENT_CHARS + 500))

    assert verdict.keep is True
    assert "characters omitted" in verdict.text


# ----------------------------------------------------------------------
# ATIF import
# ----------------------------------------------------------------------


def _atif_trace() -> dict:
    return {
        "schema_version": "ATIF-v1.7",
        "session_id": "trace-session",
        "agent": {"name": "nemo-retriever-agentic", "model_name": "test-model"},
        "steps": [
            {
                "step_id": 1,
                "timestamp": "2026-01-01T00:00:00.000Z",
                "source": "user",
                "message": "Which extraction methods support scanned PDFs?",
            },
            {
                "step_id": 2,
                "timestamp": "2026-01-01T00:00:05.000Z",
                "source": "agent",
                "message": "ok",
                "tool_calls": [{"function": {"name": "retrieve", "arguments": {"query": "scanned pdf"}}}],
                "observation": {"results": [{"content": "Nemotron Parse handles scanned page images."}]},
            },
        ],
    }


def test_atif_import_preserves_session_and_event_times(memory: AgenticIngestor) -> None:
    from nemo_retriever.memory.atif_import import import_atif_trajectory

    result = import_atif_trajectory(memory, _atif_trace())

    assert result.written == 3
    replayed = memory.timeline("trace-session")
    assert [hit.event_type for hit in replayed] == ["message", "tool_call", "observation"]
    assert replayed[0].role == "user"
    assert replayed[0].occurred_at.startswith("2026-01-01T00:00:00")


def test_atif_import_drops_the_boilerplate_agent_message(memory: AgenticIngestor) -> None:
    from nemo_retriever.memory.atif_import import import_atif_trajectory

    import_atif_trajectory(memory, _atif_trace())

    assert all(hit.text != "ok" for hit in memory.timeline("trace-session"))


def test_atif_import_can_skip_observations(memory: AgenticIngestor) -> None:
    from nemo_retriever.memory.atif_import import import_atif_trajectory

    import_atif_trajectory(memory, _atif_trace(), include_observations=False)

    assert all(hit.event_type != "observation" for hit in memory.timeline("trace-session"))


# ----------------------------------------------------------------------
# MemoryTap
# ----------------------------------------------------------------------


def _chat_response(content: str) -> dict:
    return {"choices": [{"message": {"role": "assistant", "content": content}}]}


def test_memory_tap_captures_both_sides_of_a_turn(memory: AgenticIngestor) -> None:
    from nemo_retriever.memory.tap import MemoryTap

    tap = MemoryTap(
        memory,
        lambda **_: _chat_response("Nemotron Parse handles scanned page images."),
        session_id="tapped",
    )

    tap(messages=[{"role": "user", "content": "Which method reads scanned PDFs?"}])
    memory.flush()

    captured = memory.timeline("tapped")
    assert [hit.role for hit in captured] == ["user", "assistant"]


def test_memory_tap_returns_the_response_when_capture_fails(memory: AgenticIngestor) -> None:
    from nemo_retriever.memory.tap import MemoryTap

    def exploding_salience(text, *, role=None, event_type=None):
        raise RuntimeError("salience is broken")

    tap = MemoryTap(memory, lambda **_: _chat_response("still fine"), salience=exploding_salience)

    response = tap(messages=[{"role": "user", "content": "anything at all here"}])

    assert response["choices"][0]["message"]["content"] == "still fine"


# ----------------------------------------------------------------------
# Service-side scope isolation
# ----------------------------------------------------------------------


@pytest.fixture()
def memory_service(tmp_path):
    from nemo_retriever.service.services.memory_service import MemoryService

    embedder = _HashEmbedder()
    service = MemoryService.__new__(MemoryService)
    from nemo_retriever.common.vdb.memory_store import MemoryStore

    service._store = MemoryStore(uri=str(tmp_path / "memory"), create_fts_index=False)
    service._embed_fn = embedder.embed
    service._max_write_batch = 512
    import threading

    service._write_lock = threading.Lock()
    return service


def test_memory_service_isolates_scopes_sharing_a_namespace(memory_service) -> None:
    from nemo_retriever.common.schemas.memory import MemoryRecallRequest

    secret = "The production database password rotation is overdue."
    memory_service.remember([MemoryRecord(text=secret, namespace="default")], scope="tenant-a")

    request = MemoryRecallRequest(query=secret, top_k=10, strategy="dense")
    assert [hit.text for hit in memory_service.recall(request, scope="tenant-a")] == [secret]
    assert memory_service.recall(request, scope="tenant-b") == []


def test_memory_service_reports_the_callers_own_namespace(memory_service) -> None:
    from nemo_retriever.common.schemas.memory import MemoryRecallRequest

    text = "The retention window for audit logs is ninety days."
    memory_service.remember([MemoryRecord(text=text, namespace="audit")], scope="tenant-a")

    hits = memory_service.recall(
        MemoryRecallRequest(query=text, top_k=1, strategy="dense", filter=MemoryFilter(namespace="audit")),
        scope="tenant-a",
    )

    # The scope prefix is a storage detail and must not leak to the caller.
    assert hits[0].namespace == "audit"
    assert memory_service.stats(scope="tenant-a", namespace="audit").namespace == "audit"


def test_memory_service_rejects_an_oversized_write_batch(memory_service) -> None:
    memory_service._max_write_batch = 2
    records = [MemoryRecord(text=f"Event number {index} happened.") for index in range(3)]

    with pytest.raises(ValueError, match="the limit is 2"):
        memory_service.remember(records, scope="tenant-a")


def test_memory_service_rejects_an_untargeted_forget(memory_service) -> None:
    from nemo_retriever.common.schemas.memory import MemoryForgetRequest as ForgetRequest

    with pytest.raises(ValueError, match="refusing to match every memory"):
        memory_service.forget(ForgetRequest(filter=MemoryFilter(namespace="default")), scope="tenant-a")
