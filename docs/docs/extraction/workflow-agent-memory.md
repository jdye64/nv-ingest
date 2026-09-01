# Workflow: Agent memory

Use this workflow when an agent must remember what happened across turns and
across sessions. The `agentic` run mode stores what an agent experienced and
recalls it later, instead of loading documents into a searchable collection.

`create_ingestor(run_mode="agentic")` returns an `AgenticIngestor`. That object
provides the memory verbs: `remember()`, `remember_many()`, `recall()`,
`timeline()`, `forget()`, `consolidate()`, `stats()`, `flush()`, `optimize()`,
and the `session()` context manager. The same operations are available as
Retriever service REST routes under `/v1/memory` and as Model Context Protocol
(MCP) tools.

Agent memory is a separate surface from [agentic
retrieval](workflow-agentic-retrieval.md). Agentic retrieval runs a Reason and
Act (ReAct) loop over your document table. Agent memory stores and recalls the
agent's own experience in a memory table with its own schema.

## Memory types { #memory-types }

Every memory carries a `memory_type`. The following two types matter to most
applications.

| `memory_type` | Purpose | Typical recall |
| --- | --- | --- |
| `episodic` | Time-ordered events such as messages, tool calls, and observations. This is the default. | Within one session, often through `timeline()`. |
| `semantic` | Durable facts distilled from episodes, such as a stated user preference. | Across sessions, through `recall()`. |

The value `procedural` also validates, but the library does not treat it
differently from the other types today.

A memory can also carry a `role` (`user`, `assistant`, `tool`, or `system`) and
an `event_type` (`message`, `tool_call`, `observation`, `fact`, `summary`, or
`document`). Any other value for these fields is rejected before storage.

## Choose a memory backend { #choose-a-memory-backend }

`AgenticIngestor` has a pluggable backend, so the deployment choice stays a
constructor argument.

| `memory_backend` | Behavior | Use it when |
| --- | --- | --- |
| `service` | Forwards every memory operation to a running Retriever service. This is the default. | Several agents or processes share one memory. |
| `local` | Embeds and stores in-process against the LanceDB directory in `memory_uri`. | You want memory with no server, or you are developing locally. |

Both backends apply a `namespace` to every write and, unless you override it,
every read. Recall never crosses namespaces implicitly.

## Store and recall memories in-process { #store-and-recall-memories-in-process }

The local backend needs no service. It embeds memory text with the same
embedding path that retrieval uses and writes to the LanceDB directory you
name in `memory_uri`.

```python
from nemo_retriever import create_ingestor

memory = create_ingestor(
    run_mode="agentic",
    memory_backend="local",
    memory_uri="./agent-memory",
    namespace="support-bot",
    agent_id="support-bot-v1",
)

with memory.session("thread-42", user_id="ada") as session:
    session.remember(
        "The user prefers pytest over unittest for new test files.",
        memory_type="semantic",
        role="user",
    )
    session.remember(
        "Ran the test suite; 3 failures in test_ingest.py.",
        role="assistant",
        event_type="observation",
    )

for hit in memory.recall("what testing framework does the user like?", top_k=5):
    print(hit.score, hit.memory_type, hit.text)
```

`session()` yields the same ingestor with the session defaults applied, so a
nested block does not have to pass identifiers to every call. The defaults are
restored when the block exits, even if the body raises.

## Store and recall memories through the service { #store-and-recall-memories-through-the-service }

The service backend sends the same verbs to a running Retriever service, so
every agent that shares the service shares one memory table. Pass the service
address, the bearer token, and the workspace scope.

```python
import os

from nemo_retriever import create_ingestor

memory = create_ingestor(
    run_mode="agentic",
    memory_backend="service",
    base_url="http://localhost:7670",
    api_key=os.environ["NEMO_RETRIEVER_API_TOKEN"],
    scope="workspace-123",
    namespace="support-bot",
)

with memory.session("thread-42", user_id="ada") as session:
    session.remember("The user's production cluster runs in us-east-1.", memory_type="semantic")

for hit in memory.recall("which region does the user deploy to?", top_k=5):
    print(hit.score, hit.namespace, hit.text)

print(memory.stats().total)
```

`optimize()` compacts storage on the local backend. On the service backend it
returns `False`, because compaction is a server-side maintenance concern.
Refer to [Enable agent memory on the service](#enable-agent-memory-on-the-service).

!!! note "Scopes cannot read each other's memories"

    In service mode, the authorized `X-NRL-Scope` is folded into the stored
    namespace. A caller cannot recall another scope's memories by asking for
    that scope's namespace. The scope prefix never appears in responses, so
    `namespace` in a hit is the namespace you asked for.

## Writes are queued and flushed in batches { #writes-are-queued-and-flushed-in-batches }

`remember()` enqueues a record and returns it. It does not embed or store the
record, and it does not block the agent's turn. The queued record has no
`memory_id` until the batch is flushed.

With `autoflush=True`, which is the default, the queue flushes on any of these
events:

- The queue reaches `flush_threshold` records. The default is 16.
- The oldest queued record reaches `flush_interval_seconds`, checked on the
  next write. The default is 30.0 seconds.
- Any of `recall()`, `timeline()`, `forget()`, `consolidate()`, `stats()`, or
  `optimize()` runs. Each flushes first so that it observes queued writes.
- A `session()` block exits.
- The ingestor is used as a context manager and that block exits.

Set `autoflush=False` to disable every automatic path and control writes with
`flush()`. Read `pending` for the number of queued records. If the backend
write fails, the batch is returned to the queue so a caller that handles the
error can retry rather than lose the agent's history.

`flush_threshold` and `flush_interval_seconds` are `AgenticIngestor`
constructor arguments. Construct the class directly when you need to change
them.

```python
from nemo_retriever.ingestor.agentic_ingestor import AgenticIngestor

with AgenticIngestor(
    backend="local",
    memory_uri="./agent-memory",
    namespace="support-bot",
    autoflush=False,
    flush_threshold=64,
) as memory:
    for line in ("Opened case 4821.", "Reproduced the failure on 26.08.1."):
        memory.remember(line, role="assistant")
    result = memory.flush()

print(result.written, result.memory_ids)
```

`AgenticIngestor` names the memory table `table_name`, while
`create_ingestor()` names the same field `memory_table_name`. The default
table is `nemo-retriever-memory`.

## Narrow recall with a structured filter { #narrow-recall-with-a-structured-filter }

Recall filters are described with the `MemoryFilter` model. Clients never send
SQL. The service compiles a validated filter into a predicate over typed
columns, so a caller cannot reach a column, table, or expression that the
model does not name. Recall always prefilters, so a scoped recall ranks only
the memories that match the filter.

The following fields are available on `MemoryFilter`.

| Field | Effect |
| --- | --- |
| `namespace` | Restricts to one namespace. The caller's namespace is applied even when this is omitted. |
| `session_id`, `agent_id`, `user_id` | Restricts to one session, agent, or user. |
| `memory_type` | Restricts to `episodic`, `semantic`, or `procedural`. |
| `memory_ids` | Restricts to an explicit list of identifiers. |
| `role`, `event_type` | Restricts to one role or event type. |
| `occurred_after`, `occurred_before` | Bounds `occurred_at`. `occurred_after` must be earlier than `occurred_before`. |
| `tags_any` | Matches memories carrying any of the listed tags. |
| `min_importance`, `min_confidence` | Applies a floor between 0.0 and 1.0. |
| `include_superseded` | When `True`, also returns memories that a `forget()` or a consolidation retired. The default is `False`. |

`recall()` also accepts `session_id` and `memory_type` directly as shorthands
for the common cases. When you pass both a filter and a shorthand, the
shorthand fills only the fields the filter left unset.

```python
from datetime import datetime, timedelta, timezone

from nemo_retriever.common.schemas.memory import MemoryFilter

recent = MemoryFilter(
    memory_type="semantic",
    user_id="ada",
    tags_any=["preference", "environment"],
    min_importance=0.6,
    occurred_after=datetime.now(timezone.utc) - timedelta(days=30),
)

for hit in memory.recall("what do we know about this user's setup?", top_k=10, filter=recent):
    print(hit.importance, hit.tags, hit.text)
```

## Choose a recall strategy { #choose-a-recall-strategy }

`recall()` accepts a `strategy` of `hybrid` (the default) or `dense`. Hybrid
recall combines the dense vector search with a full-text search over the
memory text. If the full-text index is missing or stale, hybrid recall falls
back to dense recall rather than failing the agent's turn.

Set `recency_halflife_seconds` to blend the retrieval score with exponential
recency decay, so older memories rank lower when several are equally relevant.
The value must be greater than 0. When you set it, recall widens its candidate
pool before reranking, so a slightly less similar but much fresher memory can
still enter the ranking.

```python
hits = memory.recall(
    "what is the user working on right now?",
    top_k=5,
    strategy="hybrid",
    recency_halflife_seconds=3600.0,
)
for hit in hits:
    print(hit.score, hit.occurred_at, hit.text)
```

Each `MemoryHit` carries `score`, where higher is better, and `distance`, the
native dense-vector distance, where lower is more similar. `score` is the
final ranking value after any recency weighting.

## Replay a session in order { #replay-a-session-in-order }

`timeline()` returns one session's memories in chronological order and does
not embed the query, so it costs no embedding call. Use it when you need what
came before or after something rather than what is most similar to a query.

```python
events = memory.timeline("thread-42", limit=50, ascending=True)
for event in events:
    print(event.occurred_at, event.role, event.text)
```

`timeline()` requires a session. It raises `ValueError` when you pass no
`session_id` and the ingestor has none. `limit` accepts 1 through 1,000 and
defaults to 100. Pass `since` and `until` to bound the window.

Capture layers that replay a transcript should set `occurred_at` explicitly on
each record, so the timeline reflects the conversation rather than the import.
When `occurred_at` is omitted, it defaults to the time the record was stored.

## Retire memories with forget { #retire-memories-with-forget }

`forget()` is a soft forget by default. It stamps `valid_to` on the matching
rows, so the memories stop appearing in recall while the rows remain
auditable. Pass `hard=True` to delete the rows outright.

```python
from nemo_retriever.common.schemas.memory import MemoryFilter

# Soft: the memory stops appearing in recall but the row remains.
soft = memory.forget(memory_id="mem_9f2c1b7a4d0e4f5a8c3b6d1e2f7a0b45")

# Hard: the matching rows are deleted.
hard = memory.forget(filter=MemoryFilter(session_id="thread-42"), hard=True)

print(soft.forgotten, hard.forgotten)
```

`forget()` requires either `memory_id` or `filter`. It also requires the
filter to narrow beyond a namespace: a namespace alone matches every memory
the caller owns, which is almost never what a forget meant. A filter that
names only a namespace is rejected. Set at least one of `memory_ids`,
`session_id`, `agent_id`, `user_id`, `memory_type`, `role`, `event_type`,
`tags_any`, a time bound, or an importance or confidence floor.

The result reports `matched`, `forgotten`, and `hard`.

## Consolidate episodes into durable facts { #consolidate-episodes-into-durable-facts }

A long session buries the few things worth remembering under hundreds of
turns. `consolidate()` reads a window of episodic memories, asks a language
model for the durable facts, and writes those back as `semantic` records.
Each new fact carries `source_memory_ids` pointing at the episodes it came
from.

Facts are never overwritten. A new fact that contradicts an earlier one
supersedes it by stamping `valid_to` on the older row, so the record of what
the agent once believed survives. Only identifiers that the model was actually
shown can be superseded.

```python
result = memory.consolidate("thread-42", max_episodes=200)
print(result.episodes_considered, result.facts_written, result.facts_superseded)
```

Consolidation costs one language-model call per session, so it is explicitly
triggered. Nothing in the library schedules it. Run it at the end of a session
rather than during one. `max_episodes` accepts 1 through 2,000 and defaults to
200. Pass `since` to bound the window. When you omit `session_id`,
consolidation falls back to a broad recall over the namespace's episodic
memories instead of replaying one timeline.

## Load reference documents into a memory namespace { #load-reference-documents-into-a-memory-namespace }

Agents usually need their own experience and their reference material in one
recall. `AgenticIngestor.ingest()` runs the ordinary extraction graph, then
writes the resulting chunks into the memory table instead of the document
table. Each chunk becomes one record with `memory_type="semantic"` and
`event_type="document"`, so reference material and remembered experience are
recalled through the same query.

```python
from nemo_retriever.ingestor.agentic_ingestor import AgenticIngestor

memory = AgenticIngestor(
    backend="local",
    memory_uri="./agent-memory",
    namespace="support-bot",
)

result = (
    memory.files(["data/multimodal_test.pdf"])
    .extract(extract_text=True, extract_tables=True)
    .metadata(source="runbook")
    .tags("reference")
    .ingest()
)
print(result.written)
```

`ingest()` runs extraction in-process and requires `backend="local"`. In
service mode it raises `NotImplementedError`; upload documents through the
service ingest API instead, then recall them from the same namespace. Embeddings
produced by the extraction graph are reused directly rather than recomputed.

Configure at least one input source with `files()`, `texts()`, or `buffers()`
before you call `ingest()`. Omitting input configuration raises `ValueError`.

## Attach metadata and tags to every write { #attach-metadata-and-tags-to-every-write }

`metadata(**fields)` and `tags(*values)` attach caller-supplied attributes to
every record the builder writes. Repeated `metadata()` calls merge. Repeated
`tags()` calls append, and duplicates collapse while preserving order.

These two methods exist on the shared ingestor interface, but only run modes
that advertise `SUPPORTS_SOURCE_METADATA` accept them. Today that is
`AgenticIngestor` only. `GraphIngestor` and `ServiceIngestor` raise
`NotImplementedError` rather than accept attributes they would drop before
storage.

```python
memory.metadata(product="retriever", environment="staging").tags("triage", "p1")
memory.remember("The staging cluster rejects tool calls from the answer NIM.")
```

Call `capabilities()` on any ingestor to get the set of verbs that instance
implements. A tool layer can build its surface from that set instead of
probing methods and catching `NotImplementedError`.

## Capture agent turns automatically { #capture-agent-turns-automatically }

An agent that must decide to call `remember()` on its own often does not.
`MemoryTap` wraps the OpenAI-compatible chat callable the agent already uses
and mirrors each turn into episodic memory on the way past, so capture is a
wiring change rather than a change to the agent's reasoning.

```python
from nemo_retriever.memory import MemoryTap

tap = MemoryTap(memory, completion_fn=my_chat_client, session_id="thread-42", user_id="ada")
response = tap(messages=[{"role": "user", "content": "Why did the ingest job fail?"}])
```

The wrapped callable is invoked unchanged and its result is returned
untouched. `MemoryTap` captures the newest user message and the assistant
message, and by default also captures the assistant's tool calls as separate
memories. Set `capture_prompt=False` when your application already writes user
turns, or `capture_tool_calls=False` to skip tool calls. Use `acall()` for a
callable that returns an awaitable.

Capture never blocks the agent. Turns are queued on the ingestor and embedded
by its batched flush. A capture failure is logged rather than raised, so an
agent does not fail its turn because remembering it failed.

### Filter boilerplate before embedding { #filter-boilerplate-before-embedding }

Capture layers see every message, tool call, and observation. Embedding all of
it adds cost and buries the useful memories under transcript noise, so recall
becomes less accurate as the agent runs longer. A salience filter runs first
and drops the low-signal text.

The default filter is a cheap heuristic with no model call. It drops empty
text, bare acknowledgements such as "Got it" or "Sounds good," filler openings
such as "Let me check," and any remaining text shorter than 24 characters. It
keeps text that states durable user intent even when that text is short, and it
stores a prefix of anything longer than 4,000 characters. It also sets
`importance`, weighting what the user said above what the agent said, and both
above whatever a tool returned.

The heuristic is a default, not a fixed policy. Pass
`salience=keep_everything` to store every non-empty candidate, which is useful
when replaying a full transcript, or pass your own callable for a
domain-specific rule.

```python
from nemo_retriever.memory.salience import keep_everything
from nemo_retriever.memory.tap import MemoryTap

tap = MemoryTap(memory, completion_fn=my_chat_client, salience=keep_everything)
```

## Import existing ATIF trajectories { #import-existing-atif-trajectories }

Agentic retrieval already writes Agent Trajectory Interchange Format (ATIF)
traces under `./agentic-traces`. Replaying those files is the fastest way to
fill a memory namespace with real agent traffic. The trace's own `session_id`
becomes the memory session and each step's timestamp becomes `occurred_at`,
so an imported trajectory replays through `timeline()` in its original order.

```python
from nemo_retriever.memory import import_atif_directory, import_atif_trajectory

one_trace = import_atif_trajectory(memory, "agentic-traces/trace-001.json", namespace="support-bot")
all_traces = import_atif_directory(memory, "agentic-traces", namespace="support-bot")

print(one_trace.written, all_traces.written)
```

Both functions flush before returning, so the memories are durable when the
call returns. `import_atif_directory()` logs and skips a malformed file rather
than aborting, so a partially corrupt trace directory still yields usable
memory. Pass `include_observations=False` to import messages and tool calls
only. Refer to [Result contract](workflow-agentic-retrieval.md#result-contract)
for what a trajectory contains.

## Use the memory MCP tools { #use-the-memory-mcp-tools }

The MCP tools give any MCP-capable host episodic and semantic memory over the
transport the service already exposes, with no SDK work.
`retriever service start` mounts the FastMCP HTTP endpoint at `/mcp` by
default.

The service registers the following memory tools.

| Tool | Purpose |
| --- | --- |
| `remember` | Store one memory. Accepts `text`, `memory_type`, `session_id`, `namespace`, `agent_id`, `user_id`, `role`, `tags`, `importance`, and `metadata`. |
| `recall` | Search stored memories by meaning, narrowed by `session_id`, `memory_type`, `agent_id`, `user_id`, `tags_any`, `min_importance`, or a time window, and optionally weighted by `recency_halflife_seconds`. |
| `timeline` | Replay one session's memories in the order they happened. |
| `forget` | Retire memories. Requires `memory_id`, `session_id`, or `tags_any`. Pass `hard=true` to delete rows. |
| `consolidate` | Distill a session's episodic memories into durable facts. |
| `memory_stats` | Report how many memories a namespace holds and the time span they cover. |

The memory tools are registered when `mcp.enable_memory_tools` is true, which
is the default. Set it to `false` in `retriever-service.yaml` to omit them.

```yaml
mcp:
  enabled: true
  path: /mcp
  enable_memory_tools: true
```

These tools call the `/v1/memory` routes, so they need an embedding backend on
the VectorDB service. Refer to
[Enable agent memory on the service](#enable-agent-memory-on-the-service).

For local stdio-based agents, run the MCP server as a shim that points at an
existing retriever service, as described in
[Query with MCP](workflow-agentic-retrieval.md#query-with-mcp).

## Call the memory REST API { #call-the-memory-rest-api }

The Retriever service gateway exposes the memory routes under `/v1/memory`.
The gateway authenticates the caller and forwards the authorized scope to the
VectorDB service, which owns the memory table.

| Method | Path | Purpose |
| --- | --- | --- |
| `POST` | `/v1/memory/remember` | Store a batch of memories. |
| `POST` | `/v1/memory/recall` | Rank stored memories against a natural-language query. |
| `POST` | `/v1/memory/timeline` | Replay one session's memories in chronological order. |
| `POST` | `/v1/memory/forget` | Retire memories by identifier or filter. |
| `POST` | `/v1/memory/consolidate` | Distill episodic memories into durable semantic facts. |
| `GET` | `/v1/memory/stats` | Return coarse counters for one memory namespace. |
| `POST` | `/v1/memory/optimize` | Compact the memory table after a burst of small writes. |

When service authentication is enabled, send `Authorization: Bearer <token>`
and send `X-NRL-Scope` with a workspace scope that the token is allowed to
use. Do not send `X-NRL-Internal-Token`; that header is for service-internal
hops.

Store a batch of memories:

```bash
curl -sS -X POST http://localhost:7670/v1/memory/remember \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer <token>' \
  -H 'X-NRL-Scope: workspace-123' \
  -d '{
        "records": [
          {
            "text": "The user prefers pytest over unittest for new test files.",
            "memory_type": "semantic",
            "namespace": "support-bot",
            "session_id": "thread-42",
            "user_id": "ada",
            "role": "user",
            "importance": 0.9,
            "tags": ["preference"]
          }
        ]
      }'
```

The response reports `memory_ids` and `written`. One request writes at most
512 records; a larger batch returns HTTP 422.

Recall with a structured filter and recency weighting:

```bash
curl -sS -X POST http://localhost:7670/v1/memory/recall \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer <token>' \
  -H 'X-NRL-Scope: workspace-123' \
  -d '{
        "query": "what testing framework does the user like?",
        "top_k": 5,
        "strategy": "hybrid",
        "recency_halflife_seconds": 86400,
        "filter": {
          "namespace": "support-bot",
          "memory_type": "semantic",
          "min_importance": 0.5
        }
      }'
```

The response is `{"hits": [...]}`. `/v1/memory/timeline` returns the same
envelope.

Retire a session's memories and read the namespace counters:

```bash
curl -sS -X POST http://localhost:7670/v1/memory/forget \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer <token>' \
  -H 'X-NRL-Scope: workspace-123' \
  -d '{"filter": {"namespace": "support-bot", "session_id": "thread-42"}, "hard": false}'

curl -sS 'http://localhost:7670/v1/memory/stats?namespace=support-bot' \
  -H 'Authorization: Bearer <token>' \
  -H 'X-NRL-Scope: workspace-123'
```

`RetrieverServiceClient` wraps the same contract with typed models. It provides
`remember()`, `recall()`, `memory_timeline()`, `forget()`,
`consolidate_memory()`, and `memory_stats()`, plus the `aremember()`,
`arecall()`, `amemory_timeline()`, `aforget()`, `aconsolidate_memory()`, and
`amemory_stats()` async variants.

### Memory error responses { #memory-error-responses }

- HTTP `404` when VectorDB is disabled in the service configuration, when the
  request reaches a realtime or batch worker pod instead of the gateway, or
  when agent memory is disabled on the VectorDB service.
- HTTP `501` when the VectorDB service has no embedding backend. Memory
  requires one for both writes and recall.
- HTTP `400` from `/v1/memory/consolidate` when agentic is not enabled on the
  VectorDB service. Consolidation needs a language model.
- HTTP `422` when a `forget` filter names only a namespace, or when a request
  body fails validation.
- HTTP `502` when the gateway cannot reach the VectorDB process.

## Enable agent memory on the service { #enable-agent-memory-on-the-service }

The memory table lives in the same LanceDB directory as the document table but
under its own name and schema, so memory writes never disturb document
retrieval and the two can be compacted on different schedules.

Agent memory starts with the VectorDB service when that process has an
embedding backend. Configure one with `--embed-endpoint` for a remote NIM or
`--local-embed` for in-pod embedding. Without an embedding backend, the memory
routes return HTTP 501.

Memory arrives as many small appends, and every append creates a new fragment
until compaction runs. The VectorDB service therefore compacts the memory
table on a schedule, every 900 seconds by default. `POST /v1/memory/optimize`
runs the same compaction on demand.

To turn agent memory off, rename the memory table, or change the compaction
interval, pass `memory_enabled`, `memory_table_name`, or
`memory_compaction_interval_seconds` to `create_vectordb_app()`. There are no
command-line flags for these values today.

`/v1/memory/consolidate` additionally requires agentic to be enabled on the
VectorDB service, because consolidation makes a language-model call. Refer to
[Enable agentic retrieval in the service](workflow-agentic-retrieval.md#enable-agentic-retrieval-in-the-service).

## Limitations { #limitations }

- `AgenticIngestor` does not implement `dedup()`, `caption()`, `store()`,
  `vdb_upload()`, or `webhook()`. Each raises `NotImplementedError`. The memory
  table is the vector store for this run mode, and duplicate facts are handled
  during consolidation rather than by a dedup stage.
- `AgenticIngestor.ingest()` requires `backend="local"`. Service-mode document
  loading goes through the service ingest API.
- `metadata()` and `tags()` are accepted only by the `agentic` run mode. Other
  run modes raise `NotImplementedError`.
- Nothing schedules consolidation. Trigger it explicitly, and expect one
  language-model call per pass.
- Agent memory has no CLI or Helm surface today. Configure it through the
  Python API, the REST routes, the MCP tools, or `create_vectordb_app()`.
- The memory table is created on the first write. `recall()` and `timeline()`
  raise until then, while `stats()` reports zeros.

## Related Topics { #related-topics }

- [Workflow: Agentic retrieval](workflow-agentic-retrieval.md)
- [Agentic retrieval (concept)](agentic-retrieval-concept.md)
- [Agent memory (Python API)](nemo-retriever-api-reference.md#agent-memory)
- [Collection management API](../reference/collection-management-api.md)
- [Authentication and API keys](api-keys.md)
- [Vector databases](vdbs.md)
