# Agent Instructions — NeMo Retriever

These standards apply to AI coding agents working in this repository (Cursor, Claude Code, and compatible hosts).

## Project Overview

NVIDIA NeMo Retriever Library (NRL) is the multimodal extraction and retrieval library formerly known in docs as NV-Ingest. Published customer documentation lives primarily under `docs/docs/extraction/` and builds with MkDocs (`docs/mkdocs.yml`). Library and Helm sources live under `nemo_retriever/`.

## Repository Map

| Path | Purpose |
|------|---------|
| `docs/docs/extraction/` | Published NRL extraction documentation (MkDocs) |
| `docs/mkdocs.yml` | Nav, redirects, and MkDocs config |
| `nemo_retriever/` | Library source, CLI, Helm chart, and in-repo README examples |
| `nemo_retriever/tests/` | Library tests |
| `.github/workflows/` | CI, including NRL docs publish workflows |

## Documentation

- Treat `docs/docs/extraction/` and related MkDocs pages as the source of truth for user-facing NRL documentation. Follow [`docs/AGENTS.md`](docs/AGENTS.md).
- Before completing a code change, determine whether it changes a **user-visible** surface. This includes a public API, CLI, configuration, Helm values or defaults, workflow, error message or error contract, supported file type, or other supported product behavior.
- When it does and the host supports subagents, start a documentation authoring subagent while the primary agent continues the implementation. Direct it to read `docs/AGENTS.md`, update the affected docs, and run validation. Give it the changed sources and user-visible impact.
- Reconcile the authoring subagent's documentation changes and validation evidence before completing the implementation. Include the required documentation in the same change when the repository workflow allows a combined PR. If policy requires a **docs-only** follow-up PR, open that PR in the same task and link it from the code PR.
- If the host cannot run subagents, read `docs/AGENTS.md` in the primary task, complete the documentation work, and run its documented validation. Do not omit required documentation because parallel execution is unavailable.
- Do not document defaults or behavior that `main` does not have yet.
- Documentation PRs that change published NRL prose must stay docs-scoped. Do not change `nemo_retriever/src/**`, tests, Helm chart behavior, lockfiles, or runtime CI env on a docs PR unless the user explicitly requests eng work.
- Verified product surfaces that often need docs updates: Python `create_ingestor` / `GraphIngestor` APIs, `retriever` CLI, Helm chart README and values, support matrix and NIM defaults, authentication and environment variables, error and troubleshoot guidance, and release notes.

### NVIDIA DORI Routing

Select the documentation path from current host capabilities.
Do not ask the user to classify themselves or store repository-scoped identity
state during a normal documentation task.

1. Check whether the current agent exposes `dori_handle` or `dori_route` and
   `dori_collections`.
   If the user explicitly asks not to use DORI, use the
   [Writing Style Guide](docs/AGENTS.md#writing-style-guide) instead.
2. When those tools are available, list the installed collections.
   - If a collection source contains `tech-docs/skill-library`, use DORI for
     task routing.
   - If the collection is missing, inaccessible, or cannot be verified,
     continue with the
     [Writing Style Guide](docs/AGENTS.md#writing-style-guide).
3. When the DORI tools are unavailable, continue with the Writing Style Guide.
   Do not inspect a shell-visible CLI, install software, or configure the host
   during a normal documentation task.
4. Use [NVIDIA DORI Setup](docs/DORI_SETUP.md) only when the user explicitly
   asks to install or configure DORI.

Capability detection does not approve installation or host configuration.
DORI unavailability must not block documentation work.

## Engineering Guardrails

- Prefer small, focused diffs that match existing style.
- Do not invent APIs, CLI flags, Helm keys, or defaults. Verify against checked-in source or tests.
- Never commit secrets, API keys, or credentials.
- Do not add lint, hooks, or CI from agent guidance alone. Those require a separately reviewed repository change.
- Do not create or modify `CLAUDE.md` as part of documentation-agent setup.

## Validation Shortcuts

| Change type | Validation |
|-------------|------------|
| Docs under `docs/` | From `docs/`: `python -m mkdocs build --strict --config-file mkdocs.yml` when the environment supports it |
| Library code | Run the targeted tests that cover the changed modules |
| Docs-only PR scope | `git diff --name-only upstream/main...HEAD` (or `origin/main...HEAD`) and confirm no runtime/out-of-scope paths |

## Cursor Cloud specific instructions

This repo is the **NVIDIA NeMo Retriever** monorepo. The primary product is the
`nemo_retriever` Python package (RAG ingestion + retrieval): a `retriever` CLI,
a FastAPI service, and a Ray-based batch pipeline. Storage defaults to embedded
LanceDB. Extraction/embedding normally run on NVIDIA NIMs (remote via
`NVIDIA_API_KEY`) or on local GPUs.

### Environment layout
- Python **3.12** (see `.python-version`). Dependencies are managed with **`uv`**
  (installed via `pip --user`; the astral.sh installer is blocked here).
- Dev deps are installed into a uv venv at **`/workspace/.venv`** by the startup
  update script (`uv pip install -e "nemo_retriever[all,dev]"`, matching the CI
  unit-test install in `.github/workflows/retriever-unit-tests.yml`).
- Activate with `source /workspace/.venv/bin/activate` (or prefix commands with
  `uv run`). `retriever` and `pytest` are then on `PATH`.

### Known environment constraints (important)
This VM has **no GPU**, **no `NVIDIA_API_KEY`/`NGC_API_KEY`**, and **`huggingface.co`
egress is blocked**. Consequences:
- Full PDF/Office ingest (page-elements, OCR, table-structure, embedding NIMs)
  and local model inference (`[local]` extra, vLLM) **cannot run** end-to-end.
- Integration tests are deselected by default (`addopts = -m 'not integration'`)
  and require `NVIDIA_API_KEY`.
- A few unit tests in `nemo_retriever/tests/test_ingest_interface.py` that fetch
  the `nvidia/llama-nemotron-embed-1b-v2` tokenizer over the network fail **only
  because `huggingface.co` is blocked** — not a code/setup problem. The rest of
  the suite passes (~2640 passed / ~100 skipped).

### Lint
Linters come from `.pre-commit-config.yaml` (black 25.9.0, flake8 7.3.0). Running
`pre-commit run` itself needs GitHub egress to fetch hook repos (blocked here), so
run the linters directly (the update script installs `black`/`flake8` into the venv):
```
black --check --line-length=120 nemo_retriever/src nemo_retriever/tests
flake8 --max-line-length=120 --extend-ignore=E203,E266,F403,F405,E402 nemo_retriever/src
```

### Tests
Matches CI:
```
PYTHONPATH=nemo_retriever/src python -m pytest nemo_retriever/tests -q
```
The full suite takes a few minutes.

### Run the app
- CLI: `retriever --help`, `retriever ingest ...`, `retriever query ...`.
- Service (FastAPI): `retriever service start --port 7670`; probe
  `http://localhost:7670/v1/health`, OpenAPI at `/openapi.json` (Swagger `/docs`).
  In standalone mode `/v1/query` proxies to a separate vectordb sidecar (disabled
  by default), so use the `retriever query` CLI for direct LanceDB queries.

### Offline hello-world (no GPU / API key / HF egress)
Use **sparse/FTS** index mode, which skips embedding entirely, on a plain-text
corpus (plain-text extraction needs no NIM):
```
retriever ingest <text-dir> --lancedb-uri <db> --table-name hello --index-mode sparse
retriever query "<question>" --lancedb-uri <db> --table-name hello --top-k 2
```
Caveat: the `.txt`/`.html` splitter (`common/modality/txt/split.py`) still loads
the `nvidia/llama-nemotron-embed-1b-v2` tokenizer from HF just to count tokens.
When HF egress is blocked, seed a **lossless byte-level** tokenizer into the HF
cache under that repo id + pinned revision (see
`models/hf_model_registry.py::HF_MODEL_REVISIONS`), then export
`HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` before running ingest. For
sparse/FTS the tokenizer only affects chunk boundaries, so any lossless
tokenizer works; a byte-level BPE with the 256 base-byte vocab and no merges
round-trips text exactly.
