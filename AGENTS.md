# AGENTS.md

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
