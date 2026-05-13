# retriever compare

Comparison utilities for retrieval / extraction runs. Subcommands are
registered conditionally based on which optional modules are installed:

- `retriever compare json` — diff two JSON artifacts (only if the
  `compare_json` module is importable).
- `retriever compare results` — diff two result bundles (only if the
  `compare_results` module is importable).

If neither shows up under `retriever compare --help`, the corresponding
modules aren't available in your install. The top-level `compare` app
silently skips any that fail to import.

If flags look stale, re-check `retriever compare <cmd> --help`.

## When to use this

- You ran the same pipeline twice (e.g. before and after a flag change) and
  want a structured diff of the outputs.
- You're investigating a regression between two harness runs (paired with
  [harness](#related)).

**Use a different command when:**

- You just want to inspect a single run → use the runtime metrics JSON or
  `scripts/inspect_lancedb.py` from the skill.
- You're aggregating sweep results → `retriever harness summary` /
  `retriever harness compare` is purpose-built for that.

## Canonical invocations

JSON file diff:

```bash
retriever compare json baseline.json new.json
```

Results-bundle diff:

```bash
retriever compare results runs/baseline/ runs/new/
```

(Exact flag set depends on the underlying module; run `--help` to see what
your install exposes.)

## Common failure modes

- **`No such command 'json'`** — the `compare_json` submodule wasn't
  importable on your install; either the extra is missing or the module
  is intentionally not shipped in this build. Skip the subcommand or use
  `retriever harness compare`.
- **Subcommand not listed** — same root cause; `compare/__main__.py`
  silently drops unimportable submodules.

## Related

- `retriever harness compare` — purpose-built for comparing harness run
  bundles (preferred over generic `compare`).
- [pipeline](pipeline.md) — produces the runtime metrics / detection
  summaries you'll typically be comparing.
