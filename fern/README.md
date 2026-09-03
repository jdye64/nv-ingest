# NeMo Retriever — Fern Docs

This directory holds the [Fern](https://buildwithfern.com) documentation site for
NeMo Retriever.

## Configuration reference

`docs/pages/configuration.mdx` is **generated** from the code — it is the single,
self-documenting reference for every NeMo Retriever configuration value, grouped
by what each setting changes (model selection, performance, throughput,
accuracy, general, security).

Do not edit `configuration.mdx` by hand. Regenerate it after changing any
configuration model or its category annotations:

```bash
retriever config docs --output fern/docs/pages/configuration.mdx --format fern
```

## Previewing locally

```bash
npm install -g fern-api
fern docs dev        # live preview
fern generate --docs # build/publish
```
