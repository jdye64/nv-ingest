# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Attach category metadata to configuration models defined elsewhere.

The existing ``ServiceConfig`` sub-sections and pipeline ``*Params`` classes
predate the centralized config system. Rather than rewrite them (which would be
invasive and risky), we attach :class:`~nemo_retriever.config.categories.ConfigMeta`
to them externally so they still show up, correctly categorized, in generated
docs and the ``config`` API.

:func:`apply_annotations` is idempotent and is called once when
``nemo_retriever.config`` is imported.
"""

from __future__ import annotations

from nemo_retriever.config.categories import (
    ConfigCategory,
    ConfigMeta,
    SectionMeta,
    annotate_fields,
)

_CAT = ConfigCategory
_applied = False


def _m(categories, impact, tuning_hint=None, stability="stable"):
    if not isinstance(categories, (list, tuple)):
        categories = [categories]
    return ConfigMeta(categories=tuple(categories), impact=impact, tuning_hint=tuning_hint, stability=stability)


def apply_annotations() -> None:
    """Annotate externally-defined config models. Safe to call repeatedly."""
    global _applied
    if _applied:
        return
    _applied = True
    _annotate_service_config()
    _annotate_pipeline_params()


def _annotate_service_config() -> None:
    from nemo_retriever.service import config as svc

    annotate_fields(
        svc.ServerConfig,
        {
            "host": _m(_CAT.GENERAL, "Network interface the HTTP service binds to."),
            "port": _m(_CAT.GENERAL, "TCP port the HTTP service listens on."),
        },
        section=SectionMeta(_CAT.GENERAL, title="Server", description="HTTP server bind address."),
    )

    annotate_fields(
        svc.LoggingConfig,
        {
            "level": _m(_CAT.GENERAL, "Root log verbosity.", "INFO in production; DEBUG when triaging."),
            "file": _m(_CAT.GENERAL, "Path of the rotating log file."),
            "format": _m(_CAT.GENERAL, "Python logging format string."),
        },
        section=SectionMeta(_CAT.GENERAL, title="Logging"),
    )

    annotate_fields(
        svc.NimEndpointsConfig,
        {
            "page_elements_invoke_url": _m(
                _CAT.MODEL_SELECTION,
                "Remote NIM used for page-element detection instead of a local GPU model.",
            ),
            "ocr_invoke_url": _m(_CAT.MODEL_SELECTION, "Remote OCR NIM endpoint."),
            "table_structure_invoke_url": _m(_CAT.MODEL_SELECTION, "Remote table-structure NIM endpoint."),
            "embed_invoke_url": _m(_CAT.MODEL_SELECTION, "Remote embedding NIM endpoint."),
            "embed_model_name": _m(
                _CAT.MODEL_SELECTION,
                "Embedding model SKU passed to the remote endpoint. Server-owned; "
                "changing it changes embedding quality and vector dimensionality.",
            ),
            "embed_model_provider_prefix": _m(
                _CAT.MODEL_SELECTION, "LiteLLM provider prefix for namespaced embed model IDs."
            ),
            "rerank_invoke_url": _m(_CAT.MODEL_SELECTION, "Remote reranker NIM endpoint."),
            "audio_grpc_endpoint": _m(_CAT.MODEL_SELECTION, "Remote Parakeet ASR gRPC endpoint."),
            "caption_invoke_url": _m(_CAT.MODEL_SELECTION, "Remote VLM endpoint fulfilling caption requests."),
            "caption_model_name": _m(_CAT.MODEL_SELECTION, "Caption VLM SKU passed to the remote endpoint."),
            "api_key": _m(_CAT.SECURITY, "Credential presented to remote NIM endpoints."),
        },
        section=SectionMeta(
            _CAT.MODEL_SELECTION,
            title="NIM Endpoints",
            description="Remote NIM microservice endpoints used instead of local GPU models.",
        ),
    )

    annotate_fields(
        svc.LocalModelsConfig,
        {
            "enabled": _m(
                _CAT.MODEL_SELECTION,
                "Load Nemotron Hugging Face weights inside the worker pod instead of calling remote NIMs.",
            ),
            "hf_cache_dir": _m(_CAT.GENERAL, "Directory for cached Hugging Face weights."),
            "device": _m(_CAT.PERFORMANCE, "Torch device local models load onto (e.g. cuda:0)."),
            "warmup_on_startup": _m(
                _CAT.PERFORMANCE,
                "Load local models at startup to remove first-request latency, at the cost of higher idle VRAM.",
                "Enable with low worker counts on dedicated GPU pods.",
            ),
            "max_tasks_per_child": _m(_CAT.PERFORMANCE, "Process-pool worker recycling threshold."),
            "max_process_pool_workers": _m(
                _CAT.THROUGHPUT,
                "Caps ingest process-pool workers; each loads the full model stack into GPU memory.",
                "Often 1 for local models; raise only if VRAM allows num_workers x model stack.",
            ),
        },
        section=SectionMeta(
            _CAT.MODEL_SELECTION,
            title="Local Models",
            description="In-pod Hugging Face model loading for extraction, embedding, and ASR.",
        ),
    )

    annotate_fields(
        svc.LocalExtractConfig,
        {
            "enabled": _m(_CAT.MODEL_SELECTION, "Enable in-pod extraction models."),
            "use_table_structure": _m(_CAT.ACCURACY, "Run table-structure recognition for richer table extraction."),
            "ocr_version": _m(_CAT.ACCURACY, "OCR model generation; v2 is more accurate.", "Use v2 unless pinned."),
            "ocr_lang": _m(_CAT.ACCURACY, "OCR language pack (multi vs english)."),
        },
        section=SectionMeta(_CAT.ACCURACY, title="Local Extract"),
    )

    annotate_fields(
        svc.LocalEmbedConfig,
        {
            "enabled": _m(_CAT.MODEL_SELECTION, "Enable in-pod embedding model."),
            "model_name": _m(_CAT.MODEL_SELECTION, "Local embedding model SKU; sets embedding quality/dimensions."),
            "local_ingest_embed_backend": _m(
                _CAT.PERFORMANCE, "Local embedding backend (hf vs vllm).", "vllm is faster on supported GPUs."
            ),
            "gpu_memory_utilization": _m(
                _CAT.PERFORMANCE,
                "Fraction of GPU memory reserved for the embedder.",
                "0.45 default; lower if co-locating extraction models.",
            ),
        },
        section=SectionMeta(_CAT.MODEL_SELECTION, title="Local Embed"),
    )

    annotate_fields(
        svc.LLMConfig,
        {
            "enabled": _m(_CAT.MODEL_SELECTION, "Enable remote LLM answer generation for /v1/answer."),
            "model": _m(_CAT.MODEL_SELECTION, "LLM model identifier used for RAG answers."),
            "api_base": _m(_CAT.MODEL_SELECTION, "Base URL of the LLM endpoint."),
            "api_key": _m(_CAT.SECURITY, "Credential for the LLM endpoint."),
            "temperature": _m(_CAT.ACCURACY, "Sampling temperature; 0 is deterministic.", "0.0 for grounded RAG."),
            "top_p": _m(_CAT.ACCURACY, "Nucleus sampling probability mass."),
            "max_tokens": _m(_CAT.ACCURACY, "Maximum generated answer length.", "512 default."),
            "num_retries": _m(_CAT.GENERAL, "Retry attempts for transient LLM errors."),
            "timeout": _m(_CAT.PERFORMANCE, "Per-request LLM timeout in seconds."),
            "reasoning_enabled": _m(_CAT.ACCURACY, "Allow reasoning-style generation when the model supports it."),
        },
        section=SectionMeta(
            _CAT.MODEL_SELECTION,
            title="LLM",
            description="Remote LLM configuration for service-mode RAG answer generation.",
        ),
    )

    annotate_fields(
        svc.ResourceLimitsConfig,
        {
            "max_memory_mb": _m(_CAT.PERFORMANCE, "Hard RSS limit for the worker process."),
            "max_cpu_cores": _m(_CAT.PERFORMANCE, "CPU affinity cap for the worker process."),
            "gpu_devices": _m(_CAT.PERFORMANCE, "GPU device IDs exposed via CUDA_VISIBLE_DEVICES."),
            "max_upload_bytes": _m(_CAT.GENERAL, "Maximum accepted upload size before buffering."),
        },
        section=SectionMeta(_CAT.PERFORMANCE, title="Resource Limits"),
    )

    annotate_fields(
        svc.AuthConfig,
        {
            "api_token": _m(_CAT.SECURITY, "Bearer token required on protected routes; empty disables auth."),
            "header_name": _m(_CAT.SECURITY, "HTTP header carrying the bearer token."),
            "bypass_paths": _m(_CAT.SECURITY, "Paths served without authentication (health, docs)."),
        },
        section=SectionMeta(_CAT.SECURITY, title="Authentication"),
    )

    annotate_fields(
        svc.MCPConfig,
        {
            "enabled": _m(_CAT.GENERAL, "Mount the FastMCP agent endpoint."),
            "path": _m(_CAT.GENERAL, "HTTP mount path for the MCP app."),
            "base_url": _m(_CAT.GENERAL, "Service URL MCP tools call internally."),
            "enable_write_tools": _m(_CAT.SECURITY, "Expose write-capable MCP tools to agents."),
            "max_concurrency": _m(_CAT.THROUGHPUT, "Maximum concurrent MCP tool invocations."),
            "request_timeout_s": _m(_CAT.PERFORMANCE, "Per-tool request timeout."),
            "ingest_timeout_s": _m(_CAT.PERFORMANCE, "Timeout for MCP-initiated ingest jobs."),
            "poll_interval_s": _m(_CAT.PERFORMANCE, "Job status poll interval for MCP tools."),
        },
        section=SectionMeta(_CAT.GENERAL, title="MCP"),
    )

    annotate_fields(
        svc.GatewayConfig,
        {
            "realtime_url": _m(_CAT.GENERAL, "Backend URL for the realtime worker pool (gateway mode)."),
            "batch_url": _m(_CAT.GENERAL, "Backend URL for the batch worker pool (gateway mode)."),
            "timeout_s": _m(_CAT.PERFORMANCE, "Per-request forwarding timeout."),
            "max_connections": _m(_CAT.THROUGHPUT, "httpx connection pool limit per backend."),
        },
        section=SectionMeta(_CAT.GENERAL, title="Gateway"),
    )

    annotate_fields(
        svc.PipelinePoolConfig,
        {
            "realtime_workers": _m(
                _CAT.THROUGHPUT,
                "Concurrent workers for low-latency page processing.",
                "Scale with CPU cores and remote NIM capacity.",
            ),
            "realtime_queue_size": _m(_CAT.THROUGHPUT, "Max queued items before the realtime pool rejects."),
            "batch_workers": _m(_CAT.THROUGHPUT, "Concurrent workers for bulk document processing."),
            "batch_queue_size": _m(_CAT.THROUGHPUT, "Max queued items before the batch pool rejects."),
        },
        section=SectionMeta(_CAT.THROUGHPUT, title="Pipeline Pool"),
    )

    annotate_fields(
        svc.WorkQueueConfig,
        {
            "gateway_url": _m(_CAT.GENERAL, "Gateway URL used by split-mode workers."),
            "spool_directory": _m(_CAT.GENERAL, "Local spool directory for queued work."),
            "spool_limit_bytes": _m(_CAT.THROUGHPUT, "Maximum on-disk spool size."),
            "claim_timeout_s": _m(_CAT.PERFORMANCE, "Long-poll timeout for claiming work."),
            "lease_ttl_s": _m(_CAT.GENERAL, "Work lease time-to-live before redelivery."),
            "heartbeat_interval_s": _m(_CAT.GENERAL, "Worker heartbeat cadence for held leases."),
            "max_delivery_attempts": _m(_CAT.GENERAL, "Redelivery attempts before a work item is dead-lettered."),
            "max_active_leases_realtime": _m(_CAT.THROUGHPUT, "Concurrent realtime leases a worker may hold."),
            "max_active_leases_batch": _m(_CAT.THROUGHPUT, "Concurrent batch leases a worker may hold."),
        },
        section=SectionMeta(_CAT.THROUGHPUT, title="Work Queue"),
    )

    annotate_fields(
        svc.VectorDbConfig,
        {
            "enabled": _m(_CAT.GENERAL, "Run the dedicated LanceDB vector database pod."),
            "lancedb_uri": _m(_CAT.GENERAL, "LanceDB storage URI."),
            "table_name": _m(_CAT.GENERAL, "LanceDB table receiving embeddings."),
            "embed_model": _m(_CAT.MODEL_SELECTION, "Embedding model the vectordb pod uses for query encoding."),
            "embed_model_provider_prefix": _m(_CAT.MODEL_SELECTION, "Provider prefix for the query embed model."),
            "vectordb_url": _m(_CAT.GENERAL, "URL workers POST embeddings to."),
        },
        section=SectionMeta(_CAT.GENERAL, title="Vector Database"),
    )

    annotate_fields(
        svc.SinksConfig,
        {
            "storage_uri_schemes": _m(_CAT.SECURITY, "Allowed URI schemes for image store egress."),
            "webhook_url_prefixes": _m(_CAT.SECURITY, "Allowed URL prefixes for webhook egress."),
            "vdb_uri_schemes": _m(_CAT.SECURITY, "Allowed URI schemes for per-request VDB writes."),
        },
        section=SectionMeta(_CAT.SECURITY, title="Sinks Allowlist"),
    )

    annotate_fields(
        svc.PipelineOverridesConfig,
        {
            "mode": _m(
                _CAT.SECURITY,
                "How permissively per-request pipeline overrides are accepted (reject/allow_list/allow_all).",
                "allow_list for multi-tenant; allow_all only on trusted dev clusters.",
            ),
        },
        section=SectionMeta(
            _CAT.SECURITY,
            title="Pipeline Overrides Policy",
            description="Trust boundary governing client-supplied PipelineSpec overrides.",
        ),
    )


def _annotate_pipeline_params() -> None:
    from nemo_retriever.common.params import models as p

    annotate_fields(
        p.ExtractParams,
        {
            "extract_text": _m(_CAT.ACCURACY, "Extract text content from documents."),
            "extract_tables": _m(_CAT.ACCURACY, "Detect and extract tables."),
            "extract_charts": _m(_CAT.ACCURACY, "Detect and extract charts."),
            "extract_infographics": _m(_CAT.ACCURACY, "Detect and extract infographics."),
            "method": _m(_CAT.MODEL_SELECTION, "Extraction backend (pdfium vs nemotron_parse)."),
            "use_page_elements": _m(_CAT.ACCURACY, "Use page-element detection to structure extraction."),
            "dpi": _m(_CAT.ACCURACY, "Rasterization DPI; higher improves OCR at higher cost.", "300 typical."),
            "ocr_version": _m(_CAT.ACCURACY, "OCR model generation."),
            "ocr_lang": _m(_CAT.ACCURACY, "OCR language pack."),
        },
        section=SectionMeta(
            _CAT.ACCURACY,
            title="Extraction Defaults",
            description="Default document extraction behaviour for ingest pipelines.",
        ),
    )

    annotate_fields(
        p.EmbedParams,
        {
            "model_name": _m(_CAT.MODEL_SELECTION, "Embedding model SKU; sets quality and vector dimensionality."),
            "embed_modality": _m(_CAT.ACCURACY, "Which content modalities are embedded."),
            "embed_granularity": _m(_CAT.ACCURACY, "Granularity of embedded units (page vs element)."),
            "local_ingest_embed_backend": _m(_CAT.PERFORMANCE, "Local embedding backend (hf vs vllm)."),
            "nim_http_max_concurrent": _m(_CAT.THROUGHPUT, "Max concurrent HTTP requests to the embed NIM."),
        },
        section=SectionMeta(
            _CAT.MODEL_SELECTION,
            title="Embedding Defaults",
            description="Default embedding behaviour for ingest pipelines.",
        ),
    )

    annotate_fields(
        p.TextChunkParams,
        {
            "max_tokens": _m(
                _CAT.ACCURACY,
                "Maximum tokens per chunk; larger chunks preserve context but reduce retrieval granularity.",
                "1024 general; 512 for dense fact lookup.",
            ),
            "overlap_tokens": _m(_CAT.ACCURACY, "Token overlap between adjacent chunks to preserve boundaries."),
            "tokenizer_model_id": _m(_CAT.MODEL_SELECTION, "Tokenizer used to measure chunk boundaries."),
        },
        section=SectionMeta(_CAT.ACCURACY, title="Text Chunking Defaults"),
    )

    annotate_fields(
        p.DedupParams,
        {
            "content_hash": _m(_CAT.ACCURACY, "Drop exact-duplicate content by hash."),
            "bbox_iou": _m(_CAT.ACCURACY, "Deduplicate overlapping detections by bounding-box IoU."),
            "iou_threshold": _m(_CAT.ACCURACY, "IoU threshold above which detections are considered duplicates."),
        },
        section=SectionMeta(_CAT.ACCURACY, title="Deduplication Defaults"),
    )

    annotate_fields(
        p.LanceDbParams,
        {
            "table_name": _m(_CAT.GENERAL, "LanceDB table name."),
            "index_type": _m(_CAT.PERFORMANCE, "Vector index type; trades recall for query latency."),
            "metric": _m(_CAT.ACCURACY, "Distance metric used for similarity search."),
            "num_partitions": _m(_CAT.PERFORMANCE, "IVF partition count; affects recall and query speed."),
            "hybrid": _m(_CAT.ACCURACY, "Enable hybrid dense+sparse retrieval.", "Improves recall on keyword queries."),
            "fts_language": _m(_CAT.ACCURACY, "Full-text-search language for the sparse index."),
        },
        section=SectionMeta(_CAT.GENERAL, title="Vector Store Defaults"),
    )


__all__ = ["apply_annotations"]
