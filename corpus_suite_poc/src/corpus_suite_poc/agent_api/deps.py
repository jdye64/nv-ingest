from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated

from fastapi import Depends, Request

from corpus_suite_poc.config import Settings
from corpus_suite_poc.ingest.service import IngestService
from corpus_suite_poc.ops.metrics import Metrics
from corpus_suite_poc.pipeline.runner import PagePipelineRunner
from corpus_suite_poc.processing import DocumentProcessor
from corpus_suite_poc.query.hybrid import QueryEngine
from corpus_suite_poc.store.blobs import BlobStore
from corpus_suite_poc.store.db import Database


@dataclass
class CorpusState:
    settings: Settings
    db: Database
    blobs: BlobStore
    ingest: IngestService
    runner: PagePipelineRunner
    processor: DocumentProcessor
    query: QueryEngine
    metrics: Metrics


def get_corpus(request: Request) -> CorpusState:
    return request.app.state.corpus


CorpusDep = Annotated[CorpusState, Depends(get_corpus)]
