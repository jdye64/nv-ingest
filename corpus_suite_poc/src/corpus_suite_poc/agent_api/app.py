from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from corpus_suite_poc.agent_api.deps import CorpusState
from corpus_suite_poc.agent_api.routes_documents import router as documents_router
from corpus_suite_poc.agent_api.routes_mgmt import router as mgmt_router
from corpus_suite_poc.agent_api.routes_query import router as query_router
from corpus_suite_poc.config import get_settings
from corpus_suite_poc.ingest.service import IngestService
from corpus_suite_poc.ops.metrics import Metrics
from corpus_suite_poc.pipeline.runner import PagePipelineRunner
from corpus_suite_poc.processing import DocumentProcessor
from corpus_suite_poc.query.hybrid import QueryEngine
from corpus_suite_poc.store.blobs import BlobStore
from corpus_suite_poc.store.db import Database


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    db = Database(str(settings.sqlite_path))
    await db.init()
    blobs = BlobStore(settings.blob_dir)
    metrics = Metrics()
    ingest = IngestService(db=db, blobs=blobs)
    runner = PagePipelineRunner(db=db, settings=settings, metrics=metrics)
    processor = DocumentProcessor(
        db=db,
        ingest=ingest,
        runner=runner,
        settings=settings,
        metrics=metrics,
    )
    query = QueryEngine(db=db)
    app.state.corpus = CorpusState(
        settings=settings,
        db=db,
        blobs=blobs,
        ingest=ingest,
        runner=runner,
        processor=processor,
        query=query,
        metrics=metrics,
    )
    yield


def create_app() -> FastAPI:
    app = FastAPI(title="Corpus Suite POC", lifespan=lifespan)
    app.include_router(mgmt_router)
    app.include_router(documents_router)
    app.include_router(query_router)
    return app


app = create_app()
