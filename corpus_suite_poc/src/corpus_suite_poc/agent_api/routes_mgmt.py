from __future__ import annotations

from fastapi import APIRouter

from corpus_suite_poc.agent_api.deps import CorpusDep
from corpus_suite_poc.ops.health import health_payload

router = APIRouter(tags=["management"])


@router.get("/health")
async def health() -> dict[str, str]:
    return health_payload()


@router.get("/v1/metrics")
async def metrics(corpus: CorpusDep) -> dict[str, int]:
    return corpus.metrics.snapshot()
