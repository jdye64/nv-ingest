from dataclasses import dataclass, field
from threading import Lock


@dataclass
class Metrics:
    """Process-local counters for quick observability."""

    _lock: Lock = field(default_factory=Lock, repr=False)
    documents_created: int = 0
    pages_processed: int = 0
    chunks_indexed: int = 0
    queries: int = 0
    errors: int = 0

    def inc(self, name: str, n: int = 1) -> None:
        with self._lock:
            cur = getattr(self, name, None)
            if cur is None:
                return
            setattr(self, name, int(cur) + n)

    def snapshot(self) -> dict[str, int]:
        with self._lock:
            return {
                "documents_created": self.documents_created,
                "pages_processed": self.pages_processed,
                "chunks_indexed": self.chunks_indexed,
                "queries": self.queries,
                "errors": self.errors,
            }
