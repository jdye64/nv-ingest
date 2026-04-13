import hashlib
from pathlib import Path


def _shard_path(root: Path, sha256_hex: str) -> Path:
    a, b = sha256_hex[:2], sha256_hex[2:4]
    return root / a / b / sha256_hex


class BlobStore:
    """Content-addressed blob storage on the local filesystem."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def put_bytes(self, data: bytes) -> str:
        h = hashlib.sha256(data).hexdigest()
        path = _shard_path(self.root, h)
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        return h

    def path_for(self, sha256_hex: str) -> Path:
        return _shard_path(self.root, sha256_hex)

    def read_bytes(self, sha256_hex: str) -> bytes:
        return self.path_for(sha256_hex).read_bytes()
