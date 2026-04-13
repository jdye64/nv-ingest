from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

ENV_HF_CACHE_BASE_DIR = "NEMO_RETRIEVER_HF_CACHE_DIR"

logger = logging.getLogger(__name__)

_ENGINE_EXTENSIONS = frozenset({".engine", ".trt", ".plan"})


def resolve_hf_cache_dir(explicit_hf_cache_dir: Optional[str] = None) -> str:
    """Resolve Hugging Face cache dir from explicit arg, env, then default."""
    candidate = explicit_hf_cache_dir or os.getenv(ENV_HF_CACHE_BASE_DIR)
    if candidate:
        return str(Path(candidate).expanduser())
    return str(Path.home() / ".cache" / "huggingface")


def configure_global_hf_cache_base(explicit_hf_cache_dir: Optional[str] = None) -> str:
    """Apply resolved HF cache base to standard Hugging Face env vars."""
    cache_base = resolve_hf_cache_dir(explicit_hf_cache_dir)
    os.environ.setdefault("HF_HOME", cache_base)
    os.environ.setdefault("HF_HUB_CACHE", str(Path(cache_base) / "hub"))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(Path(cache_base) / "transformers"))
    return cache_base


def _latest_snapshot_dir(repo_dir: Path) -> Optional[Path]:
    """Return the most-recently-modified snapshot revision directory, or None."""
    snapshots = repo_dir / "snapshots"
    if not snapshots.is_dir():
        return None
    revisions = sorted(
        (d for d in snapshots.iterdir() if d.is_dir()),
        key=lambda d: d.stat().st_mtime,
        reverse=True,
    )
    return revisions[0] if revisions else None


def _find_engine_file(directory: Path, filename: Optional[str] = None) -> Optional[Path]:
    """Find a single engine file inside *directory*.

    If *filename* is given, look for that exact name first.  Otherwise
    scan for any file whose extension is in ``_ENGINE_EXTENSIONS``.
    When multiple candidates exist, prefer ``*.engine`` then ``*.trt``
    then ``*.plan``, breaking ties by modification time (newest first).
    """
    if filename:
        candidate = directory / filename
        if candidate.is_file():
            return candidate
        resolved = candidate.resolve()
        if resolved.is_file():
            return resolved

    candidates = [
        f for f in directory.iterdir()
        if f.is_file() and f.suffix in _ENGINE_EXTENSIONS
    ]
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]

    ext_priority = {".engine": 0, ".trt": 1, ".plan": 2}
    candidates.sort(key=lambda f: (ext_priority.get(f.suffix, 99), -f.stat().st_mtime))
    return candidates[0]


def resolve_engine_path(
    path: str,
    *,
    filename: Optional[str] = None,
    model_type: str = "engine",
) -> str:
    """Resolve a user-supplied engine path that may be a file or a directory.

    Handles three layouts:

    1. **Direct file** — returned as-is.
    2. **HF/NGC hub repo directory** — contains ``snapshots/<rev>/…``; the
       latest snapshot revision is searched for an engine file.
    3. **Flat directory** — scanned directly for engine files.

    Parameters
    ----------
    path
        A file path *or* directory path (may be an HF hub cache repo dir).
    filename
        Optional logical filename to look for inside a directory
        (e.g. ``"model.engine"``).  When ``None`` any engine file is matched.
    model_type
        Label used in error messages (e.g. ``"page_elements"``).

    Returns
    -------
    str
        Absolute path to the resolved engine file.

    Raises
    ------
    FileNotFoundError
        When no engine file can be found at the given path.
    """
    p = Path(path).expanduser()

    if p.is_file():
        logger.debug("resolve_engine_path(%s): direct file", path)
        return str(p)

    if not p.is_dir():
        raise FileNotFoundError(
            f"resolve_engine_path: '{path}' is neither an existing file nor a "
            f"directory (model_type={model_type})"
        )

    snapshot = _latest_snapshot_dir(p)
    if snapshot is not None:
        engine = _find_engine_file(snapshot, filename=filename)
        if engine is not None:
            logger.info(
                "resolve_engine_path(%s): resolved via HF snapshot %s -> %s",
                path, snapshot.name, engine,
            )
            return str(engine)

    engine = _find_engine_file(p, filename=filename)
    if engine is not None:
        logger.info("resolve_engine_path(%s): resolved in flat directory -> %s", path, engine)
        return str(engine)

    raise FileNotFoundError(
        f"resolve_engine_path: no engine file found in '{path}' "
        f"(looked for {filename or 'any *' + '/'.join(_ENGINE_EXTENSIONS)} "
        f"in snapshots/ and top-level; model_type={model_type})"
    )


def resolve_model_dir(
    path: str,
    *,
    model_type: str = "model",
) -> str:
    """Resolve a user-supplied path to a model directory.

    For multi-component models (e.g. OCR with detector + recognizer) the
    caller needs the *directory* rather than a single file.  This function
    resolves HF hub repo dirs to their latest snapshot directory, or
    returns the path itself when it is already a flat directory.

    Parameters
    ----------
    path
        A directory path (may be an HF hub cache repo dir).
    model_type
        Label used in error messages.

    Returns
    -------
    str
        Absolute path to the resolved model directory.
    """
    p = Path(path).expanduser()

    if not p.is_dir():
        raise FileNotFoundError(
            f"resolve_model_dir: '{path}' is not an existing directory "
            f"(model_type={model_type})"
        )

    snapshot = _latest_snapshot_dir(p)
    if snapshot is not None:
        logger.info(
            "resolve_model_dir(%s): resolved via HF snapshot -> %s",
            path, snapshot,
        )
        return str(snapshot)

    logger.debug("resolve_model_dir(%s): using directory as-is", path)
    return str(p)
