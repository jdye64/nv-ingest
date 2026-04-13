"""Ordered steps executed per page (extend by appending names + handler in runner)."""

STEP_ORDER: tuple[str, ...] = (
    "normalize",
    "extract_text",
    "chunk",
    "index",
)
