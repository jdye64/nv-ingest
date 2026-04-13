from __future__ import annotations


def chunk_text(text: str, *, max_chars: int, overlap: int) -> list[tuple[int, int, str]]:
    """Return list of (char_start, char_end, chunk_text) covering the string."""
    t = text.strip()
    if not t:
        return []
    if max_chars <= 0:
        return [(0, len(t), t)]
    out: list[tuple[int, int, str]] = []
    start = 0
    n = len(t)
    while start < n:
        end = min(n, start + max_chars)
        piece = t[start:end]
        out.append((start, end, piece))
        if end >= n:
            break
        start = max(0, end - overlap)
    return out
