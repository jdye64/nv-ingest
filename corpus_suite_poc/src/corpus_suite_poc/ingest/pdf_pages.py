from __future__ import annotations

import io
from dataclasses import dataclass

from pypdf import PdfReader


@dataclass
class PageText:
    page_index: int
    text: str


def sniff_mime(filename: str, data: bytes) -> str:
    lower = filename.lower()
    if lower.endswith(".pdf"):
        return "application/pdf"
    if lower.endswith(".txt") or lower.endswith(".md"):
        return "text/plain"
    if data.startswith(b"%PDF"):
        return "application/pdf"
    return "application/octet-stream"


def extract_pdf_pages(data: bytes) -> list[PageText]:
    reader = PdfReader(io.BytesIO(data))
    out: list[PageText] = []
    for i, page in enumerate(reader.pages):
        t = page.extract_text() or ""
        out.append(PageText(page_index=i, text=t))
    return out


def extract_text_pages(data: bytes) -> list[PageText]:
    text = data.decode("utf-8", errors="replace")
    if "\f" in text:
        parts = text.split("\f")
        return [PageText(page_index=i, text=p.strip()) for i, p in enumerate(parts) if p.strip()]
    return [PageText(page_index=0, text=text)]


def pages_for_blob(*, mime: str | None, filename: str, data: bytes) -> list[PageText]:
    m = mime or sniff_mime(filename, data)
    if m == "application/pdf" or filename.lower().endswith(".pdf") or data.startswith(b"%PDF"):
        return extract_pdf_pages(data)
    return extract_text_pages(data)


def page_count_for_blob(*, mime: str | None, filename: str, data: bytes) -> int:
    return len(pages_for_blob(mime=mime, filename=filename, data=data))
