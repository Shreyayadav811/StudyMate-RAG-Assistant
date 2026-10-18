from __future__ import annotations

from io import BytesIO
from typing import Any

from pypdf import PdfReader


def load_pdf_bytes(file_bytes: bytes, source_name: str, max_pages: int = 500) -> list[dict[str, Any]]:
    """Extract text page-by-page while preserving source and 1-based page metadata."""
    reader = PdfReader(BytesIO(file_bytes), strict=False)
    if reader.is_encrypted:
        try:
            reader.decrypt("")
        except Exception as exc:
            raise ValueError(f"'{source_name}' is encrypted and could not be opened.") from exc
    if len(reader.pages) > max_pages:
        raise ValueError(f"'{source_name}' has {len(reader.pages)} pages; limit is {max_pages}.")

    pages = []
    for page_number, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text(extraction_mode="layout") or ""
        except TypeError:  # compatibility with older pypdf versions
            text = page.extract_text() or ""
        text = "\n".join(line.rstrip() for line in text.replace("\x00", " ").splitlines()).strip()
        if text:
            pages.append({"text": text, "source": source_name, "page": page_number})
    return pages
