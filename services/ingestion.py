from __future__ import annotations

import hashlib
from typing import Any

from services.chunker import chunk_documents
from services.config import MAX_PDF_MB, MAX_PDF_PAGES
from services.embeddings import embed_documents
from services.pdf_loader import load_pdf_bytes
from services.vector_store import add_chunks


def ingest_uploaded_pdfs(uploads, collection_name: str, chunk_size: int = 900, overlap: int = 150) -> dict[str, Any]:
    all_chunks: list[dict[str, Any]] = []
    file_summaries: list[dict[str, Any]] = []
    seen_hashes: set[str] = set()

    for upload in uploads:
        raw = upload.getvalue()
        if len(raw) > MAX_PDF_MB * 1024 * 1024:
            raise ValueError(f"'{upload.name}' exceeds the {MAX_PDF_MB} MB file-size limit.")
        digest = hashlib.sha256(raw).hexdigest()
        if digest in seen_hashes:
            continue
        seen_hashes.add(digest)
        pages = load_pdf_bytes(raw, upload.name, max_pages=MAX_PDF_PAGES)
        if not pages:
            file_summaries.append({"name": upload.name, "pages": 0, "chunks": 0})
            continue
        # Prevent duplicate names from overwriting citation identity.
        source_name = upload.name
        if any(item["name"] == source_name for item in file_summaries):
            source_name = f"{digest[:8]}_{source_name}"
            for page in pages:
                page["source"] = source_name
        chunks = chunk_documents(pages, chunk_size=chunk_size, overlap=overlap)
        all_chunks.extend(chunks)
        file_summaries.append(
            {"name": source_name, "pages": max(p["page"] for p in pages), "chunks": len(chunks)}
        )

    if all_chunks:
        embeddings = embed_documents([chunk["text"] for chunk in all_chunks])
        stored = add_chunks(collection_name, all_chunks, embeddings)
    else:
        stored = 0
    return {"files": file_summaries, "chunk_count": stored, "chunks": all_chunks}
