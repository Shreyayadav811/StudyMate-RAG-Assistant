from __future__ import annotations

import re
from typing import Any

_HEADING_RE = re.compile(
    r"^(?:chapter\s+\d+|unit\s+\d+|\d+(?:\.\d+)*\s+\S|[A-Z][A-Z0-9 \-/]{3,})\s*$",
    re.IGNORECASE,
)
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\"'“‘])")


def _clean_text(text: str) -> str:
    text = text.replace("\u00ad", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Join line wraps that split ordinary sentences, but preserve paragraph boundaries.
    lines = [line.strip() for line in text.splitlines()]
    output: list[str] = []
    for line in lines:
        if not line:
            if output and output[-1] != "":
                output.append("")
            continue
        if output and output[-1] and not output[-1].endswith((".", "?", "!", ":", ";", "•", "-")) and not _HEADING_RE.match(line):
            output[-1] = output[-1] + " " + line
        else:
            output.append(line)
    return "\n".join(output).strip()


def _split_oversized(text: str, max_chars: int) -> list[str]:
    """Split very long paragraphs at sentence/word boundaries without losing text."""
    if len(text) <= max_chars:
        return [text]
    sentences = _SENTENCE_RE.split(text)
    pieces: list[str] = []
    current = ""
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        if len(sentence) > max_chars:
            if current:
                pieces.append(current)
                current = ""
            words = sentence.split()
            word_buf = ""
            for word in words:
                candidate = f"{word_buf} {word}".strip()
                if len(candidate) <= max_chars:
                    word_buf = candidate
                else:
                    if word_buf:
                        pieces.append(word_buf)
                    # A single pathological token is still preserved.
                    word_buf = word
            if word_buf:
                pieces.append(word_buf)
        else:
            candidate = f"{current} {sentence}".strip()
            if len(candidate) <= max_chars:
                current = candidate
            else:
                if current:
                    pieces.append(current)
                current = sentence
    if current:
        pieces.append(current)
    return pieces


def chunk_documents(
    documents: list[dict[str, Any]], chunk_size: int = 900, overlap: int = 150
) -> list[dict[str, Any]]:
    """Structure-aware, page-preserving chunking with bounded overlap.

    Character counts are used for predictable runtime. Paragraphs and sentences are
    preferred boundaries; metadata and original page numbers are retained.
    """
    if chunk_size < 200:
        raise ValueError("chunk_size must be at least 200 characters.")
    if overlap < 0 or overlap >= chunk_size // 2:
        raise ValueError("overlap must be non-negative and less than half the chunk size.")

    chunks: list[dict[str, Any]] = []
    for document in documents:
        text = _clean_text(str(document.get("text", "")))
        if not text:
            continue

        # Keep headings attached to following content, while blank lines separate paragraphs.
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        atomic: list[str] = []
        for paragraph in paragraphs:
            atomic.extend(_split_oversized(paragraph, chunk_size))

        page_chunks: list[str] = []
        current = ""
        for part in atomic:
            candidate = f"{current}\n\n{part}".strip() if current else part
            if len(candidate) <= chunk_size:
                current = candidate
                continue
            if current:
                page_chunks.append(current)
            # Overlap uses a suffix of the previous chunk and is trimmed to leave room
            # for the next semantic unit. It may begin mid-sentence, which is intentional.
            suffix = current[-overlap:].strip() if current and overlap else ""
            if suffix and len(suffix) + len(part) + 2 <= chunk_size:
                current = f"{suffix}\n\n{part}"
            else:
                current = part
        if current:
            page_chunks.append(current)

        # Avoid accidentally duplicating identical chunks from repeated page headers.
        for index, chunk_text in enumerate(page_chunks, start=1):
            if not chunk_text.strip():
                continue
            chunks.append(
                {
                    "text": chunk_text.strip(),
                    "source": str(document["source"]),
                    "page": int(document["page"]),
                    "chunk_index": index,
                }
            )
    return chunks
