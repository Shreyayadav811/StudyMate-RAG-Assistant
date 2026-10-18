from __future__ import annotations

import os

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
MAX_PDF_MB = int(os.getenv("MAX_PDF_MB", "25"))
MAX_PDF_PAGES = int(os.getenv("MAX_PDF_PAGES", "500"))
CANDIDATE_K = int(os.getenv("CANDIDATE_K", "15"))
