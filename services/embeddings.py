from __future__ import annotations

from functools import lru_cache
from sentence_transformers import SentenceTransformer

from services.config import EMBEDDING_MODEL

QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


@lru_cache(maxsize=1)
def get_embedding_model() -> SentenceTransformer:
    return SentenceTransformer(EMBEDDING_MODEL, device="cpu")


def embed_documents(texts: list[str]):
    if not texts:
        return []
    return get_embedding_model().encode(
        texts,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
        batch_size=32,
    )


def embed_query(query: str):
    return get_embedding_model().encode(
        QUERY_PREFIX + query,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
