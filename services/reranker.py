from __future__ import annotations

from functools import lru_cache
from sentence_transformers import CrossEncoder

from services.config import RERANKER_MODEL


@lru_cache(maxsize=1)
def get_reranker() -> CrossEncoder:
    return CrossEncoder(RERANKER_MODEL, max_length=512, device="cpu")


def rerank(query: str, candidates: list[dict], top_k: int = 5) -> list[dict]:
    if not candidates:
        return []
    model = get_reranker()
    pairs = [(query, item["text"]) for item in candidates]
    scores = model.predict(pairs, show_progress_bar=False)
    ranked = []
    for item, score in zip(candidates, scores):
        ranked.append({**item, "score": float(score)})
    ranked.sort(key=lambda item: item["score"], reverse=True)
    return ranked[:top_k]
