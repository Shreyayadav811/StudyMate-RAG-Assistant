from __future__ import annotations

import re
from collections import defaultdict
from typing import Any

from rank_bm25 import BM25Okapi

from services.config import CANDIDATE_K
from services.embeddings import embed_query
from services.vector_store import get_collection


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+(?:['-][a-z0-9]+)?", text.lower())


def _rrf(rankings: list[list[str]], k: int = 60) -> dict[str, float]:
    scores: dict[str, float] = defaultdict(float)
    for ranking in rankings:
        for rank, item_id in enumerate(ranking, start=1):
            scores[item_id] += 1.0 / (k + rank)
    return dict(scores)


def retrieve(query: str, collection_name: str, candidate_k: int | None = None) -> list[dict[str, Any]]:
    collection = get_collection(collection_name)
    count = collection.count()
    if count == 0:
        return []
    candidate_k = max(1, min(candidate_k or CANDIDATE_K, count))
    dense = collection.query(
        query_embeddings=[embed_query(query).tolist()],
        n_results=candidate_k,
        include=["documents", "metadatas", "distances"],
    )
    dense_ids = dense["ids"][0]
    dense_docs = dense["documents"][0]
    dense_meta = dense["metadatas"][0]
    dense_dist = dense["distances"][0]

    # BM25 is run over the full session collection. For small study libraries this is
    # inexpensive and catches exact terms/numbers that dense search can miss.
    all_records = collection.get(include=["documents", "metadatas"])
    corpus_docs = all_records["documents"] or []
    corpus_ids = all_records["ids"] or []
    corpus_meta = all_records["metadatas"] or []
    tokenized = [_tokens(text or "") for text in corpus_docs]
    query_tokens = _tokens(query)
    bm25_scores: list[float] = []
    bm25_ids: list[str] = []
    if tokenized and query_tokens:
        bm25 = BM25Okapi(tokenized)
        raw_scores = bm25.get_scores(query_tokens)
        order = sorted(range(len(raw_scores)), key=lambda i: raw_scores[i], reverse=True)
        for idx in order[:candidate_k]:
            if raw_scores[idx] > 0:
                bm25_ids.append(corpus_ids[idx])
                bm25_scores.append(float(raw_scores[idx]))

    fusion_scores = _rrf([dense_ids, bm25_ids])
    by_id: dict[str, dict[str, Any]] = {}
    for item_id, text, meta, distance in zip(dense_ids, dense_docs, dense_meta, dense_dist):
        by_id[item_id] = {
            "id": item_id,
            "text": text,
            "source": meta.get("source", "Unknown source"),
            "page": int(meta.get("page", 0)),
            "distance": float(distance),
            "hybrid_score": fusion_scores.get(item_id, 0.0),
        }
    for item_id in bm25_ids:
        if item_id not in by_id:
            # Fetch just the candidate omitted from the dense top-k.
            try:
                row = collection.get(ids=[item_id], include=["documents", "metadatas"])
                meta = row["metadatas"][0]
                by_id[item_id] = {
                    "id": item_id,
                    "text": row["documents"][0],
                    "source": meta.get("source", "Unknown source"),
                    "page": int(meta.get("page", 0)),
                    "distance": 1.0,
                    "hybrid_score": fusion_scores.get(item_id, 0.0),
                }
            except Exception:
                continue
    candidates = list(by_id.values())
    candidates.sort(key=lambda item: item["hybrid_score"], reverse=True)
    return candidates[:candidate_k]
