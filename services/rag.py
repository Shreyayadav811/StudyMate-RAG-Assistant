
from __future__ import annotations

from typing import Any

from services.config import CANDIDATE_K
from services.llm import generate_answer
from services.reranker import rerank
from services.retriever import retrieve



def _build_search_queries(question: str) -> list[str]:
    """Build focused search queries without adding unrelated topics."""
    q = question.lower().strip()
    queries = [question]

    # A 5-Vs question should retrieve only Big Data characteristics.
    is_five_vs_question = any(term in q for term in [
        "5 vs",
        "five vs",
        "five v",
        "characteristics of big data",
        "volume, velocity",
        "volume and velocity",
    ])

    if is_five_vs_question:
        queries.append(
            "The five Vs of Big Data: Volume, Velocity, Variety, "
            "Veracity, and Value, including their definitions and examples."
        )
        return list(dict.fromkeys(queries))

    # Expand only when the user explicitly asks about Hadoop concepts.
    if any(term in q for term in ["hadoop", "hdfs", "yarn", "mapreduce"]):
        queries.append(
            "Hadoop components relevant to the question, including HDFS, "
            "YARN, and MapReduce where applicable."
        )

    # Do not automatically expand every Big Data question into
    # distributed storage and parallel processing topics.
    return list(dict.fromkeys(queries))



def answer_question(
    question: str,
    collection_name: str,
    top_k: int = 5,
    use_reranker: bool = True,
) -> dict[str, Any]:
    question = question.strip()
    if not question:
        raise ValueError("Please enter a question.")

    search_queries = _build_search_queries(question)

    # Retrieve candidates for the original question and each focused sub-query.
    candidates_by_id: dict[str, dict[str, Any]] = {}

    for search_query in search_queries:
        results = retrieve(
            search_query,
            collection_name,
            candidate_k=CANDIDATE_K,
        )

        for item in results:
            item_id = item.get("id")
            if not item_id:
                item_id = (
                    f"{item.get('source', '')}|"
                    f"{item.get('page', 0)}|"
                    f"{item.get('text', '')}"
                )

            if item_id not in candidates_by_id:
                candidates_by_id[item_id] = {**item, "_matched_queries": [search_query]}
            else:
                existing = candidates_by_id[item_id]
                existing["_matched_queries"].append(search_query)
                existing["hybrid_score"] = max(
                    float(existing.get("hybrid_score", 0.0)),
                    float(item.get("hybrid_score", 0.0)),
                )

    candidates = list(candidates_by_id.values())

    if not candidates:
        return {
            "answer": "I couldn't find readable text in the indexed documents.",
            "sources": [],
        }

    if use_reranker:
        try:
            # Score each candidate against focused queries too. Keep its best score,
            # so a chunk relevant to one sub-question is not unfairly discarded.
            best_scores: dict[str, float] = {}

            for search_query in search_queries:
                ranked = rerank(
                    search_query,
                    candidates,
                    top_k=len(candidates),
                )
                for item in ranked:
                    item_id = item.get("id")
                    if not item_id:
                        item_id = (
                            f"{item.get('source', '')}|"
                            f"{item.get('page', 0)}|"
                            f"{item.get('text', '')}"
                        )
                    score = float(item.get("score", 0.0))
                    best_scores[item_id] = max(
                        best_scores.get(item_id, float("-inf")),
                        score,
                    )

            for item in candidates:
                item_id = item.get("id")
                if not item_id:
                    item_id = (
                        f"{item.get('source', '')}|"
                        f"{item.get('page', 0)}|"
                        f"{item.get('text', '')}"
                    )
                item["score"] = best_scores.get(item_id, float("-inf"))

            candidates.sort(key=lambda item: item["score"], reverse=True)

        except Exception:
            # Fall back to hybrid retrieval if reranking fails.
            candidates.sort(
                key=lambda item: float(item.get("hybrid_score", 0.0)),
                reverse=True,
            )
            for item in candidates:
                item["score"] = float(item.get("hybrid_score", 0.0))
    else:
        candidates.sort(
            key=lambda item: float(item.get("hybrid_score", 0.0)),
            reverse=True,
        )
        for item in candidates:
            item["score"] = float(item.get("hybrid_score", 0.0))

    selected = candidates[:max(1, top_k)]

    context_parts = []
    sources = []

    for item in selected:
        label = f"source: {item['source']}, page {item['page']}"
        context_parts.append(f"[{label}]\n{item['text']}")

        sources.append({
            "source": item["source"],
            "page": item["page"],
            "score": float(item.get("score", item.get("hybrid_score", 0.0))),
            "excerpt": item["text"][:650] + (
                "…" if len(item["text"]) > 650 else ""
            ),
        })

    context = "\n\n---\n\n".join(context_parts)
    answer = generate_answer(question, context)

    return {"answer": answer, "sources": sources}
