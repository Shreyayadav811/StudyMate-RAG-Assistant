from __future__ import annotations

import chromadb

# Per-session collections prevent users' uploaded documents from mixing in a shared app.
_client = chromadb.EphemeralClient()


def create_session_collection(name: str):
    return _client.get_or_create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"},
    )


def get_collection(name: str):
    return _client.get_or_create_collection(name=name, metadata={"hnsw:space": "cosine"})


def delete_collection(name: str) -> None:
    try:
        _client.delete_collection(name=name)
    except Exception:
        pass


def add_chunks(collection_name: str, chunks: list[dict], embeddings) -> int:
    if len(chunks) != len(embeddings):
        raise ValueError("Chunks and embeddings must have equal lengths.")
    if not chunks:
        return 0
    collection = get_collection(collection_name)
    ids, texts, metadatas, vectors = [], [], [], []
    for chunk, embedding in zip(chunks, embeddings):
        text = chunk["text"].strip()
        if not text:
            continue
        key = f"{chunk['source']}|{chunk['page']}|{chunk.get('chunk_index', 0)}|{text}"
        ids.append(__import__("hashlib").sha256(key.encode("utf-8")).hexdigest())
        texts.append(text)
        metadatas.append(
            {
                "source": str(chunk["source"]),
                "page": int(chunk["page"]),
                "chunk_index": int(chunk.get("chunk_index", 0)),
            }
        )
        vectors.append(embedding.tolist())
    if ids:
        collection.upsert(ids=ids, documents=texts, metadatas=metadatas, embeddings=vectors)
    return len(ids)
