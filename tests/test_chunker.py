from services.chunker import chunk_documents


def test_chunker_preserves_page_metadata_and_text():
    docs = [{"text": ("Big Data has volume, velocity and variety. " * 40), "source": "notes.pdf", "page": 3}]
    chunks = chunk_documents(docs, chunk_size=400, overlap=50)
    assert len(chunks) > 1
    assert all(c["source"] == "notes.pdf" and c["page"] == 3 for c in chunks)
    assert all(len(c["text"]) <= 400 for c in chunks)
    assert all(c["text"].strip() for c in chunks)
