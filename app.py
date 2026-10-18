from __future__ import annotations

import hashlib
import os
import uuid
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from services.ingestion import ingest_uploaded_pdfs
from services.rag import answer_question
from services.vector_store import create_session_collection, delete_collection

load_dotenv()

st.set_page_config(
    page_title="StudyMate | RAG Study Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .block-container {max-width: 1180px; padding-top: 2rem; padding-bottom: 3rem;}
      .hero {padding: 1.5rem 1.7rem; border-radius: 18px; background: linear-gradient(120deg, #172554 0%, #1d4ed8 65%, #0891b2 100%); color: white; margin-bottom: 1.2rem;}
      .hero h1 {margin: 0 0 .4rem 0; color: white; font-size: 2.1rem;}
      .hero p {margin: 0; color: #e0f2fe; font-size: 1.02rem;}
      .small-muted {color: #64748b; font-size: .9rem;}
      [data-testid="stChatMessage"] {border-radius: 14px;}
    </style>
    <div class="hero">
      <h1>📚 StudyMate</h1>
      <p>Your AI study partner — upload PDFs, ask questions, and get answers grounded in your notes with page citations.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if "session_key" not in st.session_state:
    st.session_state.session_key = uuid.uuid4().hex[:16]
if "collection_name" not in st.session_state:
    st.session_state.collection_name = f"study_{st.session_state.session_key}"
    create_session_collection(st.session_state.collection_name)
if "indexed_files" not in st.session_state:
    st.session_state.indexed_files = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "last_sources" not in st.session_state:
    st.session_state.last_sources = []

with st.sidebar:
    st.header("📄 Your study library")
    st.caption("Upload one or more text-based PDFs. Scanned/image-only PDFs may need OCR first.")
    uploads = st.file_uploader(
        "Choose PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        help="Multiple PDFs are supported in one indexing run.",
    )
    chunk_size = st.select_slider(
        "Chunk size (characters)", options=[600, 800, 900, 1000, 1200], value=900
    )
    overlap = st.select_slider(
        "Chunk overlap (characters)", options=[80, 120, 150, 200], value=150
    )
    top_k = st.slider("Final context chunks", min_value=3, max_value=8, value=5)
    use_reranker = st.checkbox("Use cross-encoder reranking", value=True)

    if st.button("⚡ Index PDFs", type="primary", use_container_width=True, disabled=not uploads):
        with st.spinner("Reading PDFs, chunking, embedding and indexing… first run may download models."):
            try:
                # Rebuild this session's collection from the current selection so changed PDFs
                # cannot leave stale chunks behind. Uploaded bytes never leave the app except for
                # model/API processing described in the README.
                delete_collection(st.session_state.collection_name)
                create_session_collection(st.session_state.collection_name)
                indexed = ingest_uploaded_pdfs(
                    uploads,
                    st.session_state.collection_name,
                    chunk_size=chunk_size,
                    overlap=overlap,
                )
                st.session_state.indexed_files = indexed["files"]
                st.session_state.chat_history = []
                if indexed["chunk_count"] == 0:
                    st.warning("No readable text was found in these PDFs.")
                else:
                    st.success(
                        f"Indexed {len(indexed['files'])} PDF(s) into {indexed['chunk_count']} chunks."
                    )
            except Exception as exc:
                st.error(f"Indexing failed: {exc}")

    if st.session_state.indexed_files:
        st.divider()
        st.subheader("Indexed documents")
        for item in st.session_state.indexed_files:
            st.markdown(f"- **{item['name']}** · {item['pages']} pages")
        st.caption("Your indexed library is session-scoped and may reset when the app session restarts.")

    st.divider()
    st.markdown("**Retrieval pipeline**")
    st.markdown("1. PDF text extraction\n2. Structure-aware chunking\n3. BGE dense retrieval + BM25\n4. Reciprocal-rank fusion\n5. Cross-encoder reranking\n6. Grounded answer + citations")
    has_api_key = bool(os.getenv("GEMINI_API_KEY"))
    if not has_api_key:
        try:
            has_api_key = bool(st.secrets.get("GEMINI_API_KEY", ""))
        except Exception:
            has_api_key = False
    if not has_api_key:
        st.warning("Add GEMINI_API_KEY in the app's Secrets / environment variables to enable answers.")

if not st.session_state.indexed_files:
    left, right = st.columns([1.2, 1])
    with left:
        st.subheader("Get started in three steps")
        st.markdown("""
        1. **Upload** your class notes or textbook PDFs in the sidebar.
        2. Click **Index PDFs** and wait for processing to finish.
        3. Ask a question below. Answers include source and page references.
        """)
    with right:
        st.info("**Example questions**\n\n- Explain the 5 Vs of Big Data.\n- Compare supervised and unsupervised learning.\n- Summarize the main ideas from chapter 2.\n- What do my notes say about normalization?")
else:
    st.caption(f"Library ready · {len(st.session_state.indexed_files)} document(s) · session `{st.session_state.session_key}`")

for entry in st.session_state.chat_history:
    with st.chat_message(entry["role"]):
        st.markdown(entry["content"])
        if entry["role"] == "assistant" and entry.get("sources"):
            with st.expander("Sources used"):
                for source in entry["sources"]:
                    st.markdown(
                        f"**{source['source']}**, page {source['page']} · relevance {source.get('score', 0):.3f}"
                    )
                    st.caption(source["excerpt"])

question = st.chat_input(
    "Ask a question about your indexed PDFs…",
    disabled=not st.session_state.indexed_files,
)
if question:
    st.session_state.chat_history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        with st.spinner("Searching your notes and composing an answer…"):
            try:
                result = answer_question(
                    question=question,
                    collection_name=st.session_state.collection_name,
                    top_k=top_k,
                    use_reranker=use_reranker,
                )
                st.markdown(result["answer"])
                if result["sources"]:
                    with st.expander("Sources used", expanded=True):
                        for source in result["sources"]:
                            st.markdown(
                                f"**{source['source']}**, page {source['page']} · relevance {source.get('score', 0):.3f}"
                            )
                            st.caption(source["excerpt"])
                st.session_state.chat_history.append(
                    {"role": "assistant", "content": result["answer"], "sources": result["sources"]}
                )
            except Exception as exc:
                message = f"I couldn't complete that request: {exc}"
                st.error(message)
                st.session_state.chat_history.append({"role": "assistant", "content": message, "sources": []})
