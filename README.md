# StudyMate — RAG-based Study Assistant

A cloud-ready PDF question-answering app using structure-aware chunking, BGE embeddings, ChromaDB, hybrid dense + BM25 retrieval, reciprocal-rank fusion, cross-encoder reranking, Google Gemini API (free tier, subject to quota), and page-level source references.

## Features

- Upload and index multiple text-based PDFs.
- Page-level metadata preserved for citations.
- Paragraph/sentence-aware chunking with configurable overlap.
- Dense retrieval with `BAAI/bge-small-en-v1.5` and exact-term retrieval with BM25.
- Reciprocal-rank fusion combines dense and lexical candidate rankings.
- Optional cross-encoder reranking (`cross-encoder/ms-marco-MiniLM-L-6-v2`). If reranker loading fails, the app falls back to hybrid ranking.
- Answers are instructed to stay grounded in retrieved excerpts and admit when the documents do not contain an answer.
- Per-session in-memory Chroma collections keep users' uploaded libraries isolated.
- Upload size/page limits and a clean chat interface.

## Run locally (Windows PowerShell)

```powershell
cd path\to\RAG_Study_Assistant_Cloud
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
```

Open `.env`, set `GEMINI_API_KEY` to your own key, then run:

```powershell
streamlit run app.py
```

Create a Gemini API key in Google AI Studio at https://aistudio.google.com/apikey. The Gemini API has a free tier for selected models, but availability and rate limits depend on your account, region, and current policy; do not enable billing if you need a hard zero-cost setup. Never commit `.env` or paste the key into source code. Model availability and free quotas can change. Check the official pricing page before deploying: https://ai.google.dev/gemini-api/docs/pricing. Set `GEMINI_MODEL` to a model currently available to your account if needed.


## Retrieval flow

```text
PDF upload
   ↓
Page-wise text extraction (pypdf)
   ↓
Structure-aware chunking (paragraph/sentence boundaries + overlap)
   ↓
BGE embeddings → Chroma cosine search ─┐
                                       ├→ Reciprocal-rank fusion
BM25 lexical search ───────────────────┘
   ↓
Cross-encoder reranker (top candidates)
   ↓
Grounded LLM answer + source/page references
```

## Important limitations

- Scanned/image-only PDFs may yield no text. Add OCR support if scanned notes must be supported.
- This is a demo deployment, not a hardened multi-tenant production service. Add authentication, request throttling, observability, and persistent storage before handling sensitive or high-volume usage.
- Model downloads and CPU inference may make the first query slow. Free hosting quotas, sleeping behavior, and model API rate limits depend on the provider and can change.
- Relevance scores from the reranker are ranking signals, not calibrated probabilities.

## Evaluation checklist

- [✅] Ask a direct fact question answered in the PDF.
- [✅] Ask a multi-part question and verify each claim against cited pages.
- [✅] Ask a question absent from the PDF; the assistant should say the documents do not provide enough information.
- [✅] Upload two PDFs and confirm source filenames/pages remain distinct.
- [✅] Test with reranking enabled and disabled; compare whether top sources are more relevant.
- [✅] Confirm API key is stored only in cloud Secrets / local `.env`.
