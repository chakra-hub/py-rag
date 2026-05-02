# py-rag

A production-grade Retrieval-Augmented Generation (RAG) system built with FastAPI.

## Architecture
- **Ingestion:**
  - PDF parsing with [Docling](https://github.com/Unstructured-IO/docling)
  - Chunking with `RecursiveCharacterTextSplitter` (LangChain)
  - Embedding generation with [SentenceTransformers](https://www.sbert.net/)
  - Storage in [ChromaDB Cloud](https://docs.trychroma.com/cloud/)
- **Retrieval:**
  - Hybrid search: vector similarity (ChromaDB) + BM25 keyword matching
- **Generation:**
  - [Groq](https://groq.com/) (Llama 3.3 70B) for answer generation
  - Conversation memory per session (in-memory)

## Why Hybrid Search?
Vector search handles conceptual queries well but misses exact keyword matches. BM25 handles keywords but misses semantic meaning. Combining both gives better retrieval across query types.

## Features
- PDF ingestion and chunking
- Embedding and vector storage in ChromaDB
- BM25 keyword search (local, disk-backed)
- Hybrid retrieval (vector + BM25)
- LLM answer generation with context
- Per-session conversation history
- FastAPI API endpoints for chat and ingestion

## Known Limitations
- BM25 index stored on disk (not scalable beyond single server)
- Conversation history in memory (lost on restart)
- No evaluation metrics yet

## What I'd Add Next
- Reranking with a cross-encoder
- Langfuse monitoring
- Redis for conversation persistence
- RAGAS evaluation metrics

## Setup
1. Clone the repo:
   ```bash
   git clone https://github.com/chakra-hub/py-rag.git
   cd py-rag
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Add a `.env` file with the following variables:
   ```env
   GROQ_API_KEY=your_groq_api_key
   CHROMA_API_KEY=your_chroma_api_key
   CHROMA_TENANT_ID=your_chroma_tenant_id
   ```
4. Start the server:
   ```bash
   uvicorn main:app --reload
   ```

## API Endpoints
- `POST /api/v1/ingest` — Upload and ingest a PDF
- `POST /api/v1/chat` — Ask a question (with optional session_id for conversation)

## Project Structure
- `main.py` — FastAPI app entrypoint
- `routes/` — API route handlers
- `services/` — Business logic (ingest, chat, LLM)
- `repository/` — Vector DB and BM25 logic
- `utils/` — Chunking, embedding, and doc extraction helpers
- `models/` — Pydantic models

---

Built by [Chakradhar Pradhan](https://chakradhar.com)
