# py-rag

A production-grade RAG (Retrieval Augmented Generation) system built with FastAPI. Upload any document and ask questions about it — the system finds the most relevant parts and answers based only on what's in the document.

---

## Why I Built This

I wanted to go beyond the typical "stuff the whole document into the prompt" approach that most RAG tutorials teach. This system does proper retrieval — it finds only the relevant chunks, filters out noise, caches similar queries, and tells you when it can't find the answer instead of making things up.

---

## What It Does

- Upload a PDF or document
- Ask questions about it in natural language
- Get answers grounded strictly in the document content
- Ask the same (or similar) question again — get it instantly from cache
- Ask something unrelated — it tells you it can't find the information instead of hallucinating

---

## Architecture

```
POST /api/v1/ingest
        │
        ▼
   Docling (PDF parsing)
        │
        ▼
   RecursiveCharacterTextSplitter
   (500 char chunks, 100 char overlap)
        │
        ├──────────────────────┐
        ▼                      ▼
  ChromaDB Cloud          BM25 Index
  (vector storage)        (disk persistence)
  + HuggingFace           + singleton pattern
    Embeddings


POST /api/v1/chat
        │
        ▼
  Semantic Cache Check (Redis Stack)
        │
   cache hit ──────────────────────► return cached answer
        │
   cache miss
        │
        ▼
  Hybrid Retrieval
        ├── Vector search (ChromaDB) → top 2 by cosine similarity
        └── BM25 keyword search      → top 2 by BM25 score
        │
        ▼
  Similarity Score Filtering
  (vector: score < 1.5, BM25: score > mean)
        │
   no relevant chunks ──────────────► "No information found"
        │
   relevant chunks found
        │
        ▼
  Groq LLM (Llama 3.3 70B)
  + conversation memory (last 5 messages)
        │
        ▼
  Save to Semantic Cache
        │
        ▼
  Langfuse Trace (latency, tokens, cache hit/miss)
        │
        ▼
     Response
```

---

## Key Technical Decisions

**Why hybrid search instead of vector-only?**

Vector search is great at finding semantically similar content but struggles with exact keyword matches. BM25 is the opposite — great at keywords, misses semantic meaning. When someone asks "what are the DevOps skills", BM25 finds the chunk with the exact word "DevOps" reliably. When someone asks a conceptual question without specific keywords, vector search picks it up. Using both covers more ground than either alone.

**Why semantic cache instead of a simple key-value cache?**

A key-value cache only hits when the question is character-for-character identical. A semantic cache hits when questions mean the same thing — "what are the backend skills" and "list the backend technologies" return the same cached answer. This dramatically improves cache hit rate in real usage. The threshold (cosine distance < 0.15) is tight enough that genuinely different questions — like frontend vs. backend skills — don't incorrectly share a cached answer.

**Why similarity score filtering?**

Without filtering, the retrieval always returns chunks even when none of them are relevant. This leads to the LLM hallucinating an answer from irrelevant context. With filtering, if no chunks score above the relevance threshold, the system returns "I cannot find this information" — which is a better user experience than a confident wrong answer.

**Why singleton pattern for repositories?**

Without singletons, `IngestService` and `ChatService` each create their own `BM25Repository` instance. The ingest instance adds chunks and saves to disk, but the chat instance loaded at startup before any documents were ingested — so it stays empty. The singleton ensures both services share the same in-memory index.

**Why Docling instead of PyPDF?**

Docling handles complex document layouts — tables, multi-column text, headers — much better than PyPDF. It exports clean markdown which preserves document structure, giving the chunker better signal about where logical sections begin and end.

**Why keep only the last 5 messages in conversation history?**

Unbounded history grows indefinitely, increasing latency and cost with every message. In practice, conversation context beyond the last few exchanges rarely improves answer quality. 5 messages gives enough context for follow-up questions without the overhead.

---

## Stack

| Component | Technology |
|---|---|
| API Framework | FastAPI |
| PDF Parsing | Docling |
| Chunking | LangChain RecursiveCharacterTextSplitter |
| Embeddings | SentenceTransformers (all-MiniLM-L6-v2) |
| Vector Database | ChromaDB Cloud |
| Keyword Search | BM25 (rank-bm25) |
| Semantic Cache | Redis Stack (vector similarity) |
| LLM | Groq — Llama 3.3 70B |
| Observability | Langfuse |

---

## API Endpoints

### Ingest
```
POST /api/v1/ingest
Content-Type: multipart/form-data

file: <your PDF or document>
description: "optional description"
```

### Chat
```
POST /api/v1/chat
Content-Type: application/json

{
  "question": "What are the backend skills?",
  "session_id": "your-session-id"  // optional, generated if not provided
}
```

### Admin
```
GET    /api/v1/admin/cache   // view all cached queries
DELETE /api/v1/admin/cache   // clear all cache
```

### Health
```
GET /health
```

---

## Setup

**Prerequisites**
- Python 3.11+
- Docker (for Redis Stack)
- ChromaDB Cloud account
- Groq API key (free tier works)
- Langfuse account (free tier works)

**Install dependencies**
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Environment variables**

Create a `.env` file:
```
GROQ_API_KEY=your_groq_api_key
CHROMA_API_KEY=your_chroma_api_key
CHROMA_TENANT_ID=your_chroma_tenant_id
LANGFUSE_PUBLIC_KEY=pk-...
LANGFUSE_SECRET_KEY=sk-...
LANGFUSE_HOST=https://cloud.langfuse.com
REDIS_HOST=localhost
REDIS_PORT=6379
```

**Start Redis Stack**
```bash
docker run -d -p 6379:6379 -p 8001:8001 redis/redis-stack:latest
```

**Create Redis search index**
```bash
python core/create_index.py
```

**Run the server**
```bash
uvicorn main:app --reload
```

Visit `http://localhost:8000/docs` for the interactive API.

---

## Project Structure

```
py-rag/
├── core/
│   ├── create_index.py      # Redis vector index setup
│   ├── langfuse_client.py   # Observability init
│   └── redis_client.py      # Redis connection
├── models/
│   └── chat_model.py        # Pydantic schemas
├── repository/
│   ├── bm25_repository.py   # BM25 keyword index (singleton)
│   ├── semantic_cache.py    # Redis semantic cache
│   └── vector_database.py   # ChromaDB operations (singleton)
├── routes/
│   ├── admin.py             # Cache management endpoints
│   ├── chat.py              # Q&A endpoint
│   └── ingest.py            # Document upload endpoint
├── services/
│   ├── ask_llm.py           # Groq LLM + conversation memory
│   ├── chat_service.py      # RAG orchestration + Langfuse tracing
│   └── ingest_service.py    # Document processing pipeline
├── utils/
│   ├── createChunks.py      # Text splitting
│   ├── createEmbeddings.py  # Embedding generation
│   └── extractDocsFromRequest.py  # Docling PDF parsing
├── config.py                # Settings from .env
└── main.py                  # FastAPI app + startup
```

---

## Known Limitations

- **BM25 is single-server only** — the index lives on disk as JSON. In a multi-instance deployment each server would have its own index. The fix is moving BM25 to a shared store like Redis or Elasticsearch.
- **Conversation history is in-memory** — sessions are lost on server restart. Production fix would be persisting sessions to Redis or a database.
- **Single collection in ChromaDB** — all documents share one collection. Multi-tenant use would need per-user or per-document collections with metadata filtering.
- **No reranking yet** — retrieval could be improved by adding a cross-encoder reranker as a final step after hybrid retrieval.
- **No streaming** — responses are returned in full after generation completes. Adding Server-Sent Events would improve perceived latency.

## What I'd Add Next

- Cross-encoder reranking after hybrid retrieval
- Streaming responses with Server-Sent Events
- LangGraph agentic layer with query rewriting and relevance grading
- RAGAS evaluation metrics
- Docker Compose for one-command setup
- Per-document metadata filtering in ChromaDB