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
- Ask the same (or similar) question again — get it instantly from semantic cache
- Ask something unrelated — it tells you it can't find the information instead of hallucinating
- Use the agentic endpoint for self-correcting retrieval with automatic query rewriting

---

## Architecture

### Simple RAG (POST /api/v1/chat)

```
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
  (vector: score < 1.1, BM25: score > mean and > 0.5)
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

### Agentic RAG (POST /api/v1/chat/agentic)

Graph-based RAG using LangGraph with self-correction. Unlike simple RAG which is one-shot, the agentic pipeline thinks before answering:

```
POST /api/v1/chat/agentic
        │
        ▼
   [retrieve] — hybrid search, no score filtering
        │
        ▼
   [grade_relevance] — LLM grades if chunks can answer the question
        │
   relevant ────────────────────────► [generate] → END
        │
   not relevant + attempts < 2
        │
        ▼
   [rewrite_query] — LLM rewrites query to find better chunks
        │
        └──────────────────────────► [retrieve] (loops back)
        │
   not relevant + attempts >= 2
        │
        ▼
   [no_info] → END
```

**Key differences from simple RAG:**
- Grades retrieved chunks before generating — no hallucination from irrelevant context
- Rewrites the search query (not the original question) if first retrieval fails
- Maximum 2 retries before returning honest "no info"
- Returns `final_query` and `attempts` in response so you can see the agent's reasoning
- Uses unfiltered retrieval (`query_text_raw`) because the grade node handles relevance — filtering twice would discard chunks before the agent can evaluate them

### Document Ingestion (POST /api/v1/ingest)

```
POST /api/v1/ingest
        │
        ▼
   Docling (PDF/document parsing → clean markdown)
        │
        ▼
   RecursiveCharacterTextSplitter
   (500 char chunks, 100 char overlap)
        │
        ├──────────────────────┐
        ▼                      ▼
  ChromaDB Cloud          BM25 Index
  (vector storage         (disk persistence
  + HuggingFace           + singleton pattern
    Embeddings)           + bm25_store.json)
```

---

## Key Technical Decisions

**Why hybrid search instead of vector-only?**

Vector search is great at finding semantically similar content but struggles with exact keyword matches. BM25 is the opposite — great at keywords, misses semantic meaning. When someone asks "what are the DevOps skills", BM25 finds the chunk with the exact word "DevOps" reliably. When someone asks a conceptual question without specific keywords, vector search picks it up. Using both covers more ground than either alone.

**Why semantic cache instead of a simple key-value cache?**

A key-value cache only hits when the question is character-for-character identical. A semantic cache hits when questions mean the same thing — "what are the backend skills" and "list the backend technologies" return the same cached answer. This dramatically improves cache hit rate in real usage. The threshold (cosine distance < 0.15) is tight enough that genuinely different questions — like frontend vs. backend skills — don't incorrectly share a cached answer. I found 0.3 was too loose (different skill questions were hitting the same cache) and tuned it down to 0.15.

**Why similarity score filtering?**

Without filtering, the retrieval always returns chunks even when none are relevant. This leads to the LLM hallucinating an answer from irrelevant context. With filtering, if no chunks score above the relevance threshold, the system returns "I cannot find this information" — which is a better user experience than a confident wrong answer.

**Why two retrieval methods (query_text vs query_text_raw)?**

`query_text` applies strict score filtering — used by the simple RAG endpoint where there's no agent to evaluate relevance. `query_text_raw` skips filtering — used by the agentic pipeline where the `grade_relevance` node does the evaluation. Filtering before grading would discard chunks the agent never gets to evaluate, causing rewrite loops even when relevant chunks exist.

**Why singleton pattern for repositories?**

Without singletons, `IngestService` and `ChatService` each create their own `BM25Repository` instance. The ingest instance adds chunks and saves to disk, but the chat instance loaded at startup before any documents were ingested — so it stays empty. The singleton ensures both services share the same in-memory index.

**Why Docling instead of PyPDF?**

Docling handles complex document layouts — tables, multi-column text, headers — much better than PyPDF. It exports clean markdown which preserves document structure, giving the chunker better signal about where logical sections begin and end.

**Why keep only the last 5 messages in conversation history?**

Unbounded history grows indefinitely, increasing latency and cost with every message. In practice, conversation context beyond the last few exchanges rarely improves answer quality. 5 messages gives enough context for follow-up questions without the overhead.

**Why separate question and query in agentic state?**

`question` is what the user asked — it never changes. `query` is what we're currently searching for — it gets rewritten if retrieval fails. After rewriting, we search with the new `query` but the `generate` node still answers the original `question`. Without this separation, rewriting would change what the agent is trying to answer, not just how it's searching.

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
| Semantic Cache | Redis Stack (vector similarity search) |
| LLM | Groq — Llama 3.3 70B |
| Agentic Orchestration | LangGraph |
| Observability | Langfuse |
| Container | Docker + Docker Compose |

---

## API Endpoints

### Ingest
```
POST /api/v1/ingest
Content-Type: multipart/form-data

file: <your PDF or document>
description: "optional description"
```

### Simple RAG Chat
```
POST /api/v1/chat
Content-Type: application/json

{
  "question": "What are the backend skills?",
  "session_id": "your-session-id"
}
```

Response:
```json
{
  "response": "Backend skills include Node.js, Express...",
  "session_id": "abc-123"
}
```

### Agentic RAG Chat
```
POST /api/v1/chat/agentic
Content-Type: application/json

{
  "question": "What are the backend skills?",
  "session_id": "your-session-id"
}
```

Response:
```json
{
  "response": "Backend skills include Node.js, Express...",
  "session_id": "abc-123",
  "final_query": "backend programming skills and technologies",
  "attempts": 1
}
```

`final_query` shows what the agent actually searched for — if different from your question, the query was rewritten. `attempts` shows how many retrieval retries happened.

### Admin
```
GET    /api/v1/admin/cache   // view all cached queries with TTL
DELETE /api/v1/admin/cache   // clear all cache entries
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
- ChromaDB Cloud account (free tier)
- Groq API key (free tier)
- Langfuse account (free tier)

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

**Run with Docker (recommended)**
docker-compose up --build

**Run locally**
uvicorn main:app --reload

Visit `http://localhost:8000/docs` for the interactive API.

---

## Project Structure

```
py-rag/
├── core/
│   ├── create_index.py        # Redis vector index setup
│   ├── langfuse_client.py     # Observability init
│   └── redis_client.py        # Redis connection
├── evaluation/
│   ├── test_dataset.py        # 10 test cases with ground truth
│   └── evaluate.py            # RAGAS evaluation runner
├── models/
│   └── chat_model.py          # Pydantic schemas
├── repository/
│   ├── bm25_repository.py     # BM25 keyword index (singleton)
│   ├── semantic_cache.py      # Redis semantic cache
│   └── vector_database.py     # ChromaDB operations (singleton)
├── routes/
│   ├── admin.py               # Cache management endpoints
│   ├── chat.py                # Q&A endpoints (simple + agentic)
│   └── ingest.py              # Document upload endpoint
├── services/
│   ├── agentic_rag.py         # LangGraph graph definition
│   ├── ask_llm.py             # Groq LLM + conversation memory
│   ├── chat_service.py        # RAG orchestration + Langfuse tracing
│   └── ingest_service.py      # Document processing pipeline
├── utils/
│   ├── createChunks.py        # Text splitting
│   ├── createEmbeddings.py    # Embedding generation
│   └── extractDocsFromRequest.py  # Docling PDF parsing
├── config.py                  # Settings from .env
└── main.py                    # FastAPI app + startup
```

---

## Known Limitations

- **BM25 is single-server only** — the index lives on disk as JSON. In a multi-instance deployment each server would have its own index. The fix is moving BM25 to a shared store like Redis or Elasticsearch.
- **Conversation history is in-memory** — sessions are lost on server restart. Production fix would be persisting sessions to Redis or a database.
- **Single collection in ChromaDB** — all documents share one collection. Multi-tenant use would need per-user or per-document collections with metadata filtering.
- **No reranking yet** — retrieval could be improved by adding a cross-encoder reranker as a final step after hybrid retrieval.
- **No streaming** — responses are returned in full after generation completes. Adding Server-Sent Events would improve perceived latency.
- **Agentic grading adds latency** — each request makes at least 2 LLM calls (grade + generate) vs 1 for simple RAG. With rewrites it can be 4-6 calls. Worth it for accuracy, but a trade-off to be aware of.

---

## What I'd Add Next

- Cross-encoder reranking after hybrid retrieval
- Streaming responses with Server-Sent Events
- Persistent conversation history in Redis
- Per-document metadata filtering for multi-tenant support
- Load testing and performance benchmarking