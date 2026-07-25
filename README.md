
# py-rag

> A production-grade Retrieval-Augmented Generation (RAG) platform built with FastAPI, Docling, Chroma Cloud, BM25, LangGraph, Redis Stack, Groq and Langfuse.

---

# Overview

Unlike tutorial-style RAG applications that simply split text and retrieve embeddings, **py-rag** is designed around production concepts:

- Structure-aware document ingestion
- Hybrid retrieval (Vector + BM25)
- Semantic caching
- Agentic retrieval with self-correction
- Versioned knowledge bases
- Rich metadata preservation
- Observability
- Scalable ingestion pipeline

The objective is to demonstrate how modern enterprise RAG systems are built.

---

# Features

## Document Ingestion

- Local document ingestion
- Public URL ingestion
- Docling DocumentConverter
- Structure-aware parsing
- Docling HybridChunker
- Rich metadata extraction
- Versioned collections
- Batch uploads for Chroma Cloud

## Retrieval

- Hybrid Search
- Chroma Cloud vector retrieval
- BM25 keyword retrieval
- Semantic cache
- Conversation memory

## Agentic RAG

- LangGraph workflow
- Relevance grading
- Query rewriting
- Retry loop
- Honest "No Information Found"

## Observability

- Langfuse tracing
- Token tracking
- Latency monitoring
- Cache analytics

---

# High Level Architecture

```text
                File / URL
                    │
                    ▼
         Document Ingestion Pipeline
                    │
        ┌───────────┴────────────┐
        ▼                        ▼
  Chroma Cloud             BM25 Repository
        │                        │
        └──────────┬─────────────┘
                   ▼
            Hybrid Retrieval
                   ▼
             Agentic Workflow
                   ▼
                Groq LLM
                   ▼
               Final Answer
```

---

# Ingestion Pipeline

```text
File / URL
     │
     ▼
Docling DocumentConverter
     │
     ▼
Structured Document
     │
     ▼
HybridChunker
     │
     ▼
Metadata Extraction
     │
     ├── headings
     ├── page numbers
     ├── captions
     ├── mime type
     ├── filename
     ├── version
     └── collection
     │
     ▼
Embedding Generation
(all-MiniLM-L6-v2)
     │
     ▼
Batch Upload
     │
     ▼
Chroma Cloud

     +

BM25 Index
```

## Metadata

Each chunk stores:

- Document ID
- Collection
- Version
- Filename
- MIME type
- Chunk index
- Headings
- Page numbers
- Captions

---

# Versioning

Collections follow

```
collection_version
```

Examples

```
resume_v1
resume_v2
constitution_v1
engineering_v3
```

This enables rebuilding indexes, future rollback, and isolation between versions.

---

# Retrieval Pipeline

```text
Question
   │
   ├────────► Vector Search
   │
   ├────────► BM25 Search
   │
   ▼
Merge Results
   │
   ▼
LLM
   │
   ▼
Answer
```

---

# Agentic Pipeline

```text
Question
    │
Retrieve
    │
Grade Relevance
    │
 ┌──┴─────────────┐
 │                │
Relevant     Not Relevant
 │                │
 ▼                ▼
Generate     Rewrite Query
 │                │
 └────── Retry ───┘
```

---

# Major Design Decisions

## Why Docling?

Docling understands document structure including headings, lists, tables and page hierarchy.

## Why HybridChunker?

Chunks are created using semantic document structure instead of arbitrary character boundaries.

## Why Hybrid Retrieval?

Vector search provides semantic similarity.

BM25 provides exact keyword matching.

Combining both significantly improves recall.

## Why Metadata?

Metadata enables:

- Source citations
- Filtering
- Traceability
- Future multi-document retrieval

## Why Versioned Collections?

Safer re-indexing without affecting existing knowledge bases.

## Why Batch Uploads?

Large documents can generate thousands of chunks.

The ingestion pipeline automatically uploads chunks in batches to satisfy Chroma Cloud API limits.

---

# Technology Stack

| Layer | Technology |
|-------|------------|
| API | FastAPI |
| Parser | Docling |
| Chunking | Docling HybridChunker |
| Embeddings | all-MiniLM-L6-v2 |
| Vector DB | Chroma Cloud |
| Keyword Search | rank-bm25 |
| Cache | Redis Stack |
| Agent | LangGraph |
| LLM | Groq Llama 3.3 70B |
| Observability | Langfuse |

---

# API

## POST /api/v1/ingest

Supports:

- File
- URL

Parameters

- file
- url
- collection_name
- version
- description

---

## POST /api/v1/chat

Simple Hybrid RAG.

---

## POST /api/v1/chat/agentic

Agentic Hybrid RAG.

---

# Setup

## Requirements

- Python 3.11+
- Docker
- Redis Stack
- Chroma Cloud
- Groq API
- Langfuse

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
uvicorn main:app --reload
```

Visit

```
http://localhost:8000/docs
```

---

# Project Structure

```text
py-rag/
├── core/
├── models/
├── repository/
├── routes/
├── services/
├── utils/
├── config.py
└── main.py
```

---

# Current Capabilities

- File ingestion
- URL ingestion
- Docling parsing
- HybridChunker
- Metadata extraction
- Versioned collections
- Batch uploads
- Chroma Cloud
- BM25 indexing
- Hybrid Retrieval
- Semantic Cache
- Conversation Memory
- Agentic RAG
- Langfuse tracing

---

# Current Limitations

- No active version registry
- BM25 stored locally
- No reranker
- No streaming responses
- No rollback on partial ingestion failure

---

# Roadmap

- Cross Encoder Reranking
- Active Version Registry
- Async ingestion queue
- Confluence connector
- SharePoint connector
- S3 connector
- OCR support
- Multi-modal RAG
- Evaluation dashboard

---

# Example End-to-End Flow

```text
PDF / URL
    │
    ▼
Docling
    │
    ▼
HybridChunker
    │
    ▼
Metadata
    │
    ▼
Embeddings
    │
    ├────────────► BM25
    │
    ▼
Chroma Cloud
         │
         ▼
User Question
         │
         ▼
Hybrid Retrieval
         │
         ▼
Agentic Verification
         │
         ▼
Groq LLM
         │
         ▼
Grounded Answer
```


# Appendix

## Design Note 1

This section can be expanded with implementation details, sequence diagrams, benchmarks, and code examples as the project evolves.

## Design Note 2

This section can be expanded with implementation details, sequence diagrams, benchmarks, and code examples as the project evolves.

## Design Note 3

This section can be expanded with implementation details, sequence diagrams, benchmarks, and code examples as the project evolves.

## Design Note 4

This section can be expanded with implementation details, sequence diagrams, benchmarks, and code examples as the project evolves.

## Design Note 5

This section can be expanded with implementation details, sequence diagrams, benchmarks, and code examples as the project evolves.

## Design Note 6

This section can be expanded with implementation details, sequence diagrams, benchmarks, and code examples as the project evolves.

## Design Note 7

This section can be expanded with implementation details, sequence diagrams, benchmarks, and code examples as the project evolves.

## Design Note 8

This section can be expanded with implementation details, sequence diagrams, benchmarks, and code examples as the project evolves.

## Design Note 9

This section can be expanded with implementation details, sequence diagrams, benchmarks, and code examples as the project evolves.

## Design Note 10

This section can be expanded with implementation details, sequence diagrams, benchmarks, and code examples as the project evolves.

## Design Note 11

This section can be expanded with implementation details, sequence diagrams, benchmarks, and code examples as the project evolves.

## Design Note 12

This section can be expanded with implementation details, sequence diagrams, benchmarks, and code examples as the project evolves.

## Design Note 13

This section can be expanded with implementation details, sequence diagrams, benchmarks, and code examples as the project evolves.

## Design Note 14

This section can be expanded with implementation details, sequence diagrams, benchmarks, and code examples as the project evolves.

## Design Note 15

This section can be expanded with implementation details, sequence diagrams, benchmarks, and code examples as the project evolves.

## Design Note 16

This section can be expanded with implementation details, sequence diagrams, benchmarks, and code examples as the project evolves.

## Design Note 17

This section can be expanded with implementation details, sequence diagrams, benchmarks, and code examples as the project evolves.

## Design Note 18

This section can be expanded with implementation details, sequence diagrams, benchmarks, and code examples as the project evolves.

## Design Note 19

This section can be expanded with implementation details, sequence diagrams, benchmarks, and code examples as the project evolves.

## Design Note 20

This section can be expanded with implementation details, sequence diagrams, benchmarks, and code examples as the project evolves.

## Design Note 21

This section can be expanded with implementation details, sequence diagrams, benchmarks, and code examples as the project evolves.

## Design Note 22

This section can be expanded with implementation details, sequence diagrams, benchmarks, and code examples as the project evolves.

## Design Note 23

This section can be expanded with implementation details, sequence diagrams, benchmarks, and code examples as the project evolves.

## Design Note 24

This section can be expanded with implementation details, sequence diagrams, benchmarks, and code examples as the project evolves.

## Design Note 25

This section can be expanded with implementation details, sequence diagrams, benchmarks, and code examples as the project evolves.

## Design Note 26

This section can be expanded with implementation details, sequence diagrams, benchmarks, and code examples as the project evolves.

## Design Note 27

This section can be expanded with implementation details, sequence diagrams, benchmarks, and code examples as the project evolves.

## Design Note 28

This section can be expanded with implementation details, sequence diagrams, benchmarks, and code examples as the project evolves.

## Design Note 29

This section can be expanded with implementation details, sequence diagrams, benchmarks, and code examples as the project evolves.

## Design Note 30

This section can be expanded with implementation details, sequence diagrams, benchmarks, and code examples as the project evolves.

## Design Note 31

This section can be expanded with implementation details, sequence diagrams, benchmarks, and code examples as the project evolves.

## Design Note 32

This section can be expanded with implementation details, sequence diagrams, benchmarks, and code examples as the project evolves.

## Design Note 33

This section can be expanded with implementation details, sequence diagrams, benchmarks, and code examples as the project evolves.

## Design Note 34

This section can be expanded with implementation details, sequence diagrams, benchmarks, and code examples as the project evolves.

## Design Note 35

This section can be expanded with implementation details, sequence diagrams, benchmarks, and code examples as the project evolves.

## Design Note 36

This section can be expanded with implementation details, sequence diagrams, benchmarks, and code examples as the project evolves.

## Design Note 37

This section can be expanded with implementation details, sequence diagrams, benchmarks, and code examples as the project evolves.

## Design Note 38

This section can be expanded with implementation details, sequence diagrams, benchmarks, and code examples as the project evolves.

## Design Note 39

This section can be expanded with implementation details, sequence diagrams, benchmarks, and code examples as the project evolves.

## Design Note 40

This section can be expanded with implementation details, sequence diagrams, benchmarks, and code examples as the project evolves.
