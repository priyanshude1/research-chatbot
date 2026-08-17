# RAG Research Paper Chatbot — Project Design Manifest
> This file is the single source of truth for all design decisions, architectural choices, and implementation constraints for this project. All Claude Code sessions must read and adhere to this document before writing any code.

---

## Project Summary

A local-first Retrieval Augmented Generation (RAG) chatbot that answers questions about AI research papers. Built as a portfolio project targeting both MLOps Engineer and LLM Specialist roles in the German job market.

The system indexes a corpus of AI research papers, retrieves relevant chunks at query time, and generates grounded answers using a local LLM. It is intentionally designed to run fully offline with zero data leakage.

---

## Goals

- Demonstrate end-to-end RAG pipeline architecture
- Show MLOps tooling competence (FastAPI, Docker, structured codebase)
- Produce a clean GitHub repository with clear architecture documentation
- Deploy once in Docker locally for practice — no cloud deployment required
- Keep the system modular so components can be swapped independently

---

## Technology Decisions (Final — Do Not Change Without Reason)

| Component | Choice | Reason |
|---|---|---|
| LLM | Llama 3.2 3B via Ollama | Fits in 4GB VRAM, fast, free, fully local |
| Embedding model | sentence-transformers `all-MiniLM-L6-v2` | Lightweight (~90MB), strong semantic similarity performance |
| Vector database | ChromaDB (persistent, file-based) | Open source, no cost, no managed service needed, files on disk |
| API framework | FastAPI | Standard for MLOps Python APIs, async support |
| Containerization | Docker + docker-compose | MLOps portfolio signal, clean local deployment |
| Document corpus | AI research papers (PDF) | ~10-15 foundational NLP/LLM papers |
| Language | Python 3.10+ | Standard for ML ecosystem |

---

## Document Corpus

Store all PDFs in `./data/`. Starting set:

- Attention is All You Need (Vaswani et al., 2017)
- BERT (Devlin et al., 2018)
- GPT-2 (Radford et al., 2019)
- DistilBERT (Sanh et al., 2019)
- LoRA (Hu et al., 2021)
- RAG — Retrieval-Augmented Generation for NLP (Lewis et al., 2020)
- Llama 2 (Touvron et al., 2023)
- InstructGPT / RLHF (Ouyang et al., 2022)
- Chain of Thought Prompting (Wei et al., 2022)

Corpus can be expanded at any time by dropping PDFs into `./data/` and running `python index.py`. New documents are detected via content hash comparison — already-indexed files are skipped automatically.

---

## Architecture

### Two Phases

**Phase 1 — Indexing (offline, run once or on corpus update)**
```
PDFs in ./data/
      ↓
chunker.py        — split documents into chunks (500 tokens, 100 token overlap)
      ↓
embedder.py       — embed each chunk using sentence-transformers
      ↓
vectorstore.py    — store vectors + metadata in ChromaDB
```

**Phase 2 — Query (runtime, per user request)**
```
User question
      ↓
generator.py      — LLM call #1: decompose query into sub-queries
      ↓
embedder.py       — embed each sub-query
      ↓
retriever.py      — ChromaDB similarity search, top-5 per sub-query, deduplicate
      ↓
pipeline.py       — assemble prompt (system + chunks + question)
      ↓
generator.py      — LLM call #2: generate final answer from retrieved context
      ↓
FastAPI response
```

---

## Project Structure

```
rag-chatbot/
├── data/                      ← PDF research papers go here
├── chroma_db/                 ← ChromaDB persistent storage (gitignored)
├── src/
│   ├── chunker.py             ← PDF parsing + text chunking
│   ├── embedder.py            ← sentence-transformers embedding
│   ├── vectorstore.py         ← ChromaDB client wrapper (store + query)
│   ├── retriever.py           ← similarity search + deduplication logic
│   ├── generator.py           ← Ollama LLM calls (decomposition + generation)
│   └── pipeline.py            ← connects all components end to end
├── api/
│   ├── main.py                ← FastAPI app (endpoints, request/response models)
│   └── static/
│       └── index.html         ← single file UI (vanilla HTML + CSS + JS)
├── index.py                   ← run this to index ./data documents
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .env.example               ← environment variable template
├── CLAUDE.md                  ← this file
└── README.md                  ← portfolio-facing documentation
```

---

## Implementation Constraints

### Chunking
- Chunk size: **500 tokens**
- Overlap: **100 tokens** (last 100 tokens of chunk N become first 100 of chunk N+1)
- Each chunk stores metadata: `source` (filename), `page`, `chunk_index`, `file_hash`
- Rationale: overlap prevents information loss at chunk boundaries; 500 tokens balances context richness vs retrieval precision

### Embedding
- Model: `sentence-transformers/all-MiniLM-L6-v2`
- Output dimension: 384
- Pooling: mean pooling (handled internally by sentence-transformers)
- The embedding model is separate from the LLM — different model, different purpose

### Vector Store
- ChromaDB persistent client pointing to `./chroma_db/`
- Collection name: `"research_papers"`
- Metadata stored per chunk: `source`, `page`, `chunk_index`, `file_hash`, `text`
- Incremental indexing: compute MD5 hash of each file, skip if hash already in metadata

### Retrieval
- Default top-K: **5 chunks per sub-query**
- Deduplicate by chunk ID across sub-queries
- Support metadata filtering by `source` filename for paper-specific queries
- Similarity metric: cosine similarity (ChromaDB default)

### Query Decomposition
- LLM Call #1 to Ollama with system prompt instructing JSON output
- Decompose into 2-4 sub-queries maximum
- Temperature: 0 (deterministic decomposition)
- Parse JSON response, fall back to original query if parsing fails

### Generation
- LLM Call #2 to Ollama with retrieved chunks injected as context
- System prompt instructs model to answer only from provided context
- Model must cite which paper a claim comes from
- If context does not contain the answer, model must say so explicitly — no hallucination
- Temperature: 0.7

### Ollama Integration
- Base URL: `http://localhost:11434` (configurable via env var)
- Model: `llama3.2:3b` (configurable via env var)
- Both LLM calls use the same model instance

---

## Environment Variables

All configurable values live in `.env`. Never hardcode:

```
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.2:3b
EMBEDDING_MODEL=all-MiniLM-L6-v2
CHROMA_PATH=./chroma_db
COLLECTION_NAME=research_papers
CHUNK_SIZE=500
CHUNK_OVERLAP=100
TOP_K=5
DATA_PATH=./data
```

---

## FastAPI Endpoints

```
POST /query
    body: { "question": str, "filter_source": str (optional) }
    returns: { "answer": str, "sources": list[str], "chunks_used": int }

POST /index
    body: { "force_reindex": bool (default false) }
    returns: { "indexed": list[str], "skipped": list[str] }

GET /documents
    returns: { "documents": list[str], "total_chunks": int }

GET /health
    returns: { "status": "ok", "ollama": bool, "chromadb": bool }
```

---

## Frontend

### Decision: Vanilla HTML — No Framework

A single `index.html` file served directly by FastAPI. No React, no Vue, no npm, no build step. The UI is a thin wrapper — the substance of this project is the pipeline, not the interface.

### What the UI Contains
- Text input for the user's question
- Dropdown to optionally filter by a specific paper (populated from `GET /documents`)
- Submit button
- Response area showing the generated answer
- Sources section listing which papers were used

### How It Works
The HTML file makes a `fetch()` call to `POST /query`, receives JSON, and renders the answer and sources. Vanilla JS only.

### Project Structure Update
```
api/
├── main.py
└── static/
    └── index.html     ← single file, vanilla HTML + CSS + JS
```

FastAPI serves it with:
```python
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

app.mount("/static", StaticFiles(directory="api/static"), name="static")

@app.get("/")
def serve_ui():
    return FileResponse("api/static/index.html")
```

### What This Is Not
- Not a React/Vue/Angular app
- Not a separate frontend server
- Not styled with a CSS framework — basic clean styling only
- The UI is built last (Phase 4), after the pipeline and API are fully working

---

## What This Project Is Not

- Not a fine-tuning project — the LLM weights are never updated
- Not a cloud-deployed production system — local first, Docker for practice only
- Not a generic chatbot — it answers only from the indexed document corpus
- Not using a managed vector database — ChromaDB files on disk is intentional

---

## Phased Build Order

**Phase 1 — Core pipeline (build first, test end to end)**
`chunker.py` → `embedder.py` → `vectorstore.py` → `retriever.py` → `generator.py` → `pipeline.py` → `index.py`

**Phase 2 — API layer**
`api/main.py` with all four endpoints wired to pipeline.py

**Phase 3 — Docker**
`Dockerfile` + `docker-compose.yml` — containerize app, mount chroma_db as volume

**Phase 4 — Frontend**
`api/static/index.html` — vanilla HTML/CSS/JS UI wired to FastAPI endpoints

**Phase 5 — Polish**
`README.md` with architecture diagram, setup instructions, example queries and outputs for GitHub portfolio

---

## Known Limitations (Document These in README)

- 4GB VRAM limits model size to 3B parameters — larger models require cloud GPU
- Query decomposition adds one extra LLM call per query (~1-2 seconds latency)
- Cross-document synthesis quality limited by 3B model capability
- ChromaDB file-based storage not suitable for high-concurrency production use
- No authentication on FastAPI endpoints — demo use only

---

## Portfolio Notes

- Clean one-responsibility-per-file structure is intentional — demonstrates production code awareness
- Incremental indexing with hash comparison demonstrates real engineering thinking beyond tutorials
- Query decomposition demonstrates awareness of naive RAG failure modes
- Metadata filtering demonstrates paper-specific query handling
- The `/health` endpoint demonstrates MLOps operational awareness
- Docker volume mounting for ChromaDB persistence demonstrates container data management

---

*Last updated: Project design phase. Update this file if any architectural decision changes during implementation.*
