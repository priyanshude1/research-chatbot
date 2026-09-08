# RAG Research Paper Chatbot v2 — Agentic Conversational Assistant
> This file is the single source of truth for all design decisions, architectural choices, and implementation constraints for v2 of this project. All Claude Code sessions must read and adhere to this document before writing any code. The v1 manifest is preserved on the main branch.

---

## What Changed from v1

v1 was a single-turn RAG pipeline — one question in, one answer out, no memory, no tool choice, local Llama 3B.

v2 is a multi-turn conversational agent with:
- **Groq API** (`openai/gpt-oss-120b`) replacing local Ollama — better reasoning, free tier, no GPU needed at runtime
- **Rolling conversation memory** — last 8 turns injected into every prompt
- **ReAct agent loop** — model reasons about which tool to use, calls it, observes the result, loops until ready to answer
- **4 tools** — semantic paper search, paper summary retrieval, list papers, arXiv web search
- **Streaming responses** — tokens streamed to frontend as generated
- **Docker + cloud deployment** — containerized, deployed to Render free tier (no GPU required since LLM is on Groq)

---

## Goals

- Demonstrate agentic architecture built from scratch (no LangChain abstractions)
- Show multi-turn conversational memory management
- Show tool-use reasoning via ReAct loop
- Demonstrate production deployment with Docker + Render
- Produce a live public URL as a portfolio piece
- Understand every component deeply — framework magic is explicitly avoided

---

## Technology Decisions (Final — Do Not Change Without Reason)

| Component | Choice | Reason |
|---|---|---|
| LLM | Groq API — `openai/gpt-oss-120b` | OpenAI open-weight 120B model, fast hosted inference, no local GPU required |
| Embedding model | sentence-transformers `all-MiniLM-L6-v2` | Same as v1, unchanged |
| Vector database | ChromaDB (persistent, file-based) | Same as v1, unchanged |
| Agent framework | Custom ReAct loop | Built from scratch — no LangChain/LangGraph |
| Memory | Rolling window (last 8 turns) | Fixed context cost, covers real conversational use |
| API framework | FastAPI with SSE streaming | Adds server-sent events over v1 |
| Containerization | Docker + docker-compose | No GPU needed — LLM is on Groq |
| Cloud deployment | Render free tier | Free, supports Docker, easy GitHub integration |
| arXiv search | arXiv public API | Free, no API key needed |
| Language | Python 3.10+ | Standard for ML ecosystem |

---

## Document Corpus

Same as v1 — PDFs in `./data/`. Same 20 foundational AI/LLM research papers. Corpus can still be expanded at any time by dropping PDFs into `./data/` and running `python index.py`.

New in v2: `index.py` also generates and stores a **pre-computed summary** for each paper at indexing time, stored as a special chunk with `type: "summary"` metadata. Used by the `get_paper_summary` tool.

---

## Agent Architecture — ReAct Loop

The core of v2. ReAct = Reason + Act.

```
User message + conversation history
        ↓
Agent prompt sent to Groq LLM
        ↓
LLM reasons: "I need to search for X"
LLM outputs: {"action": "search_papers", "query": "attention mechanisms"}
        ↓
Agent loop intercepts structured output
Calls the actual tool function
Gets result back
        ↓
Result fed back to LLM as observation
        ↓
LLM reasons again: "I have enough info" or "I need another tool"
        ↓
If done: LLM outputs final answer
If not:  loop continues (max 5 iterations to prevent infinite loops)
        ↓
Final answer streamed to user
        ↓
Turn added to conversation memory
```

### Tool Output Format

The LLM must output either a tool call or a final answer in structured format:

```json
// tool call
{"action": "search_papers", "query": "how does LoRA work"}

// final answer
{"action": "final_answer", "answer": "LoRA works by..."}
```

If the LLM fails to produce valid JSON after 2 retries, agent.py falls back to treating the raw output as the final answer.

---

## The 4 Tools

### 1. search_papers
```
Input:  query (str), filter_source (str, optional)
Does:   embeds query, searches ChromaDB, returns top-5 chunks with metadata
Used:   for any factual question about paper content
```

### 2. get_paper_summary
```
Input:  filename (str) e.g. "vaswani_2017.pdf"
Does:   retrieves pre-generated summary chunk from ChromaDB by metadata filter
Used:   when user asks to summarize a specific paper
```

### 3. list_papers
```
Input:  none
Does:   returns all unique source filenames from ChromaDB metadata
Used:   when user asks what papers are available
```

### 4. web_search
```
Input:  query (str)
Does:   calls arXiv API, returns top-5 paper titles + abstracts + links
Used:   when user asks about recent papers or topics not in the local corpus
```

---

## Conversation Memory

Managed by `memory.py`. Rolling window of last 8 turns (1 turn = 1 user message + 1 assistant response).

```python
# structure stored per turn
{
    "role": "user" | "assistant",
    "content": str
}
```

Memory is **session-scoped** — persists within one browser session, resets on page refresh. No cross-session persistence.

Injected into every agent prompt as a formatted history block before the current user message. Oldest turns dropped automatically when window is full.

Context window budget:
```
System prompt:       ~500 tokens
Conversation memory: ~2000 tokens (8 turns x ~250 tokens each)
Tool observations:   ~1500 tokens
Current question:    ~100 tokens
─────────────────────────────────
Reserved for output: ~27000 tokens remaining (70B model, 32K context)
```

---

## Streaming

FastAPI endpoint `/query` uses **Server-Sent Events (SSE)** to stream tokens as they arrive from Groq.

```python
from fastapi.responses import StreamingResponse

@app.post("/query")
def query(request: QueryRequest):
    return StreamingResponse(
        agent.stream_response(request),
        media_type="text/event-stream"
    )
```

Frontend JavaScript reads the SSE stream and appends tokens to the answer div as they arrive — same typing effect as ChatGPT.

---

## Project Structure

```
research-chatbot/                   <- same repo, v2-agentic branch
├── data/                           <- same PDFs
├── chroma_db/                      <- same vector store (gitignored)
├── src/
│   ├── chunker.py                  <- UNCHANGED from v1
│   ├── embedder.py                 <- UNCHANGED from v1
│   ├── vectorstore.py              <- UNCHANGED from v1
│   ├── retriever.py                <- UNCHANGED from v1
│   ├── memory.py                   <- NEW: rolling conversation window
│   ├── tools.py                    <- NEW: 4 tool functions
│   ├── agent.py                    <- NEW: ReAct loop + tool dispatch
│   └── generator.py                <- MODIFIED: Groq instead of Ollama
├── api/
│   ├── main.py                     <- MODIFIED: SSE streaming + sessions
│   └── static/
│       └── index.html              <- MODIFIED: chat history UI + streaming
├── index.py                        <- MODIFIED: adds summary generation
├── Dockerfile                      <- NEW: no GPU, Groq handles LLM
├── docker-compose.yml              <- NEW
├── .env.example                    <- MODIFIED: adds Groq vars
├── requirements.txt                <- MODIFIED: adds groq, httpx
├── CLAUDE.md                       <- this file (v2)
└── README.md                       <- MODIFIED: v2 architecture + deploy instructions
```

---

## Environment Variables

```
# Groq (replaces Ollama)
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=openai/gpt-oss-120b

# Embedding (unchanged)
EMBEDDING_MODEL=all-MiniLM-L6-v2

# ChromaDB (unchanged)
CHROMA_PATH=./chroma_db
COLLECTION_NAME=research_papers

# Chunking (unchanged)
CHUNK_SIZE=500
CHUNK_OVERLAP=100
TOP_K=5
DATA_PATH=./data

# Agent
MAX_AGENT_ITERATIONS=5
MEMORY_WINDOW_SIZE=8
```

---

## FastAPI Endpoints

```
POST /query
    body:    { "question": str, "session_id": str, "filter_source": str (optional) }
    returns: SSE stream of tokens, ends with [DONE] + sources JSON

GET  /documents
    returns: { "documents": list[str], "total_chunks": int }

POST /index
    body:    { "force_reindex": bool }
    returns: { "indexed": list[str], "skipped": list[str] }

DELETE /session/{session_id}
    returns: { "cleared": bool }    <- clears conversation memory for that session

GET  /health
    returns: { "status": str, "groq": bool, "chromadb": bool }
```

Note: `/query` now takes a `session_id` so the server can maintain separate memory per browser session. Frontend generates a UUID on page load and passes it with every request.

---

## Deployment

**Docker:**
- Single container running FastAPI + ChromaDB
- No GPU required — Groq handles all LLM inference remotely
- ChromaDB mounted as a volume so index persists across container restarts
- GROQ_API_KEY passed as environment variable at runtime (never baked into image)

**Render free tier:**
- Connect GitHub repo, select v2-agentic branch
- Set environment variables in Render dashboard
- Auto-deploys on every push to v2-agentic
- Free tier spins down after inactivity — acceptable for portfolio demo

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

## Build Order (2-week sprint)

```
Days 1-2:   Groq integration + memory.py
Days 3-5:   tools.py + agent.py (ReAct loop) <- hardest part, most time
Days 6-7:   index.py summary generation + arXiv web search tool
Days 8-9:   FastAPI SSE streaming + session handling
Days 10-12: Docker + Render deployment
Days 13-14: Buffer — debugging, README, cleanup
```

---

## What This Project Is Not

- Not using LangChain, LangGraph, or any agent framework — ReAct is implemented manually
- Not persistent memory across sessions — session-scoped only
- Not a multi-user production system — single demo user assumed
- Not using a managed vector database — ChromaDB files on disk
- Not fine-tuning any model — inference only

---

## Known Limitations (Document in README)

- Groq free tier has rate limits (~30 req/min) — not suitable for high traffic
- ReAct loop adds latency — 2-5 LLM calls per complex question
- arXiv web search returns metadata only, not full paper content
- Session memory resets on page refresh — no persistence
- Render free tier spins down after inactivity (~30s cold start)

---

## Files Unchanged from v1

`chunker.py`, `embedder.py`, `vectorstore.py`, `retriever.py` — do not modify these unless a bug is found. All new functionality is additive, not replacing existing components.

---

*Last updated: v2 planning phase. Update this file if any architectural decision changes during implementation.*