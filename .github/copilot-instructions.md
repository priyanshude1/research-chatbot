# RAG Research Paper Chatbot v2 - Copilot Instructions

These instructions are the Copilot-specific implementation contract for this repository. Read and follow them for every task in this workspace. `CLAUDE.md` remains the full architectural source of truth; keep this file aligned with it when architectural decisions change.

## Project Goal

This repository is a research paper chatbot evolving from a single-turn RAG pipeline into a multi-turn agentic assistant. The v2 system must demonstrate:

- Custom agentic reasoning implemented from scratch
- Multi-turn conversational memory
- Tool selection through a bounded ReAct loop
- Streaming responses to the frontend
- Docker and Render deployment without a local GPU

Do not introduce LangChain, LangGraph, or another agent framework. The agent loop and tool dispatch must remain explicit and understandable.

## Technology Contract

- Python 3.10+
- Groq API with `llama-3.1-70b-versatile` for LLM inference
- `sentence-transformers` with `all-MiniLM-L6-v2` for embeddings
- Persistent, file-based ChromaDB
- FastAPI API with Server-Sent Events (SSE)
- Custom ReAct agent loop
- Session-scoped rolling memory containing the last 8 turns
- arXiv public API for web search
- Docker and docker-compose for deployment
- Render free tier as the target cloud deployment

Do not change these technology decisions without an explicit request. Groq handles inference, so GPU support is not required.

## Required v2 Components

The intended project structure is:

```text
src/
  chunker.py       # v1 component; unchanged unless a bug is found
  embedder.py      # v1 component; unchanged unless a bug is found
  vectorstore.py   # v1 component; unchanged unless a bug is found
  retriever.py     # v1 component; unchanged unless a bug is found
  memory.py        # rolling session memory
  tools.py         # the four tool functions
  agent.py         # ReAct loop and tool dispatch
  generator.py     # Groq-backed generation
api/
  main.py          # FastAPI, SSE, and session handling
  static/index.html # chat history and streaming UI
index.py           # indexing and pre-computed summaries
```

Before editing, inspect the actual repository state. The plan may describe files or dependencies that have not been implemented yet. Do not claim a component exists until it is present and verified.

## ReAct Agent Contract

The agent receives the current user message and formatted conversation history, then asks Groq for a structured decision. Each response must be either a tool call or a final answer:

```json
{"action": "search_papers", "query": "how does LoRA work"}
{"action": "final_answer", "answer": "LoRA works by..."}
```

The loop must:

1. Send the question and memory to the model.
2. Parse the model decision.
3. Dispatch an approved tool when requested.
4. Feed the tool result back as an observation.
5. Repeat until a final answer is produced or the maximum of 5 iterations is reached.
6. Fall back to treating raw model output as the answer after 2 failed JSON parsing retries.
7. Add the completed user and assistant turn to session memory.

Never execute arbitrary functions or model-provided code. Unknown actions must produce a controlled error or recovery response.

## Tools

Implement and preserve these four tools:

- `search_papers(query, filter_source=None)`: embed the query and return the top 5 relevant local chunks with metadata.
- `get_paper_summary(filename)`: retrieve the pre-computed summary chunk for a paper.
- `list_papers()`: return unique paper filenames available in ChromaDB.
- `web_search(query)`: call the arXiv public API and return the top 5 titles, abstracts, and links.

Use local corpus search for factual questions about indexed papers. Use the summary tool for requests about a specific paper's overall contents. Use the web tool for recent work or topics outside the local corpus.

## Conversation Memory

Memory is session-scoped and resets when the browser session is refreshed. Store turns in this shape:

```python
{"role": "user" | "assistant", "content": str}
```

Keep only the most recent 8 turns. Inject formatted history before the current question in every agent prompt. Do not add cross-session persistence unless explicitly requested.

## API Contract

The FastAPI service must expose:

- `POST /query`: accepts `question`, `session_id`, and optional `filter_source`; returns an SSE stream ending with completion and source information.
- `GET /documents`: returns document names and total chunk count.
- `POST /index`: accepts `force_reindex` and returns indexed and skipped files.
- `DELETE /session/{session_id}`: clears memory for one session.
- `GET /health`: reports service, Groq, and ChromaDB health.

The frontend generates a UUID per browser session and sends it with every query. Keep API behavior consistent with this contract unless the user explicitly changes it.

## Streaming Rules

Use FastAPI `StreamingResponse` with `text/event-stream`. Stream generated answer content to the frontend as it arrives. Keep tool reasoning and internal control data out of the user-visible answer unless explicitly requested. Ensure streams terminate cleanly on success, parse failure, tool failure, and client disconnect.

## Configuration and Secrets

Use environment variables and `.env` loading for configuration. Expected v2 variables include:

```text
GROQ_API_KEY
GROQ_MODEL=llama-3.1-70b-versatile
EMBEDDING_MODEL=all-MiniLM-L6-v2
CHROMA_PATH=./chroma_db
COLLECTION_NAME=research_papers
CHUNK_SIZE=500
CHUNK_OVERLAP=100
TOP_K=5
DATA_PATH=./data
MAX_AGENT_ITERATIONS=5
MEMORY_WINDOW_SIZE=8
```

Never hard-code API keys, commit secrets, or bake `GROQ_API_KEY` into a Docker image. Preserve compatibility with existing configuration while migrating from v1 only where required by the v2 contract.

## Indexing and Corpus Rules

PDFs live under `data/`. Indexing must preserve the existing chunking and embedding behavior unless a bug is found. In v2, indexing also creates one pre-computed summary chunk per paper with `type: "summary"` metadata. The ChromaDB directory is persistent and should remain gitignored.

## Deployment Rules

The target container is a single Python 3.11-slim image running FastAPI and ChromaDB. It must not require a GPU. Mount ChromaDB as persistent storage and provide secrets at runtime. The expected server command is:

```text
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Document Render free-tier cold starts and Groq rate limits in the README. Do not add deployment claims that have not been verified.

## Implementation Discipline

- Inspect nearby code and tests before editing.
- Prefer the smallest change that satisfies the request.
- Preserve existing public APIs unless the v2 contract requires a change.
- Keep v1 components (`chunker.py`, `embedder.py`, `vectorstore.py`, and `retriever.py`) unchanged unless a concrete bug is identified.
- Use standard-library or already selected dependencies where practical.
- Validate JSON, request data, environment configuration, network failures, empty retrieval results, and missing papers.
- Add focused tests for new behavior when the repository has a test setup.
- Do not fix unrelated bugs or reformat unrelated files.
- Do not commit changes or create branches unless explicitly requested.

## Known Limitations

The README should document these limitations:

- Groq free tier rate limits make the service unsuitable for high traffic.
- The ReAct loop can add latency because complex questions may require 2 to 5 model calls.
- arXiv search returns metadata, not full paper text.
- Session memory resets on page refresh.
- Render free tier can cold-start after inactivity.
