"""
api/main.py — FastAPI Application (v2, agentic)

Responsibility:
    HTTP layer on top of the ReAct agent built in src/. Exposes the five
    endpoints defined in CLAUDE.md (/query, /index, /documents, /health,
    DELETE /session/{id}) and serves the browser client from
    api/static/index.html at /. The frontend is kept inside the same
    FastAPI application so browser requests can use relative API paths and
    do not require a separate web server.

    This file only validates requests, delegates to agent.run_agent(),
    index.run_indexing(), memory.clear_session(), and vectorstore's read
    helpers, and shapes results into HTTP/SSE responses. No agent
    reasoning, retrieval, or indexing logic lives here.

Why plain `def` endpoints instead of `async def`?
    agent.run_agent() and index.run_indexing() are blocking calls (HTTP
    requests to the LLM provider, local embedding model inference, disk
    I/O on ChromaDB). FastAPI runs `def` endpoint functions — and, per
    Starlette, a synchronous generator passed to StreamingResponse — in an
    external threadpool automatically, which keeps the event loop free
    without any manual threadpool wiring. None of the underlying calls are
    actually async, so `def` is both the simpler and the correct choice
    here, including for the time.sleep() pacing in _stream_answer() below.

Streaming note:
    The ReAct agent's final answer comes back as one field of a single
    JSON-mode decision call (agent.py's {"action": "final_answer",
    "answer": ...} protocol) — it is not generated token-by-token, so
    there is nothing to relay as a live token stream. Instead, /query
    runs the agent to completion server-side and then emits the finished
    answer as a sequence of small SSE chunks, giving the frontend the same
    incremental "typing" UX CLAUDE.md describes without changing agent.py's
    already-implemented decision protocol.

Concurrency note:
    This is a single-user local/demo deployment per CLAUDE.md ("session-
    scoped memory... single demo user assumed"). The active LLM provider
    (generator.py's LLM_PROVIDER) serializes on its own rate limits and
    ChromaDB's file-based storage isn't built for concurrent writes, so no
    attempt is made here to support multiple simultaneous requests safely.
"""

import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))

from dotenv import load_dotenv
load_dotenv()

from typing import Generator, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import agent
import generator
import memory
import vectorstore
from index import run_indexing


# Simulated-streaming pacing for /query (see module docstring) — small
# enough per chunk to read as a typing effect, not so small that pacing
# overhead dominates a short answer.
_STREAM_CHUNK_WORDS = 3
_STREAM_CHUNK_DELAY_SECONDS = 0.03

app = FastAPI(title="RAG Research Paper Chatbot")

_STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")


@app.get("/")
def serve_ui():
    """Serve the single-page browser client from FastAPI's static directory."""
    return FileResponse(os.path.join(_STATIC_DIR, "index.html"))


# ── Request / response models ───────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str
    session_id: str
    filter_source: Optional[str] = None


class IndexRequest(BaseModel):
    force_reindex: bool = False


class IndexResponse(BaseModel):
    indexed: list[str]
    skipped: list[str]


class DocumentsResponse(BaseModel):
    documents: list[str]
    total_chunks: int


class HealthResponse(BaseModel):
    status: str
    llm: bool
    chromadb: bool


class SessionClearResponse(BaseModel):
    cleared: bool


# ── /query streaming helpers ────────────────────────────────────────────────

def _sse_event(data: str) -> str:
    """Format one Server-Sent Events message. Always plain text, one line."""
    return f"data: {data}\n\n"


def _stream_answer(
    question: str,
    session_id: str,
    filter_source: Optional[str],
) -> Generator[str, None, None]:
    """
    Run the ReAct agent for one question, then emit its answer as SSE chunks.

    Per CLAUDE.md's /query contract: "SSE stream of tokens, ends with
    [DONE] + sources JSON". Runs agent.run_agent() to completion first
    (this is where all the actual latency — up to MAX_AGENT_ITERATIONS LLM
    calls plus tool use — happens), then streams the finished answer
    text out in small word-chunks with a short delay between them so the
    frontend can render it incrementally instead of pasting the whole
    answer in at once. See the module docstring for why this is simulated
    rather than a live token stream from the LLM.

    A run_agent() failure (e.g. the LLM provider is unreachable) is reported as one SSE
    error event rather than raising — by the time this generator is
    iterating, HTTP headers for the streaming response have already been
    sent, so a mid-stream error can no longer become an HTTP error status;
    it has to be communicated inside the stream itself.
    """
    try:
        result = agent.run_agent(question, session_id, filter_source=filter_source)
    except Exception as e:
        yield _sse_event(json.dumps({"error": str(e)}))
        yield _sse_event("[DONE]")
        return

    words = result["answer"].split(" ")
    for i in range(0, len(words), _STREAM_CHUNK_WORDS):
        chunk_words = words[i:i + _STREAM_CHUNK_WORDS]
        chunk = " ".join(chunk_words)
        if i + _STREAM_CHUNK_WORDS < len(words):
            chunk += " "
        yield _sse_event(chunk)
        time.sleep(_STREAM_CHUNK_DELAY_SECONDS)

    yield _sse_event("[DONE]")
    yield _sse_event(json.dumps({
        "sources": result["sources"],
        "iterations": result["iterations"],
    }))


# ── Endpoints ────────────────────────────────────────────────────────────────

@app.post("/query")
def query(request: QueryRequest):
    """
    Run the ReAct agent for one question and stream the answer back as SSE.

    session_id scopes conversation memory (memory.py, keyed per browser
    session per CLAUDE.md) — the frontend generates a UUID on page load
    and sends it with every request so multi-turn context persists across
    questions in the same session. filter_source, if given, is passed
    through to the agent's search_papers tool calls.
    """
    if not request.question.strip():
        raise HTTPException(status_code=422, detail="question must not be empty")
    if not request.session_id.strip():
        raise HTTPException(status_code=422, detail="session_id must not be empty")

    return StreamingResponse(
        _stream_answer(request.question, request.session_id, request.filter_source),
        media_type="text/event-stream",
    )


@app.delete("/session/{session_id}", response_model=SessionClearResponse)
def delete_session(session_id: str):
    """
    Clear one session's conversation memory. Backs a "new chat" / reset
    action in the frontend. Clearing a session_id that was never seen is
    still a valid, non-error outcome (memory.clear_session() returns False
    rather than raising) — the end state the caller wants (no memory for
    that session_id) already holds either way.
    """
    return {"cleared": memory.clear_session(session_id)}


@app.post("/index", response_model=IndexResponse)
def index(request: IndexRequest):
    """
    Index every PDF in DATA_PATH, skipping already-indexed files unless
    force_reindex is set. Delegates to index.run_indexing(), which already
    returns this exact response shape and also generates each newly-
    indexed paper's summary (used by the get_paper_summary tool).

    Blocking: a full reindex embeds every new/changed PDF and can take a
    while. Per project scope (single demo user) this is left as a plain
    synchronous call rather than a background job.
    """
    try:
        return run_indexing(force_reindex=request.force_reindex)
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/documents", response_model=DocumentsResponse)
def documents():
    """
    List every indexed source filename and the total chunk count.
    Used by the frontend to populate the paper-filter dropdown.
    """
    return {
        "documents": vectorstore.get_all_sources(),
        "total_chunks": vectorstore.get_total_chunks(),
    }


@app.get("/health", response_model=HealthResponse)
def health():
    """
    Report whether the active LLM provider and ChromaDB are actually
    reachable, not just whether their env vars are set. The LLM check goes
    through generator.check_llm_reachable() — provider-agnostic per
    generator.py's LLM_PROVIDER modularity, so this endpoint doesn't need
    to know which provider (Groq, Cerebras, OpenRouter, ...) is active or
    its URL/key shape. ChromaDB is checked by opening the collection
    (lazy-connects on first use) and reading its count. Either check
    failing is reported, not raised — /health should always return 200
    with the status inside the body.
    """
    try:
        vectorstore.get_total_chunks()
        chromadb_ok = True
    except Exception:
        chromadb_ok = False

    return {
        "status": "ok",
        "llm": generator.check_llm_reachable(),
        "chromadb": chromadb_ok,
    }
