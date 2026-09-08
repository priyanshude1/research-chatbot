"""
api/main.py — FastAPI Application

Responsibility:
    HTTP layer on top of the pipeline built in src/. Exposes the four
    endpoints defined in CLAUDE.md (/query, /index, /documents, /health)
    and serves the browser client from api/static/index.html at /. The
    frontend is kept inside the same FastAPI application so browser requests
    can use relative API paths and do not require a separate web server.
    as thin wrappers around pipeline.answer_question(), index.run_indexing(),
    and vectorstore's read helpers. No retrieval, generation, or indexing
    logic lives here — this file only validates requests, calls the
    existing functions, and shapes errors into proper HTTP responses.

Why plain `def` endpoints instead of `async def`?
    pipeline.answer_question() and index.run_indexing() are blocking calls
    (HTTP requests to Ollama, local model inference, disk I/O on ChromaDB).
    FastAPI runs `def` endpoint functions in an external threadpool
    automatically, which keeps the event loop free without any manual
    threadpool wiring. Declaring these `async def` instead would block the
    whole server on every request, since none of the underlying calls are
    actually async. `def` is both the simpler and the correct choice here.

Concurrency note:
    This is a single-user local demo per CLAUDE.md ("no authentication...
    demo use only"). Ollama itself serializes generation and ChromaDB's
    file-based storage isn't built for concurrent writes, so no attempt is
    made here to support multiple simultaneous requests safely.

    Static-file note:
        The static directory is mounted at /static and the root route serves
        index.html. The directory is created before the page is added so this
        backend boundary is ready for the frontend implementation.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))

from dotenv import load_dotenv
load_dotenv()

from typing import Optional

import requests
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import pipeline
import vectorstore
from index import run_indexing


_OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

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
    filter_source: Optional[str] = None


class QueryResponse(BaseModel):
    answer: str
    sources: list[str]
    chunks_used: int


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
    ollama: bool
    chromadb: bool


# ── Endpoints ────────────────────────────────────────────────────────────────

@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    """
    Run the full RAG pipeline for one question: decompose, retrieve,
    generate. Delegates entirely to pipeline.answer_question(), which
    already returns this exact response shape.

    A RuntimeError here means Ollama was unreachable or errored during
    generation (generator.generate_answer() has no safe fallback, unlike
    decomposition) — surfaced as 503 Service Unavailable rather than a raw
    500, since it's an external dependency being down, not a bug.
    """
    try:
        return pipeline.answer_question(
            request.question,
            filter_source=request.filter_source,
        )
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.post("/index", response_model=IndexResponse)
def index(request: IndexRequest):
    """
    Index every PDF in DATA_PATH, skipping already-indexed files unless
    force_reindex is set. Delegates to index.run_indexing(), which already
    returns this exact response shape.

    Blocking: a full reindex embeds every new/changed PDF and can take a
    while. Per project scope (single local user, no cloud deployment) this
    is left as a plain synchronous call rather than a background job.
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
    Report whether Ollama and ChromaDB are actually reachable, not just
    whether their env vars are set. Ollama is checked with a short-timeout
    GET against its own /api/tags endpoint; ChromaDB is checked by opening
    the collection (lazy-connects on first use) and reading its count.
    Either check failing is reported, not raised — /health should always
    return 200 with the status inside the body.
    """
    try:
        response = requests.get(f"{_OLLAMA_BASE_URL}/api/tags", timeout=3)
        ollama_ok = response.status_code == 200
    except requests.exceptions.RequestException:
        ollama_ok = False

    try:
        vectorstore.get_total_chunks()
        chromadb_ok = True
    except Exception:
        chromadb_ok = False

    return {
        "status": "ok",
        "ollama": ollama_ok,
        "chromadb": chromadb_ok,
    }
