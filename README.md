# RAG Research Paper Chatbot (v2 — Agentic)

A multi-turn, tool-using conversational agent for querying a local corpus of AI/LLM research papers, built from scratch without any agent framework (no LangChain/LangGraph). Ask it questions across a conversation, and it reasons about which of four tools to call — semantic search, paper summaries, a paper list, or a live arXiv lookup — before answering.

v1 (single-turn RAG, local Llama) is preserved on `main`. This is v2: a ReAct agent loop, rolling conversational memory, a hosted LLM (swappable across providers), and Docker packaging for deployment.

## What it does

- Answers questions grounded in a fixed corpus of 20 indexed AI/LLM papers (`./data/`)
- Remembers the last 8 turns of a conversation and uses that context on follow-up questions
- Decides for itself, per turn, whether to search the corpus, fetch a paper's summary, list available papers, or search arXiv for something not in the corpus
- Streams its answer back to the browser incrementally
- Runs identically locally or in Docker

## Architecture

```
User message + conversation history
        ↓
Agent prompt sent to the active LLM provider
        ↓
LLM reasons: "I need to search for X"
LLM outputs: {"action": "search_papers", "query": "..."}
        ↓
Agent loop calls the matching tool, gets a result back
        ↓
Result fed back to the LLM as an observation
        ↓
LLM reasons again — another tool call, or:
{"action": "final_answer", "answer": "..."}
        ↓
Answer streamed to the browser (SSE), turn saved to session memory
```

Loop is capped at `MAX_AGENT_ITERATIONS` (default 5) to prevent runaway tool-calling. If the LLM can't produce valid structured output after retries, the agent falls back to treating its raw output as the final answer rather than failing the request.

### The four tools

| Tool | Input | Purpose |
|---|---|---|
| `search_papers` | `query`, optional `filter_source` | Embeds the query, returns top-5 semantically similar chunks from ChromaDB |
| `get_paper_summary` | `filename` | Returns the pre-generated summary stored for that paper at indexing time |
| `list_papers` | — | Returns every unique source filename in the index |
| `web_search` | `query` | Queries the public arXiv API for papers not in the local corpus |

## Tech stack

| Component | Choice |
|---|---|
| LLM | Groq (`openai/gpt-oss-120b`) by default — see [LLM provider modularity](#llm-provider-modularity) |
| Embeddings | `sentence-transformers` `all-MiniLM-L6-v2` |
| Vector store | ChromaDB (persistent, file-based) |
| Agent loop | Custom ReAct implementation (no agent framework) |
| API | FastAPI, SSE streaming |
| Containerization | Docker + docker-compose |

## LLM provider modularity

`src/generator.py` talks to any OpenAI-compatible `/chat/completions` endpoint through a single `openai.OpenAI` client pointed at a different `base_url`. Three providers are registered out of the box in `generator._PROVIDERS`: **Groq** (default), **Cerebras**, and **OpenRouter** (useful for free-tier headroom when Groq's daily cap is hit). Switching providers is a `.env` change (`LLM_PROVIDER` + that provider's own API key), no code change.

Note: OpenRouter's free-tagged models rotate and are sometimes flaky — `OPENROUTER_MODEL` in `.env.example` is a verified-working snapshot, not a permanent guarantee. There is currently no automatic fallback if a configured model becomes unavailable; see [Known Limitations](#known-limitations).

## Project structure

```
research-chatbot/
├── data/                    # source PDFs (20 papers)
├── chroma_db/               # persistent vector store (gitignored)
├── src/
│   ├── chunker.py           # PDF text extraction + chunking
│   ├── embedder.py          # sentence-transformers embedding
│   ├── vectorstore.py       # ChromaDB read/write wrapper
│   ├── retriever.py         # similarity search
│   ├── memory.py            # rolling per-session conversation window
│   ├── tools.py             # the 4 tool functions
│   ├── agent.py             # ReAct loop + tool dispatch
│   └── generator.py         # LLM transport (provider-agnostic)
├── api/
│   ├── main.py               # FastAPI app: SSE streaming + session endpoints
│   └── static/index.html     # single-page chat UI
├── index.py                  # indexing CLI (chunk + embed + summarize)
├── Dockerfile
├── docker-compose.yml
├── .env.example
└── requirements.txt
```

## Running locally

```bash
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt

cp .env.example .env         # then fill in at least one provider's API key
python index.py              # chunk, embed, and summarize everything in ./data

uvicorn api.main:app --reload
```

Open `http://localhost:8000`.

To add more papers later, drop PDFs into `./data/` and re-run `python index.py` — already-indexed files are skipped automatically (matched by content hash), so only new/changed papers get re-processed. Use `python index.py --force` to wipe and rebuild the whole index, or `python index.py --summaries-only` to backfill summaries for papers that don't have one yet.

## Running with Docker

```bash
docker compose build
docker compose up
```

Open `http://localhost:8000`. `chroma_db/` and `data/` are bind-mounted (see `docker-compose.yml`), so the index persists across container restarts/rebuilds and new PDFs can be dropped in without rebuilding the image.

## API

| Endpoint | Description |
|---|---|
| `POST /query` | `{question, session_id, filter_source?}` → SSE stream of answer chunks, ending with `[DONE]` + a `{sources, iterations}` JSON event |
| `GET /documents` | `{documents: [...], total_chunks}` |
| `POST /index` | `{force_reindex}` → triggers indexing, returns `{indexed, skipped}` |
| `DELETE /session/{session_id}` | Clears that session's conversation memory |
| `GET /health` | `{status, llm, chromadb, llm_provider, llm_model}` — `llm` reflects whichever provider is currently active, not Groq specifically |

`session_id` scopes conversation memory per browser tab — the frontend generates a UUID on page load. Memory is in-process and resets on server restart; there's no cross-session or cross-restart persistence.

## Environment variables

See `.env.example` for the full list with comments. At minimum you need `LLM_PROVIDER` set to one of `groq` / `cerebras` / `openrouter`, and that provider's API key.

## Known limitations

- Free-tier LLM providers are rate-limited (Groq: ~30 req/min) — not suitable for concurrent/high-traffic use
- The ReAct loop makes 2-5 LLM calls per non-trivial question, adding latency versus a single-shot answer
- `web_search` returns arXiv metadata (title, abstract, link) only, never full paper text
- Conversation memory is session-scoped and in-process — it resets on page refresh and on server restart, and does not survive across multiple app instances
- No automatic failover if the configured LLM model/provider becomes unavailable — this must be changed manually in `.env`
- This is a single-demo-user project: no auth, no multi-tenancy, no managed database

## Deployment

Built to run as a single Docker container (FastAPI + ChromaDB, no GPU required since the LLM is hosted remotely) on a platform like Render's free tier: connect the repo, set the same environment variables from `.env.example` in the platform's dashboard, and deploy. Free tiers that spin down on inactivity will show a cold-start delay on the first request after idling.
