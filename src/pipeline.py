"""
pipeline.py — End-to-End Query Pipeline

Responsibility:
    Connects every component built so far into the single entry point the
    FastAPI layer (api/main.py) calls for POST /query. Owns the one piece
    of the architecture that didn't have a home yet: prompt assembly.

    Per CLAUDE.md's Phase 2 diagram:

        User question
              |
        generator.py      — LLM call #1: decompose into sub-queries
              |
        embedder.py        — embed each sub-query   (done inside retriever.py)
              |
        retriever.py        — similarity search, top-K per sub-query, dedupe
              |
        pipeline.py        — assemble prompt (system + chunks + question)  <- here
              |
        generator.py      — LLM call #2: generate final answer
              |
        FastAPI response

    pipeline.py never talks to Ollama or ChromaDB directly — it only calls
    generator.py and retriever.py, which own those integrations. Its own
    job is narrow: decide what the LLM is told (the system prompt's
    grounding/citation rules) and how retrieved chunks are formatted into
    context, then hand the assembled prompt to generator.generate_answer().

Why cap the number of chunks sent to the LLM?
    retriever.retrieve() can return up to top_k * len(sub_queries) unique
    chunks (e.g. 4 sub-queries * top_k=5 = up to 20) before any prompt-size
    consideration — deduplication reduces this but doesn't bound it.
    Llama 3.2 3B runs locally with a limited context window, and every
    extra chunk adds latency and dilutes the model's attention on the
    passages that actually matter. Chunks are already ranked by distance
    (most similar first) by retriever.py, so capping is just "keep the
    best N" — see _MAX_CONTEXT_CHUNKS below.

Why return early when no chunks are retrieved?
    An empty context means either the corpus doesn't cover the question or
    a source filter excluded everything. Sending an empty-context prompt
    to the LLM anyway wastes a call and relies on the model to notice and
    say "I don't know" on its own. Answering deterministically here is
    faster and matches CLAUDE.md's no-hallucination requirement exactly.
"""

import os
from typing import Optional

import generator
import retriever


_TOP_K = int(os.getenv("TOP_K", "5"))

# Not one of the env vars listed in CLAUDE.md — this is an internal
# implementation detail (prompt-size budget), not a user-facing config
# knob, so it stays a local constant rather than growing the .env surface.
_MAX_CONTEXT_CHUNKS = 10

_NO_CONTEXT_ANSWER = "I don't have any information about that in the indexed papers."

_SYSTEM_PROMPT = """You are a research assistant answering questions about a corpus of AI research papers.

Answer ONLY using the context provided below. Do not use any outside knowledge.

For every claim you make, cite the source paper it came from using the filename \
shown in the context (e.g. "According to vaswani_2017.pdf, ...").

If the context does not contain enough information to answer the question, say \
so explicitly instead of guessing or making something up."""


def format_context(chunks: list[dict]) -> str:
    """
    Format retrieved chunks into the context block injected into the prompt.

    Each chunk is labeled with its source filename and page number so the
    LLM can cite it correctly, per CLAUDE.md's "model must cite which
    paper a claim comes from" requirement. Chunks are joined in the order
    given (retriever.py already ranks them by distance, most similar
    first), so the most relevant passages appear first in the prompt.

    Args:
        chunks: list of chunk dicts as returned by retriever.retrieve()
                (each has at least "source", "page", "text")

    Returns:
        a single string, chunks separated by blank lines, e.g.:

            [Source: vaswani_2017.pdf, p.3]
            The Transformer uses multi-head self-attention...

            [Source: lora_2021.pdf, p.5]
            LoRA freezes the pretrained weights and injects...
    """
    return "\n\n".join(
        f"[Source: {chunk['source']}, p.{chunk['page']}]\n{chunk['text']}"
        for chunk in chunks
    )


def answer_question(
    question: str,
    filter_source: Optional[str] = None,
    top_k: int = _TOP_K,
) -> dict:
    """
    Run the full query pipeline for one user question.

    This is the single function api/main.py's POST /query endpoint calls.
    It performs both LLM calls (decomposition, then generation) and the
    retrieval step in between, and returns exactly the shape CLAUDE.md
    specifies for the /query response.

    Args:
        question:      the user's raw question, exactly as submitted
        filter_source: optional filename to restrict retrieval to one paper
                        (passed straight through to retriever.retrieve())
        top_k:         chunks to retrieve per sub-query (default TOP_K env var)

    Returns:
        {
            "answer":      str,        # generated answer, or the fixed
                                        # no-context message if nothing relevant
                                        # was retrieved
            "sources":     list[str],  # unique source filenames actually used,
                                        # sorted
            "chunks_used": int,        # how many chunks were sent to the LLM
                                        # (0 if the no-context path was taken)
        }
    """
    sub_queries = generator.decompose_query(question)
    chunks = retriever.retrieve(sub_queries, top_k=top_k, source_filter=filter_source)

    if not chunks:
        return {
            "answer": _NO_CONTEXT_ANSWER,
            "sources": [],
            "chunks_used": 0,
        }

    chunks = chunks[:_MAX_CONTEXT_CHUNKS]

    context = format_context(chunks)
    user_prompt = f"Context:\n{context}\n\nQuestion: {question}"

    answer = generator.generate_answer(_SYSTEM_PROMPT, user_prompt)
    sources = sorted({chunk["source"] for chunk in chunks})

    return {
        "answer": answer,
        "sources": sources,
        "chunks_used": len(chunks),
    }
