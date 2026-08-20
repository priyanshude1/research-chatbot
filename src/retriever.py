"""
retriever.py — Similarity Search Orchestration and Deduplication

Responsibility:
    Sits between query decomposition (generator.py's LLM call #1) and prompt
    assembly (pipeline.py). Takes one or more sub-queries produced from the
    user's original question, embeds each one, runs a similarity search
    against ChromaDB for each, and merges the results into a single
    deduplicated, ranked list of chunks ready to inject into the LLM prompt.

Why retrieve per sub-query instead of one search on the raw question?
    A single user question can bundle multiple distinct information needs —
    e.g. "How does LoRA compare to full fine-tuning in terms of memory and
    performance?" really asks two things. Embedding the whole question as
    one vector blurs both needs together and similarity search tends to
    retrieve chunks that are generically related rather than precisely
    relevant to either sub-question. Decomposing first (generator.py) and
    retrieving per sub-query (here) gets sharper, more targeted results.

Why deduplicate?
    Running N sub-queries against the same collection commonly retrieves
    overlapping chunks — different sub-queries can be semantically close
    enough that they surface the same passage. Without deduplication the
    same chunk could appear twice in the final prompt, wasting context
    window space and skewing the LLM toward over-weighting that passage.
    Deduplication is by chunk ID (source + chunk_index), and when the same
    chunk is retrieved by more than one sub-query, the best (lowest
    distance) score is kept.

This module never talks to ChromaDB directly — all storage/search calls
go through vectorstore.py. It never talks to the LLM either — decomposition
happens upstream in generator.py, and prompt assembly happens downstream
in pipeline.py. retriever.py's only job is: sub-queries in, ranked
deduplicated chunks out.
"""

import os
from typing import Optional
from embedder import embed_text
import vectorstore


_TOP_K = int(os.getenv("TOP_K", "5"))


def retrieve_for_query(
    sub_query: str,
    top_k: int = _TOP_K,
    source_filter: Optional[str] = None
) -> list[dict]:
    """
    Run similarity search for a single query string.

    Embeds the query and delegates to vectorstore.query(). This is the
    building block retrieve() calls once per sub-query, but it's also
    useful standalone for a simple non-decomposed search (e.g. testing,
    or a future fast-path that skips decomposition for short questions).

    Args:
        sub_query:     a single question or sub-question string
        top_k:         how many chunks to retrieve (default from TOP_K env var)
        source_filter: optional filename to restrict search to one paper

    Returns:
        list of chunk dicts ranked most similar first (see vectorstore.query
        for the exact shape: id, text, source, page, chunk_index, distance)
    """
    query_vector = embed_text(sub_query)
    return vectorstore.query(query_vector, top_k=top_k, source_filter=source_filter)


def deduplicate_chunks(chunk_lists: list[list[dict]]) -> list[dict]:
    """
    Merge multiple ranked chunk lists into one deduplicated, ranked list.

    Takes the raw results from several retrieve_for_query() calls (one per
    sub-query) and flattens them into a single list, keeping only one entry
    per unique chunk ID. When a chunk was retrieved by more than one
    sub-query, its best (lowest) distance is kept — that chunk was a strong
    match for at least one sub-query, so it shouldn't be down-ranked just
    because it also loosely matched another.

    Args:
        chunk_lists: list of chunk-list results, one per sub-query, each
                     already ranked by vectorstore.query()

    Returns:
        single flat list of chunk dicts, deduplicated by "id", sorted by
        distance ascending (most similar first)

    Example:
        sub_query_1 → [chunk_A (0.30), chunk_B (0.45)]
        sub_query_2 → [chunk_B (0.38), chunk_C (0.50)]

        merged → [chunk_A (0.30), chunk_B (0.38), chunk_C (0.50)]
        (chunk_B kept once, using its better 0.38 score from sub_query_2)
    """
    best_by_id: dict[str, dict] = {}

    for chunks in chunk_lists:
        for chunk in chunks:
            chunk_id = chunk["id"]
            existing = best_by_id.get(chunk_id)

            if existing is None or chunk["distance"] < existing["distance"]:
                best_by_id[chunk_id] = chunk

    return sorted(best_by_id.values(), key=lambda c: c["distance"])


def retrieve(
    sub_queries: list[str],
    top_k: int = _TOP_K,
    source_filter: Optional[str] = None
) -> list[dict]:
    """
    Full retrieval pipeline: one or more sub-queries in, one deduplicated,
    ranked list of chunks out.

    This is the main entry point pipeline.py calls after generator.py has
    decomposed the user's question into 2-4 sub-queries (or returned the
    original question as a single-item list, if decomposition failed and
    fell back per the CLAUDE.md spec).

    Args:
        sub_queries:   list of 1-4 query strings to search for
        top_k:         chunks to retrieve per sub-query (default TOP_K env var)
        source_filter: optional filename to restrict search to one paper,
                        applied identically to every sub-query

    Returns:
        deduplicated list of chunk dicts, ranked by distance ascending,
        ready to be passed into pipeline.py for prompt assembly

    Notes:
        The returned list is NOT truncated to top_k overall — with up to
        4 sub-queries at top_k=5 each, this can return up to 20 unique
        chunks before deduplication (typically fewer once dedup runs).
        Trimming to however many chunks actually fit the prompt is
        pipeline.py's responsibility, not retriever.py's.
    """
    if not sub_queries:
        return []

    chunk_lists = [
        retrieve_for_query(sub_query, top_k=top_k, source_filter=source_filter)
        for sub_query in sub_queries
    ]

    return deduplicate_chunks(chunk_lists)
