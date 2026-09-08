"""
tools.py — The 4 Tool Functions the ReAct Agent Can Call

Responsibility:
    Implements the 4 tools specified in CLAUDE.md, each a plain Python
    function: search_papers, get_paper_summary, list_papers, web_search.
    agent.py's ReAct loop parses the LLM's structured {"action": ...} output
    and dispatches to one of these via the TOOLS registry at the bottom of
    this file — agent.py never needs an if/elif chain of tool names.

    Every tool returns a plain JSON-serializable dict. agent.py feeds that
    straight back to the LLM as the "observation" for the next loop
    iteration (typically via json.dumps), so the shape of what these
    functions return IS the shape the LLM sees — keep it lean and
    consistent (either the requested data, or an "error"/"message" key
    explaining why there isn't any) rather than raising exceptions for
    ordinary "not found" cases.

Why do search_papers and web_search take a single query instead of the
multi-subquery decomposition pipeline.py uses?
    pipeline.py's decompose-then-retrieve flow (generator.decompose_query +
    retriever.retrieve) exists to handle one big multi-part question in a
    single non-agentic pass. The agent doesn't need that: it can reason
    about a complex question step by step, issuing several focused
    single-query tool calls across ReAct iterations instead of decomposing
    up front. So search_papers calls retriever.retrieve_for_query()
    directly (embed one query, search once) rather than retriever.retrieve().
"""

import os
import requests
import xml.etree.ElementTree as ET

import retriever
import vectorstore


_TOP_K = int(os.getenv("TOP_K", "5"))
_ARXIV_API_URL = "http://export.arxiv.org/api/query"
_ATOM_NS = "{http://www.w3.org/2005/Atom}"


def search_papers(query: str, filter_source: str | None = None) -> dict:
    """
    Semantic search over the indexed paper corpus.

    Embeds `query` and runs one similarity search against ChromaDB (via
    retriever.retrieve_for_query, which itself delegates to embedder.py +
    vectorstore.py). Used by the agent for any factual question about
    paper content.

    Args:
        query:         the search text — the agent's own phrasing of what
                        it needs to find, not necessarily the user's raw
                        question verbatim
        filter_source: optional filename to restrict the search to one
                        paper (e.g. the agent already knows the user is
                        asking specifically about "vaswani_2017.pdf")

    Returns:
        {
            "query":   str,
            "results": [
                {"source": str, "page": int, "text": str, "distance": float},
                ...
            ]
        }
        "results" is [] (with a "message" key) if nothing matched — this
        is a normal outcome (e.g. filter_source excluded everything), not
        an error.
    """
    chunks = retriever.retrieve_for_query(query, top_k=_TOP_K, source_filter=filter_source)

    if not chunks:
        return {
            "query": query,
            "results": [],
            "message": "No matching chunks found in the indexed papers.",
        }

    return {
        "query": query,
        "results": [
            {
                "source": chunk["source"],
                "page": chunk["page"],
                "text": chunk["text"],
                "distance": chunk["distance"],
            }
            for chunk in chunks
        ],
    }


def get_paper_summary(filename: str) -> dict:
    """
    Retrieve the pre-generated summary for one paper.

    Looks up a summary chunk (vectorstore.get_summary) by exact filename
    match — no embedding/similarity search involved, since a summary
    request names a specific paper rather than a topic. Used when the
    user asks to summarize a specific paper.

    Args:
        filename: exact source filename, e.g. "vaswani_2017.pdf" — the
                  agent should get this from list_papers if it isn't
                  already certain of the exact name

    Returns:
        {"filename": str, "summary": str} on success, or
        {"filename": str, "error": str} if no summary chunk exists for
        that filename (wrong filename, or the corpus hasn't been indexed
        with summary generation yet).
    """
    summary = vectorstore.get_summary(filename)

    if summary is None:
        return {
            "filename": filename,
            "error": (
                f"No summary found for '{filename}'. Double-check the "
                "filename with list_papers, or a summary may not have "
                "been generated for this paper yet."
            ),
        }

    return {"filename": filename, "summary": summary["text"]}


def list_papers() -> dict:
    """
    List every paper currently indexed in the corpus.

    Thin wrapper over vectorstore.get_all_sources() — used when the user
    asks what papers are available, or when the agent needs an exact
    filename to pass to get_paper_summary or search_papers' filter_source.

    Returns:
        {"papers": list[str], "count": int}
    """
    papers = vectorstore.get_all_sources()
    return {"papers": papers, "count": len(papers)}


def _parse_arxiv_feed(xml_text: str) -> list[dict]:
    """
    Parse an arXiv Atom API response into a list of plain paper dicts.

    Kept separate from web_search() so the "talk to arXiv" step (network,
    error handling) and the "make sense of the XML" step can be reasoned
    about independently.

    Args:
        xml_text: raw Atom XML response body from the arXiv API

    Returns:
        list of {"title": str, "abstract": str, "link": str, "published": str},
        one per <entry> in the feed, in the order arXiv returned them
        (relevance-ranked by arXiv's own search).
    """
    root = ET.fromstring(xml_text)

    entries = []
    for entry in root.findall(f"{_ATOM_NS}entry"):
        title = entry.findtext(f"{_ATOM_NS}title", default="").strip()
        abstract = entry.findtext(f"{_ATOM_NS}summary", default="").strip()
        link = entry.findtext(f"{_ATOM_NS}id", default="").strip()
        published = entry.findtext(f"{_ATOM_NS}published", default="").strip()
        entries.append({
            "title": title,
            "abstract": abstract,
            "link": link,
            "published": published,
        })

    return entries


def web_search(query: str) -> dict:
    """
    Search arXiv for papers matching `query`, outside the local corpus.

    Used when the user asks about recent papers or topics not covered by
    the indexed PDFs. Calls the free, keyless arXiv public API directly —
    no API key needed, matching CLAUDE.md's tool spec. Returns metadata
    only (title, abstract, link) — never the full paper text.

    Args:
        query: search text, passed to arXiv's "all fields" search

    Returns:
        {"query": str, "results": [...]} — up to 5 entries (see
        _parse_arxiv_feed for each entry's shape), or
        {"query": str, "results": [], "error": str} if the arXiv request
        itself failed (network issue, arXiv downtime, etc.) — this is
        reported back to the LLM as a tool observation rather than raised,
        so the agent can tell the user the web search wasn't available.
    """
    params = {
        "search_query": f"all:{query}",
        "start": 0,
        "max_results": 5,
    }

    try:
        response = requests.get(_ARXIV_API_URL, params=params, timeout=10)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        return {"query": query, "results": [], "error": f"arXiv request failed: {e}"}

    return {"query": query, "results": _parse_arxiv_feed(response.text)}


# ── Tool dispatch registry ───────────────────────────────────────────────────
# agent.py parses the LLM's {"action": "<name>", ...other kwargs} output and
# calls TOOLS["<name>"](**other_kwargs) — this dict is the single source of
# truth for which action names are valid, so adding/removing a tool means
# changing this file alone, not agent.py's dispatch logic.
# ─────────────────────────────────────────────────────────────────────────────
TOOLS = {
    "search_papers": search_papers,
    "get_paper_summary": get_paper_summary,
    "list_papers": list_papers,
    "web_search": web_search,
}
