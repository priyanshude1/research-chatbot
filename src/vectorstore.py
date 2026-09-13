"""
vectorstore.py — ChromaDB Persistent Storage Wrapper

Responsibility:
    Takes embedded chunks (produced by embedder.py) and stores them in a
    persistent ChromaDB collection on disk. Also provides the similarity
    search interface that retriever.py builds on top of, and the hash
    lookup used by index.py to support incremental indexing.

Why ChromaDB?
    ChromaDB is an open-source, file-based vector database — no managed
    service, no external cost, no network dependency. Its persistent client
    writes directly to a folder on disk (./chroma_db/), which fits the
    "local-first, zero data leakage" goal of this project. It stores each
    chunk's vector alongside arbitrary metadata (source, page, chunk_index,
    file_hash) and the original text, so a similarity search returns
    everything retriever.py and generator.py need in one call.

Why wrap ChromaDB instead of calling it directly everywhere?
    Isolating all ChromaDB-specific code (client setup, collection naming,
    the add/query call signatures) in one module means the rest of the
    pipeline never imports chromadb directly. If the vector database were
    ever swapped out, only this file would need to change.

Incremental indexing (recap from chunker.py):
    Every chunk carries a file_hash (MD5 of the source PDF). Before
    embedding and indexing a document, index.py can call
    get_indexed_hashes() to check whether that hash is already present
    in the collection. If so, the file is skipped — this avoids
    re-embedding and re-storing unchanged papers every time indexing runs.

Storage layout:
    ./chroma_db/                    ← persistent on-disk storage (gitignored)
        collection: "research_papers"
            per chunk:
                id:        deterministic string, e.g. "vaswani_2017.pdf_0"
                embedding: list[float], length 384
                document:  the chunk's raw text
                metadata:  {source, page, chunk_index, file_hash}
"""

import os
import threading
from typing import Optional
import chromadb
from chromadb.api.models.Collection import Collection

import embedder


_CHROMA_PATH = os.getenv("CHROMA_PATH", "./chroma_db")
_COLLECTION_NAME = os.getenv("COLLECTION_NAME", "research_papers")

# ── Module-level client/collection singletons ──────────────────────────────────
# Like the embedding model in embedder.py, the ChromaDB client is expensive
# to set up (it opens/creates the on-disk database). It is created once on
# first use and reused for every subsequent call in this process.
# ─────────────────────────────────────────────────────────────────────────────
_client: chromadb.ClientAPI | None = None
_collection: Collection | None = None
_init_lock = threading.Lock()


def _get_collection() -> Collection:
    """
    Lazy-load the persistent ChromaDB client and collection on first use.

    get_or_create_collection() means this is safe to call whether the
    collection already exists on disk (subsequent runs) or not (first run).

    FastAPI runs sync endpoints in a threadpool, so concurrent first-callers
    can otherwise both see `_collection is None` and race into
    chromadb.PersistentClient() for the same path at once — Chroma's own
    client registry isn't safe against that (raises KeyError). The lock plus
    re-check inside it (double-checked locking) ensures only one thread ever
    constructs the client; every other caller just waits and reuses it.

    Returns:
        the ChromaDB Collection instance (cached after first call)
    """
    global _client, _collection
    if _collection is None:
        with _init_lock:
            if _collection is None:
                print(f"Opening ChromaDB at: {_CHROMA_PATH}")
                _client = chromadb.PersistentClient(path=_CHROMA_PATH)
                _collection = _client.get_or_create_collection(
                    name=_COLLECTION_NAME,
                    metadata={"hnsw:space": "cosine"}  # explicit cosine similarity
                )
                print(f"Collection '{_COLLECTION_NAME}' ready. Existing chunks: {_collection.count()}")
    return _collection


def _make_chunk_id(chunk: dict) -> str:
    """
    Build a deterministic, unique ID for a chunk.

    ChromaDB requires a unique string ID per stored item. Combining the
    source filename with the chunk_index guarantees uniqueness within a
    document and makes the ID human-readable for debugging (you can tell
    which paper and which chunk an ID refers to just by looking at it).

    Args:
        chunk: a chunk dict with at least "source" and "chunk_index" keys

    Returns:
        string ID, e.g. "vaswani_2017.pdf_42"
    """
    return f"{chunk['source']}_{chunk['chunk_index']}"


def add_chunks(chunks: list[dict]) -> int:
    """
    Store a list of embedded chunks in ChromaDB.

    Expects chunks that have already been through embedder.embed_chunks(),
    i.e. each dict has an "embedding" key in addition to the metadata
    produced by chunker.py.

    Each input chunk dict:
        {
            "text":        str,
            "source":      str,
            "page":        int,
            "chunk_index": int,
            "file_hash":   str,
            "embedding":   list[float]
        }

    ChromaDB separates what it stores into three parallel lists:
        - ids:        unique identifier per item
        - embeddings: the vectors used for similarity search
        - documents:  the raw text (returned alongside search results)
        - metadatas:  everything else, used for filtering (e.g. by source)

    Args:
        chunks: list of embedded chunk dicts from embedder.embed_chunks()

    Returns:
        number of chunks added

    Notes:
        add() upserts by ID — calling this twice with the same chunks
        (same source + chunk_index) overwrites rather than duplicates.
    """
    if not chunks:
        return 0

    collection = _get_collection()

    ids = [_make_chunk_id(chunk) for chunk in chunks]
    embeddings = [chunk["embedding"] for chunk in chunks]
    documents = [chunk["text"] for chunk in chunks]
    metadatas = [
        {
            "source":      chunk["source"],
            "page":        chunk["page"],
            "chunk_index": chunk["chunk_index"],
            "file_hash":   chunk["file_hash"],
        }
        for chunk in chunks
    ]

    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas
    )

    return len(chunks)


def get_indexed_hashes() -> set[str]:
    """
    Return the set of all file_hash values currently stored in the collection.

    Used by index.py to implement incremental indexing: before processing
    a PDF, compute its hash (chunker.compute_file_hash) and check whether
    it's already in this set. If so, skip the file entirely — no re-chunking,
    re-embedding, or re-storing needed.

    Returns:
        set of MD5 hash strings, e.g. {"a3f2c1d4...", "b7e91a02..."}
        empty set if the collection has no chunks yet

    Notes:
        get() with no filters and only the "metadatas" field pulls back
        every stored item's metadata but not its (much larger) embedding
        vector, keeping this call cheap even for large collections.
    """
    collection = _get_collection()

    if collection.count() == 0:
        return set()

    results = collection.get(include=["metadatas"])
    return {metadata["file_hash"] for metadata in results["metadatas"]}


def query(
    query_embedding: list[float],
    top_k: int = 5,
    source_filter: Optional[str] = None
) -> list[dict]:
    """
    Run a similarity search against the collection.

    This is the core retrieval primitive that retriever.py calls once per
    sub-query (after query decomposition). It takes an already-embedded
    query vector — embedding happens in embedder.py, not here — and
    returns the top_k most similar chunks.

    Args:
        query_embedding: the embedded query, list[float] of length 384
        top_k:           how many results to return (default 5, per CLAUDE.md)
        source_filter:   optional filename — if given, restricts the search
                          to chunks from that one source document (used for
                          paper-specific queries, e.g. "in the LoRA paper...")

    Returns:
        list of result dicts, ranked most similar first:
            [
                {
                    "id":         str,      # e.g. "vaswani_2017.pdf_12"
                    "text":       str,      # the chunk's raw text
                    "source":     str,
                    "page":       int,
                    "chunk_index": int,
                    "distance":   float     # cosine distance, lower = more similar
                },
                ...
            ]

    Notes:
        ChromaDB's `where` filter takes a dict; passing None means no filter.
        Cosine distance (not similarity) is what ChromaDB returns — smaller
        values mean closer vectors. Deduplication across multiple sub-query
        results happens in retriever.py, not here.

        Always excludes "type": "summary" chunks (added by add_summary() for
        v2's get_paper_summary tool) — a pre-generated summary is a
        different kind of content than the page-level chunks this function
        searches over, and mixing it into ordinary similarity search results
        would surface a whole-paper summary alongside specific passages.
        Summaries are only ever fetched directly, by exact filename, via
        get_summary().
    """
    collection = _get_collection()

    exclude_summaries = {"type": {"$ne": "summary"}}
    where = (
        {"$and": [exclude_summaries, {"source": source_filter}]}
        if source_filter
        else exclude_summaries
    )

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        where=where
    )

    # results come back as parallel lists nested one level for the single query
    ids = results["ids"][0]
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    return [
        {
            "id":          ids[i],
            "text":        documents[i],
            "source":      metadatas[i]["source"],
            "page":        metadatas[i]["page"],
            "chunk_index": metadatas[i]["chunk_index"],
            "distance":    distances[i],
        }
        for i in range(len(ids))
    ]


def get_all_sources() -> list[str]:
    """
    Return the sorted list of unique source filenames in the collection.

    Used by the GET /documents endpoint and to populate the paper-filter
    dropdown in the frontend UI.

    Returns:
        sorted list of filenames, e.g. ["bert_2018.pdf", "vaswani_2017.pdf"]
        empty list if the collection has no chunks yet
    """
    collection = _get_collection()

    if collection.count() == 0:
        return []

    results = collection.get(include=["metadatas"])
    sources = {metadata["source"] for metadata in results["metadatas"]}
    return sorted(sources)


def get_summary(filename: str) -> Optional[dict]:
    """
    Retrieve the pre-generated summary chunk for one paper, if one exists.

    Added for v2's get_paper_summary tool (tools.py). Unlike query(), this
    is a metadata-only lookup — no embedding involved — because a summary
    is fetched by exact filename match, not similarity search. index.py is
    expected to store each paper's summary as its own chunk carrying
    {"source": filename, "type": "summary"} metadata (in addition to the
    ordinary content chunks for that same source, which have no "type" key).

    Args:
        filename: exact source filename, e.g. "vaswani_2017.pdf"

    Returns:
        {"source": filename, "text": summary_text} if a summary chunk is
        found, otherwise None (e.g. filename doesn't exist, or index.py
        hasn't generated summaries yet)
    """
    collection = _get_collection()

    if collection.count() == 0:
        return None

    results = collection.get(
        where={"$and": [{"source": filename}, {"type": "summary"}]},
        include=["documents"],
    )

    if not results["ids"]:
        return None

    return {"source": filename, "text": results["documents"][0]}


def add_summary(filename: str, summary_text: str, file_hash: str) -> None:
    """
    Store one paper's pre-generated summary as its own chunk.

    Added for v2's index.py summary-generation step (CLAUDE.md: index.py
    "generates and stores a pre-computed summary for each paper at indexing
    time, stored as a special chunk with type: 'summary' metadata"). Kept
    separate from add_chunks() rather than folded into it, since a summary
    isn't a content chunk — it has no page or chunk_index, is embedded from
    its own text rather than a slice of the paper, and get_summary() looks
    it up by source+type rather than by similarity search. query() excludes
    "type": "summary" chunks so this never surfaces in ordinary search
    results.

    Args:
        filename:     source filename, e.g. "vaswani_2017.pdf"
        summary_text: the generated summary (generator.generate_paper_summary)
        file_hash:    MD5 hash of the source PDF, stored alongside the
                      summary for consistency with content chunk metadata

    Notes:
        add() upserts by ID (f"{filename}_summary"), so re-summarizing the
        same paper (e.g. after --force reindex) overwrites rather than
        duplicates.
    """
    collection = _get_collection()
    embedding = embedder.embed_text(summary_text)

    collection.add(
        ids=[f"{filename}_summary"],
        embeddings=[embedding],
        documents=[summary_text],
        metadatas=[{"source": filename, "type": "summary", "file_hash": file_hash}],
    )


def get_total_chunks() -> int:
    """
    Return the total number of chunks currently stored in the collection.

    Used by the GET /documents endpoint to report corpus size.

    Returns:
        int count of stored chunks
    """
    return _get_collection().count()


def reset_collection() -> None:
    """
    Delete and recreate the collection, wiping all stored chunks.

    Used when force_reindex=True on POST /index — a full rebuild needs to
    start from an empty collection rather than upserting on top of
    potentially stale data (e.g. after changing chunk_size or the
    embedding model, which changes vector dimensions).

    Notes:
        This does not delete the ./chroma_db/ folder itself, only the
        named collection within it, then immediately recreates it empty.
    """
    global _collection
    client = chromadb.PersistentClient(path=_CHROMA_PATH) if _client is None else _client

    print(f"Resetting collection '{_COLLECTION_NAME}'...")
    client.delete_collection(name=_COLLECTION_NAME)
    _collection = client.get_or_create_collection(
        name=_COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )
    print("Collection reset. Chunk count: 0")
