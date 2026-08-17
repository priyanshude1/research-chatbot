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
from typing import Optional
import chromadb
from chromadb.api.models.Collection import Collection


_CHROMA_PATH = os.getenv("CHROMA_PATH", "./chroma_db")
_COLLECTION_NAME = os.getenv("COLLECTION_NAME", "research_papers")

# ── Module-level client/collection singletons ──────────────────────────────────
# Like the embedding model in embedder.py, the ChromaDB client is expensive
# to set up (it opens/creates the on-disk database). It is created once on
# first use and reused for every subsequent call in this process.
# ─────────────────────────────────────────────────────────────────────────────
_client: chromadb.ClientAPI | None = None
_collection: Collection | None = None


def _get_collection() -> Collection:
    """
    Lazy-load the persistent ChromaDB client and collection on first use.

    get_or_create_collection() means this is safe to call whether the
    collection already exists on disk (subsequent runs) or not (first run).

    Returns:
        the ChromaDB Collection instance (cached after first call)
    """
    global _client, _collection
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
    """
    collection = _get_collection()

    where = {"source": source_filter} if source_filter else None

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


# ── quick test ────────────────────────────────────────────────────────────────
# Run this file directly to index every PDF in ./data (skipping any whose
# file_hash is already stored) and run a sample similarity search against
# the real corpus. This exercises the full chunker → embedder → vectorstore
# pipeline end to end, not just this module in isolation.
# Usage: python src/vectorstore.py
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import os
    from chunker import compute_file_hash, chunk_document
    from embedder import embed_text, embed_chunks

    print("Running vectorstore test on ./data...\n")

    already_indexed = get_indexed_hashes()
    print(f"Chunks already in collection: {get_total_chunks()}")
    print(f"Known file hashes: {len(already_indexed)}\n")

    data_dir = "./data"
    pdf_files = sorted(f for f in os.listdir(data_dir) if f.endswith(".pdf"))

    for filename in pdf_files:
        pdf_path = os.path.join(data_dir, filename)

        # check the hash BEFORE parsing/chunking — that's the expensive part.
        # skipping here avoids re-running PyMuPDF text extraction and chunk
        # splitting on every rerun for files that are already indexed.
        file_hash = compute_file_hash(pdf_path)
        if file_hash in already_indexed:
            print(f"  Skipping {filename} (already indexed)\n")
            continue

        print(f"Chunking: {filename}")
        chunks = chunk_document(pdf_path)
        print(f"  → {len(chunks)} chunks extracted")

        embedded_chunks = embed_chunks(chunks)
        added = add_chunks(embedded_chunks)
        print(f"  Indexed {filename}: {added} chunks stored\n")

    print(f"Total chunks in collection: {get_total_chunks()}")
    print(f"Sources: {get_all_sources()}\n")

    # sample query against the now-populated collection
    test_question = "Why is masking necessary?"
    query_vector = embed_text(test_question)
    results = query(query_vector, top_k=5)

    print(f"Query: '{test_question}'")
    for r in results:
        print(f"  [{r['distance']:.4f}] {r['source']} p.{r['page']}: {r['text'][:80]}...")
