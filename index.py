"""
index.py — CLI Entry Point for Indexing ./data

Responsibility:
    Run this file to index every PDF in DATA_PATH into ChromaDB, driving
    chunker.py -> embedder.py -> vectorstore.py for each document in turn.
    This is the "Phase 1 — Indexing" step from CLAUDE.md's architecture
    diagram, and the CLI counterpart of POST /index.

    Incremental indexing: each PDF's MD5 hash (chunker.compute_file_hash)
    is checked against the hashes already stored in the collection
    (vectorstore.get_indexed_hashes). A file whose hash is already present
    is skipped — only new or changed PDFs are chunked and embedded. This
    means dropping new papers into ./data and rerunning this script only
    does the work the new papers actually require.

Usage:
    python index.py            # incremental — skip files already indexed
    python index.py --force    # wipe the collection, reindex everything
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from dotenv import load_dotenv
load_dotenv()

import chunker
import embedder
import vectorstore


_DATA_PATH = os.getenv("DATA_PATH", "./data")


def run_indexing(force_reindex: bool = False) -> dict:
    """
    Index every PDF in DATA_PATH, skipping files already indexed unless
    force_reindex is set.

    Args:
        force_reindex: if True, wipes the existing collection first
                        (vectorstore.reset_collection) so every PDF is
                        reindexed from scratch regardless of hash

    Returns:
        {"indexed": list[str], "skipped": list[str]} — same shape as the
        POST /index response defined in CLAUDE.md, so api/main.py can call
        run_indexing() directly and return its result unchanged.
    """
    if force_reindex:
        vectorstore.reset_collection()

    already_indexed = vectorstore.get_indexed_hashes()

    indexed: list[str] = []
    skipped: list[str] = []

    if not os.path.exists(_DATA_PATH):
        raise FileNotFoundError(f"Data directory not found: {_DATA_PATH}")

    pdf_files = sorted(f for f in os.listdir(_DATA_PATH) if f.endswith(".pdf"))
    if not pdf_files:
        print(f"No PDF files found in {_DATA_PATH}")
        return {"indexed": indexed, "skipped": skipped}

    for filename in pdf_files:
        pdf_path = os.path.join(_DATA_PATH, filename)

        # hash check happens before chunking — chunking/parsing is the
        # expensive part, so skip it entirely for already-indexed files
        file_hash = chunker.compute_file_hash(pdf_path)
        if file_hash in already_indexed:
            print(f"Skipping {filename} (already indexed)")
            skipped.append(filename)
            continue

        print(f"Indexing {filename}...")
        chunks = chunker.chunk_document(pdf_path)
        embedded_chunks = embedder.embed_chunks(chunks)
        added = vectorstore.add_chunks(embedded_chunks)
        print(f"  -> {added} chunks stored")

        indexed.append(filename)
        already_indexed.add(file_hash)

    print(f"\nDone. Indexed {len(indexed)}, skipped {len(skipped)}.")
    print(f"Total chunks in collection: {vectorstore.get_total_chunks()}")
    print(f"Sources: {vectorstore.get_all_sources()}")

    return {"indexed": indexed, "skipped": skipped}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Index PDF research papers into ChromaDB.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Wipe the existing collection and reindex every PDF in DATA_PATH from scratch.",
    )
    args = parser.parse_args()

    run_indexing(force_reindex=args.force)
