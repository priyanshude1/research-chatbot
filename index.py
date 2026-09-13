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

    v2 addition: for each newly-indexed PDF, also generates a whole-paper
    summary (generator.generate_paper_summary) from its full extracted
    text and stores it as its own "type": "summary" chunk
    (vectorstore.add_summary), backing the get_paper_summary agent tool.
    A summary generation failure is logged and skipped rather than failing
    the whole run — the paper's content chunks are still indexed and
    usable for search either way.

Usage:
    python index.py                # incremental — skip files already indexed
    python index.py --force        # wipe the collection, reindex everything
    python index.py --summaries-only  # backfill missing summaries only,
                                       # no re-chunking/re-embedding
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from dotenv import load_dotenv
load_dotenv()

import chunker
import embedder
import generator
import vectorstore


_DATA_PATH = os.getenv("DATA_PATH", "./data")


def _generate_and_store_summary(filename: str, pdf_path: str, file_hash: str) -> bool:
    """
    Generate and store one paper's summary, logging and swallowing a
    RuntimeError rather than raising — shared by run_indexing() (summary
    right after a paper is freshly indexed) and backfill_summaries()
    (summary for a paper indexed earlier whose summary attempt failed).

    Returns:
        True if the summary was generated and stored, False if it failed.
    """
    print(f"  Generating summary for {filename}...")
    pages = chunker.extract_text_by_page(pdf_path)
    full_text = "\n".join(page["text"] for page in pages)
    try:
        summary = generator.generate_paper_summary(filename, full_text)
        vectorstore.add_summary(filename, summary, file_hash)
        print("  -> summary stored")
        return True
    except RuntimeError as e:
        print(f"  Summary generation failed ({e}) — continuing without one.")
        return False


def backfill_summaries() -> dict:
    """
    Generate summaries for already-indexed papers that don't have one yet.

    Does not touch content chunks or re-embed anything — for use after
    run_indexing()'s summary step failed for some/all papers (e.g. Groq
    was unreachable at index time) without redoing the expensive
    chunk/embed/store work that already succeeded.

    Returns:
        {"generated": list[str], "failed": list[str]}
    """
    generated: list[str] = []
    failed: list[str] = []

    for filename in vectorstore.get_all_sources():
        if vectorstore.get_summary(filename) is not None:
            continue

        pdf_path = os.path.join(_DATA_PATH, filename)
        if not os.path.exists(pdf_path):
            print(f"Skipping {filename}: source PDF not found at {pdf_path}")
            failed.append(filename)
            continue

        file_hash = chunker.compute_file_hash(pdf_path)
        if _generate_and_store_summary(filename, pdf_path, file_hash):
            generated.append(filename)
        else:
            failed.append(filename)

    print(f"\nDone. Generated {len(generated)} summaries, {len(failed)} failed.")
    return {"generated": generated, "failed": failed}


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

        _generate_and_store_summary(filename, pdf_path, file_hash)

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
    parser.add_argument(
        "--summaries-only",
        action="store_true",
        help="Backfill summaries for already-indexed papers that don't have one yet, "
             "without re-chunking or re-embedding content.",
    )
    args = parser.parse_args()

    if args.summaries_only:
        backfill_summaries()
    else:
        run_indexing(force_reindex=args.force)
