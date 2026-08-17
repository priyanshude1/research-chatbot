"""
chunker.py — Document Parsing and Text Chunking

Responsibility:
    Takes raw PDF files from ./data/, extracts clean text page by page,
    and splits that text into overlapping chunks of a fixed token size.

Why chunking?
    LLMs and embedding models have a maximum input size. A full research
    paper (10,000+ tokens) cannot be embedded as one unit — the embedding
    would lose fine-grained detail and similarity search would be imprecise.
    Smaller chunks give more precise retrieval at the cost of losing
    broader context, which is why we use overlap to bridge chunk boundaries.

Chunking strategy:
    - Chunk size:    500 tokens (balances context richness vs retrieval precision)
    - Overlap:       100 tokens (last 100 tokens of chunk N = first 100 of chunk N+1)
    - Unit:          words (approximation of tokens — true tokenization not needed here
                     since sentence-transformers handles its own tokenization internally)

Output per chunk:
    A dictionary with:
        - text:        the raw chunk text (fed to embedder)
        - source:      filename of the source PDF
        - page:        page number the chunk originated from
        - chunk_index: sequential index of this chunk within the document
        - file_hash:   MD5 hash of the source file (used for incremental indexing)
"""

import os
import hashlib
from typing import Generator
import fitz  # PyMuPDF — imported as fitz for historical reasons


def compute_file_hash(pdf_path: str) -> str:
    """
    Compute the MD5 hash of a PDF file's raw bytes.

    Used for incremental indexing — if a file's hash is already in ChromaDB,
    it has been indexed before and can be skipped. If the hash changed,
    the file was updated and needs reindexing.

    Args:
        pdf_path: absolute or relative path to the PDF file

    Returns:
        hex string of the MD5 hash (e.g. 'a3f2c1d4...')
    """
    with open(pdf_path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def extract_text_by_page(pdf_path: str) -> list[dict]:
    """
    Open a PDF and extract raw text from each page separately.

    We extract page by page (not the whole document at once) so that
    each chunk can carry accurate page number metadata. This makes it
    possible to tell the user exactly which page of which paper an
    answer came from.

    Args:
        pdf_path: path to the PDF file

    Returns:
        list of dicts, one per page:
            [
                {"page": 1, "text": "Introduction\nTransformers are..."},
                {"page": 2, "text": "Related Work\n..."},
                ...
            ]

    Notes:
        - fitz.open() loads the PDF into memory
        - page.get_text() extracts raw unicode text from the page
        - Pages with no extractable text (e.g. scanned images) return empty strings
          and are filtered out
    """
    pages = []
    doc = fitz.open(pdf_path)

    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text()

        # skip empty pages (scanned images, blank pages, figure-only pages)
        if text.strip():
            pages.append({
                "page": page_num + 1,  # 1-indexed for human readability
                "text": text.strip()
            })

    doc.close()
    return pages


def split_into_chunks(
    text: str,
    chunk_size: int = 500,
    overlap: int = 100
) -> list[str]:
    """
    Split a string of text into overlapping fixed-size chunks.

    We split by words (not characters or sentences) as a simple approximation
    of tokens. True tokenization (using the actual sentence-transformer tokenizer)
    is not necessary here — slight size variation per chunk is acceptable and
    the embedding model handles variable-length inputs internally.

    How overlap works:
        chunk 0: words[0   : 500]
        chunk 1: words[400 : 900]    ← starts 100 words before chunk 0 ends
        chunk 2: words[800 : 1300]   ← starts 100 words before chunk 1 ends

    This ensures that information near a chunk boundary appears in both
    adjacent chunks, preventing retrieval misses for queries that happen
    to span a boundary.

    Args:
        text:       raw text string to split
        chunk_size: number of words per chunk (default 500)
        overlap:    number of words shared between adjacent chunks (default 100)

    Returns:
        list of chunk strings
    """
    words = text.split()
    chunks = []
    step = chunk_size - overlap  # how far to advance the window each iteration

    for start in range(0, len(words), step):
        end = start + chunk_size
        chunk = " ".join(words[start:end])

        # only add non-empty chunks
        if chunk.strip():
            chunks.append(chunk)

        # stop if we've passed the end of the text
        if end >= len(words):
            break

    return chunks


def chunk_document(
    pdf_path: str,
    chunk_size: int = 500,
    overlap: int = 100
) -> list[dict]:
    """
    Full pipeline for one PDF: extract text → split into chunks → attach metadata.

    This is the main function called by index.py for each PDF in ./data/.
    It combines extract_text_by_page() and split_into_chunks() and attaches
    all metadata needed by ChromaDB for storage and later retrieval.

    Args:
        pdf_path:   path to the PDF file
        chunk_size: words per chunk (passed through to split_into_chunks)
        overlap:    overlap words between chunks (passed through to split_into_chunks)

    Returns:
        list of chunk dicts, each containing:
            {
                "text":        str,   # the chunk text — fed to the embedding model
                "source":      str,   # filename e.g. "vaswani_2017.pdf"
                "page":        int,   # page number this chunk came from
                "chunk_index": int,   # sequential index within this document
                "file_hash":   str    # MD5 hash of the source PDF
            }

    Example output:
        [
            {
                "text": "Attention mechanisms allow a model to...",
                "source": "vaswani_2017.pdf",
                "page": 2,
                "chunk_index": 0,
                "file_hash": "a3f2c1d4..."
            },
            ...
        ]
    """
    filename = os.path.basename(pdf_path)
    file_hash = compute_file_hash(pdf_path)
    pages = extract_text_by_page(pdf_path)

    all_chunks = []
    chunk_index = 0  # global chunk counter across all pages of this document

    for page_data in pages:
        page_chunks = split_into_chunks(
            page_data["text"],
            chunk_size=chunk_size,
            overlap=overlap
        )

        for chunk_text in page_chunks:
            all_chunks.append({
                "text":        chunk_text,
                "source":      filename,
                "page":        page_data["page"],
                "chunk_index": chunk_index,
                "file_hash":   file_hash
            })
            chunk_index += 1

    return all_chunks


def chunk_all_documents(
    data_dir: str = "./data",
    chunk_size: int = 500,
    overlap: int = 100
) -> Generator[tuple[str, list[dict]], None, None]:
    """
    Generator that iterates over all PDFs in data_dir and yields chunks per document.

    Using a generator (yield) instead of returning a giant list means we process
    one PDF at a time without loading everything into memory simultaneously.
    For 10-15 papers this doesn't matter much, but it's the correct pattern
    for larger corpora.

    Args:
        data_dir:   path to the folder containing PDF files
        chunk_size: words per chunk
        overlap:    overlap words between chunks

    Yields:
        tuple of (filename, list_of_chunks) for each PDF found in data_dir

    Usage:
        for filename, chunks in chunk_all_documents("./data"):
            print(f"{filename}: {len(chunks)} chunks")
    """
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    pdf_files = [f for f in os.listdir(data_dir) if f.endswith(".pdf")]

    if not pdf_files:
        raise ValueError(f"No PDF files found in {data_dir}")

    for filename in sorted(pdf_files):
        pdf_path = os.path.join(data_dir, filename)
        print(f"Chunking: {filename}")

        chunks = chunk_document(pdf_path, chunk_size=chunk_size, overlap=overlap)
        print(f"  → {len(chunks)} chunks extracted")

        yield filename, chunks


# ── quick test ────────────────────────────────────────────────────────────────
# Run this file directly to test chunking on whatever PDFs are in ./data/
# Usage: python src/chunker.py
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Running chunker test...\n")

    for filename, chunks in chunk_all_documents("./data"):
        print(f"  First chunk preview:")
        print(f"  Source:  {chunks[0]['source']}")
        print(f"  Page:    {chunks[0]['page']}")
        print(f"  Hash:    {chunks[0]['file_hash']}")
        print(f"  Text:    {chunks[0]['text'][:200]}...")
        print()
