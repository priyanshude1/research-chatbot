"""
embedder.py — Chunk Embedding Using Sentence Transformers

Responsibility:
    Takes raw text chunks (produced by chunker.py) and converts each one
    into a single dense vector (embedding) that represents its semantic meaning.
    These vectors are what gets stored in ChromaDB and compared at query time.

Why a separate embedding model and not the LLM?
    The LLM (Llama 3.2 3B) is a generative model — it produces text.
    Embedding requires a different kind of model: one trained specifically
    to map text into a vector space where semantic similarity = geometric proximity.
    Sentence-transformers models are fine-tuned on datasets of similar/dissimilar
    sentence pairs using contrastive learning, making their vector space optimized
    for cosine similarity search. The LLM's internal representations are not
    suitable for this purpose.

Model choice: all-MiniLM-L6-v2
    - 6-layer MiniLM architecture (distilled from larger models)
    - Output dimension: 384 (compact, fast to search)
    - Size: ~90MB (loads quickly, fits in RAM with no GPU needed)
    - Performance: strong on semantic similarity benchmarks despite small size
    - Mean pooling: all token vectors averaged into one fixed-size vector

How embeddings work here (recap):
    Input chunk text (variable length)
          ↓
    Tokenizer splits into subword tokens
          ↓
    MiniLM encoder forward pass → (num_tokens, 384) contextual vectors
          ↓
    Mean pooling → average across token dimension → (384,) single vector
          ↓
    L2 normalization → unit vector (cosine similarity = dot product)
          ↓
    Stored in ChromaDB alongside original text and metadata

The same model must be used at both indexing time and query time.
If you change the embedding model, you must wipe and rebuild the ChromaDB index.
"""

import os
from typing import Union
from sentence_transformers import SentenceTransformer


# ── Module-level model instance ───────────────────────────────────────────────
# The model is loaded once when this module is first imported and reused
# for all subsequent calls. Loading a model is expensive (~1-2 seconds);
# reloading it on every embed call would be unacceptably slow.
#
# This pattern (module-level singleton) is standard for ML model serving.
# In a multi-worker FastAPI deployment you would manage this differently,
# but for a single-process local server it is correct and efficient.
# ─────────────────────────────────────────────────────────────────────────────

_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    """
    Lazy-load the sentence-transformer model on first use.

    Lazy loading means the model is not loaded at import time — only when
    embed_text() or embed_chunks() is first called. This keeps startup time
    fast if the embedder module is imported but not immediately used.

    Returns:
        the loaded SentenceTransformer model instance (cached after first call)
    """
    global _model
    if _model is None:
        print(f"Loading embedding model: {_MODEL_NAME}")
        _model = SentenceTransformer(_MODEL_NAME)
        print(f"Embedding model loaded. Output dimension: {_model.get_sentence_embedding_dimension()}")
    return _model


def embed_text(text: str) -> list[float]:
    """
    Embed a single string into a dense vector.

    Used at query time to embed the user's question (or each sub-query
    after decomposition) before similarity search in ChromaDB.

    Args:
        text: any string — a question, a sub-query, or a chunk of text

    Returns:
        list of floats of length 384 (the embedding vector)
        ChromaDB expects list[float], not a numpy array, so we convert.

    Example:
        vector = embed_text("What is scaled dot product attention?")
        # vector is a list of 384 floats
        # ChromaDB will find chunks whose vectors are closest to this one
    """
    model = _get_model()
    embedding = model.encode(text, normalize_embeddings=True)
    return embedding.tolist()


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """
    Embed a list of chunk dicts produced by chunker.py.

    Processes all chunks in a single batched call to the model rather than
    one at a time. Batching is significantly faster because the model can
    process multiple texts in parallel on GPU/CPU using matrix operations.

    Each input chunk dict (from chunker.py):
        {
            "text":        str,   ← this gets embedded
            "source":      str,
            "page":        int,
            "chunk_index": int,
            "file_hash":   str
        }

    Each output chunk dict adds one key:
        {
            "text":        str,
            "source":      str,
            "page":        int,
            "chunk_index": int,
            "file_hash":   str,
            "embedding":   list[float]   ← added here, length 384
        }

    Args:
        chunks: list of chunk dicts from chunker.chunk_document()

    Returns:
        same list of dicts with "embedding" key added to each

    Notes:
        - normalize_embeddings=True means vectors are L2-normalized (unit length)
          This means cosine similarity reduces to a simple dot product,
          which is what ChromaDB uses internally for faster computation.
        - show_progress_bar=True prints a tqdm progress bar for large batches
          useful when indexing 15 papers at once (hundreds of chunks)
    """
    model = _get_model()

    # extract just the text strings for batched encoding
    texts = [chunk["text"] for chunk in chunks]

    print(f"  Embedding {len(texts)} chunks...")
    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=True,
        batch_size=32       # process 32 chunks at a time — safe for CPU and GPU
    )

    # attach each embedding back to its corresponding chunk dict
    embedded_chunks = []
    for chunk, embedding in zip(chunks, embeddings):
        embedded_chunk = chunk.copy()           # don't mutate the original
        embedded_chunk["embedding"] = embedding.tolist()
        embedded_chunks.append(embedded_chunk)

    return embedded_chunks


def get_embedding_dimension() -> int:
    """
    Return the output vector dimension of the loaded embedding model.

    Useful for debugging and for verifying ChromaDB collection compatibility.
    If you switch embedding models, the dimension may change and the existing
    ChromaDB collection (built with the old dimension) will reject new vectors.

    Returns:
        int — 384 for all-MiniLM-L6-v2
    """
    return _get_model().get_sentence_embedding_dimension()


# ── quick test ────────────────────────────────────────────────────────────────
# Run this file directly to verify the embedding model loads and produces
# vectors of the correct shape.
# Usage: python src/embedder.py
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Running embedder test...\n")

    # test single embedding
    test_sentence = "Attention mechanisms allow transformers to focus on relevant tokens."
    vector = embed_text(test_sentence)

    print(f"Input:           {test_sentence}")
    print(f"Vector length:   {len(vector)}")
    print(f"First 5 values:  {vector[:5]}")
    print(f"Expected dim:    {get_embedding_dimension()}")
    print()

    # test that similar sentences produce closer vectors than dissimilar ones
    # this sanity-checks that the model is working semantically
    from sentence_transformers import util

    v1 = embed_text("How does self-attention work in transformers?")
    v2 = embed_text("Explain the attention mechanism in neural networks.")
    v3 = embed_text("What is the capital of France?")

    import torch
    t1 = torch.tensor(v1)
    t2 = torch.tensor(v2)
    t3 = torch.tensor(v3)

    sim_related   = util.cos_sim(t1, t2).item()
    sim_unrelated = util.cos_sim(t1, t3).item()

    print(f"Similarity (related sentences):   {sim_related:.4f}   ← should be high")
    print(f"Similarity (unrelated sentences): {sim_unrelated:.4f}   ← should be lower")
    print()
    print("If related > unrelated, the embedding model is working correctly.")
