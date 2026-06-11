"""Semantic retrieval using SentenceTransformers and FAISS index stored on disk."""

from __future__ import annotations

import os
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

CHUNK_SIZE = 400
CHUNK_OVERLAP = 80
TOP_K_CHUNKS = 5
MAX_RAG_CONTEXT_CHARS = 28_000


def _iter_chunks(text: str) -> list[str]:
    # Word-level chunking helper kept for reference / compat
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        chunk_words = words[start : start + CHUNK_SIZE]
        chunks.append(" ".join(chunk_words))
        if start + CHUNK_SIZE >= len(words):
            break
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


def retrieve_rag_context(kb_docs: list, query: str, api_key: str = "") -> str:
    """
    Load the FAISS index and chunk metadata from disk, embed the query,
    perform a cosine similarity search, and return the formatted top matching chunks.
    """
    index_path = "vector_store/index.faiss"
    chunks_path = "vector_store/chunks.pkl"

    if not os.path.exists(index_path) or not os.path.exists(chunks_path):
        return "(No PDFs uploaded or vector store not initialized. Please upload PDFs first.)"

    q = (query or "").strip()
    if not q:
        return ""

    try:
        # Load FAISS index and metadata
        index = faiss.read_index(index_path)
        with open(chunks_path, "rb") as f:
            chunks = pickle.load(f)

        if not chunks:
            return ""

        # Initialize local SentenceTransformer model
        model = SentenceTransformer("all-MiniLM-L6-v2")

        # Generate normalized embedding for the query
        query_emb = model.encode([q], normalize_embeddings=True)
        query_vector = np.array(query_emb, dtype=np.float32)

        # Search index
        k = min(TOP_K_CHUNKS, len(chunks))
        distances, indices = index.search(query_vector, k=k)

        parts = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(chunks):
                continue
            
            c = chunks[idx]
            citation = f"### {c['source']} (Page {c['page']}, Chunk {c['chunk_id']})"
            image_note = f"\n*[Image context: {c['image_description']}]*" if c.get("has_image") and c.get("image_description") else ""
            
            parts.append(f"{citation}\n{c['text']}{image_note}")

        out = "\n\n".join(parts)
        if len(out) > MAX_RAG_CONTEXT_CHARS:
            out = out[:MAX_RAG_CONTEXT_CHARS] + "\n\n[...retrieval truncated...]"
        return out

    except Exception as e:
        return f"(Error performing semantic search: {e})"
