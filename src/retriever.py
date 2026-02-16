# src/retriever.py
import os

import voyageai
from pinecone import Pinecone

VOYAGE_MODEL = "voyage-3"
SCORE_THRESHOLD = 0.3
TOP_K_FETCH = 10

_voyage_client = None
_pinecone_index = None


def _get_voyage_client():
    global _voyage_client
    if _voyage_client is None:
        _voyage_client = voyageai.Client(
            api_key=os.environ.get("VOYAGE_API_KEY", ""),
        )
    return _voyage_client


def _get_pinecone_index():
    global _pinecone_index
    if _pinecone_index is None:
        pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY", ""))
        index_name = os.environ.get("PINECONE_INDEX_NAME", "irs-documents")
        _pinecone_index = pc.Index(index_name)
    return _pinecone_index


def _text_overlap_ratio(text_a: str, text_b: str) -> float:
    """Return the fraction of text_a's words that appear in text_b."""
    words_a = set(text_a.lower().split())
    words_b = set(text_b.lower().split())
    if not words_a:
        return 0.0
    return len(words_a & words_b) / len(words_a)


def _deduplicate(chunks: list[dict]) -> list[dict]:
    """Remove chunks where >50% of words overlap with an already-selected chunk."""
    selected = []
    for chunk in chunks:
        is_dup = False
        for kept in selected:
            if _text_overlap_ratio(chunk["text"], kept["text"]) > 0.5:
                is_dup = True
                break
        if not is_dup:
            selected.append(chunk)
    return selected


def retrieve_relevant_chunks(query: str, top_k: int = 5) -> list[dict]:
    """
    Retrieve relevant chunks using Pinecone vector search.
    Embeds the query via Voyage AI, queries Pinecone, filters by score,
    deduplicates, and returns top_k results.
    Returns a list of dicts with 'text', 'source_url', 'title'.
    """
    client = _get_voyage_client()
    result = client.embed([query], model=VOYAGE_MODEL, input_type="query")
    query_embedding = result.embeddings[0]

    index = _get_pinecone_index()
    query_result = index.query(
        vector=query_embedding,
        top_k=TOP_K_FETCH,
        include_metadata=True,
    )

    chunks = []
    for match in query_result["matches"]:
        if match["score"] < SCORE_THRESHOLD:
            continue
        meta = match["metadata"]
        chunks.append(
            {
                "text": meta.get("text", ""),
                "source_url": meta.get("source_url", ""),
                "title": meta.get("title", ""),
            }
        )

    deduped = _deduplicate(chunks)
    return deduped[:top_k]
