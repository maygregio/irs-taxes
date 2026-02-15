# src/retriever.py
import json
import os

import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
COLLECTION_NAME = "irs_documents"
CHROMA_DIR = "data/chroma"
BM25_CORPUS_PATH = "data/bm25_corpus.json"
DISTANCE_THRESHOLD = 0.8
TOP_K_FETCH = 10

_model = None
_collection = None
_bm25 = None
_bm25_corpus = None


def _get_embedding_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _model


def _get_collection():
    global _collection
    if _collection is None:
        client = chromadb.PersistentClient(path=CHROMA_DIR)
        _collection = client.get_collection(COLLECTION_NAME)
    return _collection


def _get_bm25():
    global _bm25, _bm25_corpus
    if _bm25 is None:
        with open(BM25_CORPUS_PATH) as f:
            _bm25_corpus = json.load(f)
        tokenized = [doc["text"].lower().split() for doc in _bm25_corpus]
        _bm25 = BM25Okapi(tokenized)
    return _bm25, _bm25_corpus


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


def _vector_search(query: str, n_results: int = TOP_K_FETCH) -> list[dict]:
    """Run vector similarity search via ChromaDB. Returns chunks with distances."""
    model = _get_embedding_model()
    embedding = model.encode([query])
    query_embedding = embedding.tolist() if hasattr(embedding, "tolist") else embedding

    collection = _get_collection()
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )

    chunks = []
    for doc, meta, dist in zip(
        results["documents"][0], results["metadatas"][0], results["distances"][0]
    ):
        if dist > DISTANCE_THRESHOLD:
            continue
        chunks.append(
            {
                "text": doc,
                "source_url": meta.get("source_url", ""),
                "title": meta.get("title", ""),
            }
        )
    return chunks


def _bm25_search(query: str, n_results: int = TOP_K_FETCH) -> list[dict]:
    """Run BM25 keyword search over the corpus."""
    bm25, corpus = _get_bm25()
    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)

    # Get top n indices
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[
        :n_results
    ]

    chunks = []
    for idx in top_indices:
        if scores[idx] <= 0:
            continue
        entry = corpus[idx]
        chunks.append(
            {
                "text": entry["text"],
                "source_url": entry["source_url"],
                "title": entry["title"],
            }
        )
    return chunks


def _reciprocal_rank_fusion(
    result_lists: list[list[dict]], k: int = 60
) -> list[dict]:
    """Merge multiple ranked lists using Reciprocal Rank Fusion."""
    scores: dict[str, float] = {}
    chunk_map: dict[str, dict] = {}

    for result_list in result_lists:
        for rank, chunk in enumerate(result_list):
            key = chunk["text"][:200]  # Use first 200 chars as dedup key
            scores[key] = scores.get(key, 0) + 1.0 / (k + rank + 1)
            if key not in chunk_map:
                chunk_map[key] = chunk

    sorted_keys = sorted(scores, key=lambda x: scores[x], reverse=True)
    return [chunk_map[key] for key in sorted_keys]


def retrieve_relevant_chunks(query: str, top_k: int = 5) -> list[dict]:
    """
    Retrieve relevant chunks using hybrid search (vector + BM25),
    with relevance threshold filtering and deduplication.
    Returns a list of dicts with 'text', 'source_url', 'title'.
    """
    # Run both searches
    vector_results = _vector_search(query, n_results=TOP_K_FETCH)

    try:
        bm25_results = _bm25_search(query, n_results=TOP_K_FETCH)
    except (FileNotFoundError, json.JSONDecodeError):
        # BM25 corpus not available — fall back to vector-only
        bm25_results = []

    # Merge with RRF
    if bm25_results:
        merged = _reciprocal_rank_fusion([vector_results, bm25_results])
    else:
        merged = vector_results

    # Deduplicate overlapping chunks
    deduped = _deduplicate(merged)

    return deduped[:top_k]
