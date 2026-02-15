# src/retriever.py
import chromadb
from sentence_transformers import SentenceTransformer

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
COLLECTION_NAME = "irs_documents"
CHROMA_DIR = "data/chroma"

_model = None
_collection = None


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


def retrieve_relevant_chunks(query: str, top_k: int = 5) -> list[dict]:
    """
    Embed the query and retrieve the top_k most relevant chunks from ChromaDB.
    Returns a list of dicts with 'text', 'source_url', 'title'.
    """
    model = _get_embedding_model()
    embedding = model.encode([query])
    query_embedding = embedding.tolist() if hasattr(embedding, "tolist") else embedding

    collection = _get_collection()
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k,
    )

    chunks = []
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        chunks.append(
            {
                "text": doc,
                "source_url": meta.get("source_url", ""),
                "title": meta.get("title", ""),
            }
        )
    return chunks
