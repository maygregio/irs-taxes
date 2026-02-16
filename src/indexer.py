# src/indexer.py
import json
import os

from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb
from sentence_transformers import SentenceTransformer


def load_documents(raw_dir: str) -> list[dict]:
    """Load all JSON documents from the raw data directory."""
    docs = []
    for filename in os.listdir(raw_dir):
        if not filename.endswith(".json"):
            continue
        filepath = os.path.join(raw_dir, filename)
        with open(filepath) as f:
            doc = json.load(f)
            docs.append(doc)
    return docs


def chunk_documents(
    docs: list[dict], chunk_size: int = 1000, chunk_overlap: int = 200
) -> list[dict]:
    """Split documents into chunks with metadata."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = []
    for doc in docs:
        texts = splitter.split_text(doc["content"])
        for i, text in enumerate(texts):
            chunks.append(
                {
                    "text": text,
                    "metadata": {
                        "source_url": doc["url"],
                        "title": doc.get("title", ""),
                        "content_type": doc.get("content_type", ""),
                        "chunk_index": i,
                    },
                }
            )
    return chunks


EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
COLLECTION_NAME = "irs_documents"


def build_index(
    raw_dir: str = "data/raw",
    chroma_dir: str = "data/chroma",
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> int:
    """
    Load documents, chunk them, embed, and store in ChromaDB.
    Returns the number of chunks indexed.
    """
    print("Loading documents...")
    docs = load_documents(raw_dir)
    if not docs:
        print("No documents found.")
        return 0

    print(f"Loaded {len(docs)} documents. Chunking...")
    chunks = chunk_documents(docs, chunk_size, chunk_overlap)
    print(f"Created {len(chunks)} chunks. Embedding...")

    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    texts = [c["text"] for c in chunks]
    embeddings = model.encode(texts, show_progress_bar=True).tolist()

    client = chromadb.PersistentClient(path=chroma_dir)

    try:
        client.delete_collection(COLLECTION_NAME)
    except (ValueError, chromadb.errors.NotFoundError):
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    batch_size = 500
    for i in range(0, len(chunks), batch_size):
        batch_end = min(i + batch_size, len(chunks))
        collection.add(
            ids=[f"chunk_{j}" for j in range(i, batch_end)],
            embeddings=embeddings[i:batch_end],
            documents=texts[i:batch_end],
            metadatas=[c["metadata"] for c in chunks[i:batch_end]],
        )

    # Save BM25 corpus for hybrid search
    bm25_corpus = [
        {
            "text": c["text"],
            "source_url": c["metadata"]["source_url"],
            "title": c["metadata"]["title"],
        }
        for c in chunks
    ]
    corpus_path = os.path.join(os.path.dirname(chroma_dir), "bm25_corpus.json")
    with open(corpus_path, "w") as f:
        json.dump(bm25_corpus, f)

    print(f"Indexed {len(chunks)} chunks into ChromaDB.")
    print(f"Saved BM25 corpus ({len(bm25_corpus)} entries) to {corpus_path}")
    return len(chunks)
