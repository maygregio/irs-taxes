# src/indexer.py
import json
import os
import time

import voyageai
from pinecone import Pinecone, ServerlessSpec
from langchain_text_splitters import RecursiveCharacterTextSplitter

VOYAGE_MODEL = "voyage-4-lite"
VOYAGE_DIMENSION = 1024
PINECONE_INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME", "irs-documents")
PINECONE_CLOUD = "aws"
PINECONE_REGION = "us-east-1"


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


def build_index(
    raw_dir: str = "data/raw",
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> int:
    """
    Load documents, chunk them, embed via Voyage AI, and upsert to Pinecone.
    Returns the number of chunks indexed.
    """
    print("Loading documents...")
    docs = load_documents(raw_dir)
    if not docs:
        print("No documents found.")
        return 0

    print(f"Loaded {len(docs)} documents. Chunking...")
    chunks = chunk_documents(docs, chunk_size, chunk_overlap)
    print(f"Created {len(chunks)} chunks. Embedding via Voyage AI...")

    # Initialize clients
    voyage_client = voyageai.Client(
        api_key=os.environ.get("VOYAGE_API_KEY", ""),
    )
    pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY", ""))

    # Create index if it doesn't exist
    existing_indexes = pc.list_indexes().names()
    if PINECONE_INDEX_NAME not in existing_indexes:
        print(f"Creating Pinecone index '{PINECONE_INDEX_NAME}'...")
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=VOYAGE_DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION),
        )

    index = pc.Index(PINECONE_INDEX_NAME)

    # Clear existing vectors (may fail on a fresh index)
    try:
        index.delete(delete_all=True)
    except Exception:
        pass

    # Embed and upsert in batches
    batch_size = 96  # Voyage AI batch limit
    texts = [c["text"] for c in chunks]

    for i in range(0, len(chunks), batch_size):
        batch_end = min(i + batch_size, len(chunks))
        batch_texts = texts[i:batch_end]

        batch_num = i // batch_size + 1
        total_batches = (len(chunks) + batch_size - 1) // batch_size
        print(f"  Embedding batch {batch_num}/{total_batches} ({i}–{batch_end})...")

        for attempt in range(5):
            try:
                result = voyage_client.embed(
                    batch_texts, model=VOYAGE_MODEL, input_type="document"
                )
                break
            except voyageai.error.RateLimitError:
                wait = 21 * (attempt + 1)
                print(f"    Rate limited. Waiting {wait}s (attempt {attempt + 1}/5)...")
                time.sleep(wait)
        else:
            raise RuntimeError(f"Failed to embed batch {batch_num} after 5 retries")
        embeddings = result.embeddings

        vectors = []
        for j, (emb, chunk) in enumerate(
            zip(embeddings, chunks[i:batch_end])
        ):
            vectors.append(
                {
                    "id": f"chunk_{i + j}",
                    "values": emb,
                    "metadata": {
                        "text": chunk["text"],
                        "source_url": chunk["metadata"]["source_url"],
                        "title": chunk["metadata"]["title"],
                        "content_type": chunk["metadata"]["content_type"],
                        "chunk_index": chunk["metadata"]["chunk_index"],
                    },
                }
            )
        index.upsert(vectors=vectors)

    print(f"Indexed {len(chunks)} chunks into Pinecone.")
    return len(chunks)
