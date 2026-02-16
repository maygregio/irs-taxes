# src/indexer.py
import json
import os
import time
import hashlib

import voyageai
from pinecone import Pinecone, ServerlessSpec
from langchain_text_splitters import RecursiveCharacterTextSplitter

VOYAGE_MODEL = "voyage-4-lite"
VOYAGE_DIMENSION = 1024
PINECONE_INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME", "irs-documents")
PINECONE_CLOUD = "aws"
PINECONE_REGION = "us-east-1"
MANIFEST_VERSION = 1


def load_documents(raw_dir: str) -> list[dict]:
    """Load all JSON documents from the raw data directory."""
    docs = []
    for filename in sorted(os.listdir(raw_dir)):
        if not filename.endswith(".json"):
            continue
        filepath = os.path.join(raw_dir, filename)
        with open(filepath) as f:
            doc = json.load(f)
            doc["_raw_file"] = filepath
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


def _default_manifest_path(raw_dir: str) -> str:
    return os.path.join(os.path.dirname(raw_dir), "index_manifest.json")


def _hash_document(doc: dict) -> str:
    payload = {
        "url": doc.get("url", ""),
        "title": doc.get("title", ""),
        "content_type": doc.get("content_type", ""),
        "content": doc.get("content", ""),
    }
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _build_current_manifest_entries(docs: list[dict]) -> dict[str, dict]:
    entries = {}
    for doc in docs:
        source_url = doc.get("url", "")
        if not source_url:
            continue
        entries[source_url] = {
            "doc_hash": _hash_document(doc),
            "raw_file": doc.get("_raw_file", ""),
        }
    return entries


def _load_manifest(manifest_path: str) -> dict:
    if not os.path.exists(manifest_path):
        return {"version": MANIFEST_VERSION, "entries": {}}
    with open(manifest_path) as f:
        data = json.load(f)
    entries = data.get("entries", {})
    if not isinstance(entries, dict):
        entries = {}
    return {"version": data.get("version", MANIFEST_VERSION), "entries": entries}


def _save_manifest(manifest_path: str, entries: dict[str, dict]) -> None:
    manifest_dir = os.path.dirname(manifest_path)
    if manifest_dir:
        os.makedirs(manifest_dir, exist_ok=True)
    payload = {"version": MANIFEST_VERSION, "entries": entries}
    with open(manifest_path, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def _vector_id(source_url: str, chunk_index: int, text: str) -> str:
    url_hash = hashlib.sha1(source_url.encode("utf-8")).hexdigest()[:16]
    text_hash = hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]
    return f"{url_hash}_{chunk_index}_{text_hash}"


def build_index(
    raw_dir: str = "data/raw",
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    full_reindex: bool = False,
    manifest_path: str | None = None,
    changed_urls: list[str] | None = None,
) -> int:
    """
    Load documents, chunk them, embed via Voyage AI, and upsert to Pinecone.
    By default, only new/changed documents are indexed.
    Set full_reindex=True to clear and rebuild all vectors.
    Returns the number of chunks indexed in this run.
    """
    print("Loading documents...")
    docs = load_documents(raw_dir)
    if not docs:
        print("No documents found.")
        return 0

    print(f"Loaded {len(docs)} documents.")

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

    manifest_path = manifest_path or _default_manifest_path(raw_dir)
    manifest_exists = os.path.exists(manifest_path)
    current_entries = _build_current_manifest_entries(docs)
    previous_entries = _load_manifest(manifest_path)["entries"]

    if full_reindex:
        print("Running in full reindex mode (delete_all=True).")
        try:
            index.delete(delete_all=True)
        except Exception:
            pass
        docs_to_index = docs
        deleted_urls = []
    else:
        manifest_changed_urls = [
            url
            for url, current in current_entries.items()
            if previous_entries.get(url, {}).get("doc_hash") != current.get("doc_hash")
        ]
        deleted_urls = [url for url in previous_entries.keys() if url not in current_entries]

        if changed_urls is not None:
            scraped_changed_urls = [url for url in changed_urls if url in current_entries]
        else:
            scraped_changed_urls = None

        if not manifest_exists and scraped_changed_urls is not None:
            changed_urls = sorted(set(scraped_changed_urls))
            print(
                "Manifest missing; bootstrap incremental mode from scraped URLs "
                f"({len(changed_urls)} docs)."
            )
        elif scraped_changed_urls is not None:
            changed_urls = sorted(set(manifest_changed_urls).union(scraped_changed_urls))
        else:
            changed_urls = manifest_changed_urls

        if deleted_urls:
            print(f"Removing vectors for {len(deleted_urls)} deleted documents...")
            for url in deleted_urls:
                index.delete(filter={"source_url": {"$eq": url}})

        if changed_urls:
            print(f"Refreshing vectors for {len(changed_urls)} new/updated documents...")
            for url in changed_urls:
                index.delete(filter={"source_url": {"$eq": url}})

        if not changed_urls and not deleted_urls:
            if not manifest_exists:
                _save_manifest(manifest_path, current_entries)
            print("No document changes detected. Skipping embedding/upsert.")
            return 0

        changed_set = set(changed_urls)
        docs_to_index = [doc for doc in docs if doc.get("url", "") in changed_set]

    # Embed and upsert in batches
    batch_size = 96  # Voyage AI batch limit
    chunks = chunk_documents(docs_to_index, chunk_size, chunk_overlap)
    if not chunks:
        _save_manifest(manifest_path, current_entries)
        print("No changed document chunks to index.")
        return 0

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
            source_url = chunk["metadata"]["source_url"]
            chunk_index = chunk["metadata"]["chunk_index"]
            text = chunk["text"]
            vectors.append(
                {
                    "id": _vector_id(source_url, chunk_index, text),
                    "values": emb,
                    "metadata": {
                        "text": text,
                        "source_url": source_url,
                        "title": chunk["metadata"]["title"],
                        "content_type": chunk["metadata"]["content_type"],
                        "chunk_index": chunk_index,
                    },
                }
            )
        index.upsert(vectors=vectors)

    _save_manifest(manifest_path, current_entries)
    print(f"Indexed {len(chunks)} chunks into Pinecone.")
    return len(chunks)
