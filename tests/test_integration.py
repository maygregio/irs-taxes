# tests/test_integration.py
"""
Integration test: indexes sample documents and queries them.
Does NOT hit IRS.gov or Claude API — uses mocks for external services.
"""
import json
import os
from unittest.mock import patch, MagicMock

from src.indexer import load_documents, chunk_documents, build_index
from src.retriever import retrieve_relevant_chunks


def test_index_and_retrieve(tmp_path):
    """Test the full pipeline: load docs -> chunk -> index -> retrieve."""
    # Create sample documents
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    chroma_dir = tmp_path / "chroma"

    doc1 = {
        "url": "https://www.irs.gov/filing/deadline",
        "title": "Tax Filing Deadline",
        "content": "The tax filing deadline for individual returns is April 15. "
        "If April 15 falls on a weekend or holiday, the deadline is the next business day. "
        "You can request an automatic extension to October 15 by filing Form 4868.",
        "content_type": "forms",
    }
    doc2 = {
        "url": "https://www.irs.gov/deductions/standard",
        "title": "Standard Deduction",
        "content": "The standard deduction for single filers is $14,600 for 2024. "
        "For married filing jointly, the standard deduction is $29,200. "
        "Taxpayers who are 65 or older get an additional standard deduction.",
        "content_type": "publications",
    }

    for i, doc in enumerate([doc1, doc2]):
        with open(raw_dir / f"doc{i}.json", "w") as f:
            json.dump(doc, f)

    # Build the index
    count = build_index(
        raw_dir=str(raw_dir),
        chroma_dir=str(chroma_dir),
        chunk_size=500,
        chunk_overlap=50,
    )
    assert count > 0

    # Verify BM25 corpus was created
    bm25_path = os.path.join(str(tmp_path), "bm25_corpus.json")
    assert os.path.exists(bm25_path)
    with open(bm25_path) as f:
        corpus = json.load(f)
    assert len(corpus) == count

    # Query it (patch the retriever's globals to use our test ChromaDB and BM25)
    import src.retriever as retriever_mod
    import chromadb
    from sentence_transformers import SentenceTransformer

    old_collection = retriever_mod._collection
    old_model = retriever_mod._model
    old_bm25 = retriever_mod._bm25
    old_bm25_corpus = retriever_mod._bm25_corpus

    try:
        client = chromadb.PersistentClient(path=str(chroma_dir))
        retriever_mod._collection = client.get_collection("irs_documents")
        retriever_mod._model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        # Reset BM25 so it reloads from the test corpus
        retriever_mod._bm25 = None
        retriever_mod._bm25_corpus = None

        with patch.object(retriever_mod, "BM25_CORPUS_PATH", bm25_path):
            results = retrieve_relevant_chunks("What is the tax filing deadline?", top_k=2)
            assert len(results) >= 1
            # The deadline document should be the most relevant
            assert any("April 15" in r["text"] for r in results)
    finally:
        retriever_mod._collection = old_collection
        retriever_mod._model = old_model
        retriever_mod._bm25 = old_bm25
        retriever_mod._bm25_corpus = old_bm25_corpus
