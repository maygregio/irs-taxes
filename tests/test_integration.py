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

    # Query it (patch the retriever's globals to use our test ChromaDB)
    import src.retriever as retriever_mod
    import chromadb
    from sentence_transformers import SentenceTransformer

    old_collection = retriever_mod._collection
    old_model = retriever_mod._model

    try:
        client = chromadb.PersistentClient(path=str(chroma_dir))
        retriever_mod._collection = client.get_collection("irs_documents")
        retriever_mod._model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

        results = retrieve_relevant_chunks("What is the tax filing deadline?", top_k=2)
        assert len(results) == 2
        # The deadline document should be the most relevant
        assert any("April 15" in r["text"] for r in results)
    finally:
        retriever_mod._collection = old_collection
        retriever_mod._model = old_model
