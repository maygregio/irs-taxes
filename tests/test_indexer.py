# tests/test_indexer.py
import json
import os
from src.indexer import load_documents


def test_load_documents_reads_json_files(tmp_path):
    doc = {
        "url": "https://www.irs.gov/test",
        "title": "Test Page",
        "content": "This is test content about taxes.",
        "content_type": "forms",
        "last_scraped": "2026-01-01T00:00:00Z",
    }
    with open(tmp_path / "test.json", "w") as f:
        json.dump(doc, f)

    docs = load_documents(str(tmp_path))
    assert len(docs) == 1
    assert docs[0]["content"] == "This is test content about taxes."
    assert docs[0]["url"] == "https://www.irs.gov/test"


def test_load_documents_skips_non_json(tmp_path):
    (tmp_path / "readme.txt").write_text("not a doc")
    docs = load_documents(str(tmp_path))
    assert len(docs) == 0


from src.indexer import chunk_documents


def test_chunk_documents_splits_long_content():
    docs = [
        {
            "url": "https://www.irs.gov/test",
            "title": "Test",
            "content": "Word " * 500,  # ~2500 chars
            "content_type": "forms",
        }
    ]
    chunks = chunk_documents(docs, chunk_size=500, chunk_overlap=100)
    assert len(chunks) > 1
    assert all("source_url" in c["metadata"] for c in chunks)
    assert all("title" in c["metadata"] for c in chunks)


def test_chunk_documents_preserves_metadata():
    docs = [
        {
            "url": "https://www.irs.gov/form-1040",
            "title": "Form 1040",
            "content": "Short content.",
            "content_type": "forms",
        }
    ]
    chunks = chunk_documents(docs, chunk_size=500, chunk_overlap=100)
    assert len(chunks) == 1
    assert chunks[0]["metadata"]["source_url"] == "https://www.irs.gov/form-1040"
    assert chunks[0]["metadata"]["content_type"] == "forms"
