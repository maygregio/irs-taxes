# tests/test_indexer.py
import json
from unittest.mock import MagicMock, patch, call
from src.indexer import load_documents, chunk_documents


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


def test_chunk_documents_splits_long_content():
    docs = [
        {
            "url": "https://www.irs.gov/test",
            "title": "Test",
            "content": "Word " * 500,
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


def test_build_index_upserts_to_pinecone(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    doc = {
        "url": "https://www.irs.gov/test",
        "title": "Test",
        "content": "Some tax content for testing.",
        "content_type": "forms",
    }
    with open(raw_dir / "doc.json", "w") as f:
        json.dump(doc, f)

    mock_index = MagicMock()
    mock_pc = MagicMock()
    mock_pc.list_indexes.return_value.names.return_value = ["irs-documents"]
    mock_pc.Index.return_value = mock_index

    mock_voyage = MagicMock()
    mock_voyage.embed.return_value.embeddings = [[0.1] * 1024]

    with patch("src.indexer.Pinecone", return_value=mock_pc), \
         patch("src.indexer.voyageai") as mock_voyageai_mod:
        mock_voyageai_mod.Client.return_value = mock_voyage
        from src.indexer import build_index
        count = build_index(raw_dir=str(raw_dir))

    assert count > 0
    mock_index.upsert.assert_called()
