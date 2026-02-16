import json
from unittest.mock import MagicMock, patch
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
    mock_index.delete.assert_called_with(filter={"source_url": {"$eq": "https://www.irs.gov/test"}})


def test_build_index_skips_when_no_changes(tmp_path):
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

    manifest_path = tmp_path / "index_manifest.json"

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

        first_count = build_index(raw_dir=str(raw_dir), manifest_path=str(manifest_path))
        mock_index.reset_mock()
        mock_voyage.embed.reset_mock()

        second_count = build_index(raw_dir=str(raw_dir), manifest_path=str(manifest_path))

    assert first_count > 0
    assert second_count == 0
    mock_index.upsert.assert_not_called()
    mock_voyage.embed.assert_not_called()


def test_build_index_full_reindex_deletes_all(tmp_path):
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
        count = build_index(raw_dir=str(raw_dir), full_reindex=True)

    assert count > 0
    mock_index.delete.assert_any_call(delete_all=True)
    mock_index.upsert.assert_called()


def test_build_index_bootstrap_uses_scraped_urls_only(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()

    doc1 = {
        "url": "https://www.irs.gov/new-doc",
        "title": "New Doc",
        "content": "New tax content.",
        "content_type": "forms",
    }
    doc2 = {
        "url": "https://www.irs.gov/old-doc",
        "title": "Old Doc",
        "content": "Old tax content.",
        "content_type": "forms",
    }
    with open(raw_dir / "doc1.json", "w") as f:
        json.dump(doc1, f)
    with open(raw_dir / "doc2.json", "w") as f:
        json.dump(doc2, f)

    manifest_path = tmp_path / "index_manifest.json"

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
        count = build_index(
            raw_dir=str(raw_dir),
            manifest_path=str(manifest_path),
            changed_urls=[doc1["url"]],
        )

    assert count > 0
    mock_index.delete.assert_called_with(filter={"source_url": {"$eq": doc1["url"]}})
    deleted_urls = []
    for call in mock_index.delete.call_args_list:
        if "filter" in call.kwargs:
            deleted_urls.append(call.kwargs["filter"]["source_url"]["$eq"])
    assert doc2["url"] not in deleted_urls

    all_upserted_urls = []
    for call in mock_index.upsert.call_args_list:
        vectors = call.kwargs["vectors"]
        all_upserted_urls.extend(v["metadata"]["source_url"] for v in vectors)
    assert all_upserted_urls
    assert set(all_upserted_urls) == {doc1["url"]}
