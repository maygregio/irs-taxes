# tests/test_retriever.py
from unittest.mock import MagicMock, patch
from src.retriever import (
    retrieve_relevant_chunks,
    _deduplicate,
    _text_overlap_ratio,
)


def test_retrieve_relevant_chunks_returns_formatted_results():
    mock_query_result = {
        "matches": [
            {
                "id": "chunk_0",
                "score": 0.85,
                "metadata": {
                    "text": "Tax filing deadline is April 15.",
                    "source_url": "https://www.irs.gov/page1",
                    "title": "Filing Deadlines",
                },
            },
            {
                "id": "chunk_1",
                "score": 0.75,
                "metadata": {
                    "text": "Standard deduction is $14,600.",
                    "source_url": "https://www.irs.gov/page2",
                    "title": "Deductions",
                },
            },
        ]
    }

    mock_index = MagicMock()
    mock_index.query.return_value = mock_query_result

    mock_voyage = MagicMock()
    mock_voyage.embed.return_value.embeddings = [[0.1] * 1024]

    with patch("src.retriever._get_pinecone_index", return_value=mock_index), \
         patch("src.retriever._get_voyage_client", return_value=mock_voyage):
        results = retrieve_relevant_chunks("When is the tax deadline?")

    assert len(results) == 2
    assert results[0]["text"] == "Tax filing deadline is April 15."
    assert results[0]["source_url"] == "https://www.irs.gov/page1"
    assert results[1]["text"] == "Standard deduction is $14,600."


def test_score_threshold_filters_irrelevant():
    mock_query_result = {
        "matches": [
            {
                "id": "chunk_0",
                "score": 0.85,
                "metadata": {
                    "text": "Relevant chunk.",
                    "source_url": "https://irs.gov/1",
                    "title": "A",
                },
            },
            {
                "id": "chunk_1",
                "score": 0.15,
                "metadata": {
                    "text": "Irrelevant chunk.",
                    "source_url": "https://irs.gov/2",
                    "title": "B",
                },
            },
        ]
    }

    mock_index = MagicMock()
    mock_index.query.return_value = mock_query_result

    mock_voyage = MagicMock()
    mock_voyage.embed.return_value.embeddings = [[0.1] * 1024]

    with patch("src.retriever._get_pinecone_index", return_value=mock_index), \
         patch("src.retriever._get_voyage_client", return_value=mock_voyage):
        results = retrieve_relevant_chunks("test query")

    assert len(results) == 1
    assert results[0]["text"] == "Relevant chunk."


def test_text_overlap_ratio():
    assert _text_overlap_ratio("the cat sat on the mat", "the cat sat on the mat") == 1.0
    assert _text_overlap_ratio("hello world", "goodbye moon") == 0.0
    assert _text_overlap_ratio("", "anything") == 0.0


def test_deduplicate_removes_overlapping():
    chunks = [
        {"text": "The tax filing deadline for individual returns is April 15 each year"},
        {"text": "The tax filing deadline for individual returns is April 15 each year and extensions"},
        {"text": "Standard deduction amounts vary by filing status"},
    ]
    result = _deduplicate(chunks)
    assert len(result) == 2
    assert result[0]["text"] == chunks[0]["text"]
    assert result[1]["text"] == chunks[2]["text"]
