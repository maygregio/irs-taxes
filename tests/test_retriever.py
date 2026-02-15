# tests/test_retriever.py
from unittest.mock import MagicMock, patch
from src.retriever import (
    retrieve_relevant_chunks,
    _deduplicate,
    _text_overlap_ratio,
    _reciprocal_rank_fusion,
)


def test_retrieve_relevant_chunks_returns_formatted_results():
    mock_results = {
        "documents": [["Tax filing deadline is April 15.", "Standard deduction is $14,600."]],
        "metadatas": [
            [
                {"source_url": "https://www.irs.gov/page1", "title": "Filing Deadlines"},
                {"source_url": "https://www.irs.gov/page2", "title": "Deductions"},
            ]
        ],
        "distances": [[0.2, 0.4]],
    }

    with patch("src.retriever._get_collection") as mock_col, \
         patch("src.retriever._get_embedding_model") as mock_model, \
         patch("src.retriever._bm25_search", side_effect=FileNotFoundError):
        mock_col.return_value.query.return_value = mock_results
        mock_model.return_value.encode.return_value = [[0.1] * 384]

        results = retrieve_relevant_chunks("When is the tax deadline?")

    assert len(results) == 2
    assert results[0]["text"] == "Tax filing deadline is April 15."
    assert results[0]["source_url"] == "https://www.irs.gov/page1"
    assert results[1]["text"] == "Standard deduction is $14,600."


def test_distance_threshold_filters_irrelevant():
    mock_results = {
        "documents": [["Relevant chunk.", "Irrelevant chunk."]],
        "metadatas": [
            [
                {"source_url": "https://irs.gov/1", "title": "A"},
                {"source_url": "https://irs.gov/2", "title": "B"},
            ]
        ],
        "distances": [[0.3, 0.9]],  # second is above 0.8 threshold
    }

    with patch("src.retriever._get_collection") as mock_col, \
         patch("src.retriever._get_embedding_model") as mock_model, \
         patch("src.retriever._bm25_search", side_effect=FileNotFoundError):
        mock_col.return_value.query.return_value = mock_results
        mock_model.return_value.encode.return_value = [[0.1] * 384]

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


def test_reciprocal_rank_fusion_merges():
    list_a = [
        {"text": "chunk A", "source_url": "a", "title": "A"},
        {"text": "chunk B", "source_url": "b", "title": "B"},
    ]
    list_b = [
        {"text": "chunk B", "source_url": "b", "title": "B"},
        {"text": "chunk C", "source_url": "c", "title": "C"},
    ]
    merged = _reciprocal_rank_fusion([list_a, list_b])
    # chunk B appears in both lists, so it should rank highest
    assert merged[0]["text"] == "chunk B"
    assert len(merged) == 3
