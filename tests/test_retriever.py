# tests/test_retriever.py
from unittest.mock import MagicMock, patch
from src.retriever import retrieve_relevant_chunks


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
         patch("src.retriever._get_embedding_model") as mock_model:
        mock_col.return_value.query.return_value = mock_results
        mock_model.return_value.encode.return_value = [[0.1] * 384]

        results = retrieve_relevant_chunks("When is the tax deadline?")

    assert len(results) == 2
    assert results[0]["text"] == "Tax filing deadline is April 15."
    assert results[0]["source_url"] == "https://www.irs.gov/page1"
    assert results[1]["text"] == "Standard deduction is $14,600."
