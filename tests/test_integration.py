# tests/test_integration.py
"""
Integration test: verifies the retrieve -> ask pipeline works end-to-end.
Mocks Pinecone, Voyage AI, and Anthropic — no external calls.
"""
from unittest.mock import patch, MagicMock

from src.retriever import retrieve_relevant_chunks
from src.chain import ask


def test_retrieve_and_ask_pipeline():
    """Test the full query pipeline: embed -> search -> build prompt -> stream."""
    mock_query_result = {
        "matches": [
            {
                "id": "chunk_0",
                "score": 0.9,
                "metadata": {
                    "text": "The tax filing deadline for individual returns is April 15. "
                    "If April 15 falls on a weekend, the deadline is the next business day.",
                    "source_url": "https://www.irs.gov/filing/deadline",
                    "title": "Tax Filing Deadline",
                },
            },
            {
                "id": "chunk_1",
                "score": 0.7,
                "metadata": {
                    "text": "You can request an automatic extension to October 15 by filing Form 4868.",
                    "source_url": "https://www.irs.gov/filing/deadline",
                    "title": "Tax Filing Deadline",
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
        results = retrieve_relevant_chunks("What is the tax filing deadline?", top_k=2)

    assert len(results) >= 1
    assert any("April 15" in r["text"] for r in results)

    # Now test the full ask() pipeline
    with patch("src.retriever._get_pinecone_index", return_value=mock_index), \
         patch("src.retriever._get_voyage_client", return_value=mock_voyage), \
         patch("src.chain.classify_query", return_value={"is_form_question": False, "forms": [], "query_type": "scenario"}), \
         patch("src.chain.Anthropic"):
        sources, stream = ask("What is the tax filing deadline?", chat_history=[])

    assert len(sources) >= 1
    assert any("April 15" in s["text"] for s in sources)
    assert callable(stream)
