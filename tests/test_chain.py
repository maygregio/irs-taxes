# tests/test_chain.py
from src.chain import build_prompt


def test_build_prompt_includes_sources_and_question():
    chunks = [
        {"text": "The deadline is April 15.", "source_url": "https://www.irs.gov/p1", "title": "Deadlines"},
        {"text": "Extensions available.", "source_url": "https://www.irs.gov/p2", "title": "Extensions"},
    ]
    messages = build_prompt("When is the deadline?", chunks, chat_history=[])

    assert any("IRS" in m["content"] for m in messages if m["role"] == "system")
    system_content = next(m["content"] for m in messages if m["role"] == "system")
    assert "April 15" in system_content
    assert "https://www.irs.gov/p1" in system_content
    assert messages[-1]["role"] == "user"
    assert messages[-1]["content"] == "When is the deadline?"


def test_build_prompt_includes_chat_history():
    chunks = [{"text": "Info.", "source_url": "https://irs.gov/x", "title": "T"}]
    history = [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello!"},
    ]
    messages = build_prompt("Follow up?", chunks, chat_history=history)
    roles = [m["role"] for m in messages]
    assert roles == ["system", "user", "assistant", "user"]


from unittest.mock import patch, MagicMock
from src.chain import ask, classify_query


def test_ask_calls_claude_and_returns_response():
    mock_chunks = [
        {"text": "Deadline is April 15.", "source_url": "https://irs.gov/d", "title": "Deadlines"}
    ]

    with patch("src.chain.classify_query", return_value={"is_form_question": False, "forms": [], "query_type": "scenario"}), \
         patch("src.chain.retrieve_relevant_chunks", return_value=mock_chunks), \
         patch("src.chain.Anthropic"):
        sources, stream = ask("When is the deadline?", chat_history=[])

    assert len(sources) == 1
    assert sources[0]["source_url"] == "https://irs.gov/d"
    assert sources[0]["text"] == "Deadline is April 15."
    assert callable(stream)


def test_classify_query_detects_form_question():
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text='{"is_form_question": true, "forms": ["Schedule C"], "query_type": "scenario"}')]

    mock_client = MagicMock()
    mock_client.messages.create.return_value = mock_response

    with patch("src.chain.Anthropic", return_value=mock_client):
        result = classify_query("How do I fill out Schedule C as a freelancer?")

    assert result["is_form_question"] is True
    assert "Schedule C" in result["forms"]
    assert result["query_type"] in ("line_specific", "scenario")


def test_classify_query_detects_non_form_question():
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text='{"is_form_question": false, "forms": [], "query_type": "scenario"}')]

    mock_client = MagicMock()
    mock_client.messages.create.return_value = mock_response

    with patch("src.chain.Anthropic", return_value=mock_client):
        result = classify_query("When is the tax filing deadline?")

    assert result["is_form_question"] is False
    assert result["forms"] == []


def test_classify_query_handles_malformed_response():
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text="not valid json")]

    mock_client = MagicMock()
    mock_client.messages.create.return_value = mock_response

    with patch("src.chain.Anthropic", return_value=mock_client):
        result = classify_query("How do I fill out Form 1040?")

    assert result["is_form_question"] is False
    assert result["forms"] == []
    assert result["query_type"] == "scenario"


def test_ask_uses_classifier_and_boosts_retrieval_for_form_questions():
    mock_chunks = [
        {"text": "Schedule C instructions.", "source_url": "https://irs.gov/sc", "title": "Schedule C"}
    ]
    classification = {"is_form_question": True, "forms": ["Schedule C"], "query_type": "scenario"}

    with patch("src.chain.classify_query", return_value=classification), \
         patch("src.chain.retrieve_relevant_chunks", return_value=mock_chunks) as mock_retrieve, \
         patch("src.chain.Anthropic"):
        sources, stream = ask("How do I fill out Schedule C?", chat_history=[])

    # Verify retrieval was called with augmented query and boosted top_k
    call_args = mock_retrieve.call_args
    assert "Schedule C" in call_args[0][0]
    assert call_args[1]["top_k"] == 10


def test_ask_uses_default_flow_for_non_form_questions():
    mock_chunks = [
        {"text": "Deadline info.", "source_url": "https://irs.gov/d", "title": "Deadlines"}
    ]
    classification = {"is_form_question": False, "forms": [], "query_type": "scenario"}

    with patch("src.chain.classify_query", return_value=classification), \
         patch("src.chain.retrieve_relevant_chunks", return_value=mock_chunks) as mock_retrieve, \
         patch("src.chain.Anthropic"):
        sources, stream = ask("When is the deadline?", chat_history=[])

    # Verify retrieval was called with original query and default top_k
    call_args = mock_retrieve.call_args
    assert call_args[0][0] == "When is the deadline?"
    assert call_args[1]["top_k"] == 10


def test_build_prompt_uses_form_template_for_line_specific():
    chunks = [
        {"text": "Schedule C line 1 instructions.", "source_url": "https://irs.gov/sc", "title": "Schedule C"},
    ]
    messages = build_prompt("What goes on line 12?", chunks, chat_history=[], query_type="line_specific")
    system_content = next(m["content"] for m in messages if m["role"] == "system")
    assert "line-by-line" in system_content.lower() or "Line" in system_content


def test_build_prompt_uses_form_template_for_scenario():
    chunks = [
        {"text": "Schedule C instructions.", "source_url": "https://irs.gov/sc", "title": "Schedule C"},
    ]
    messages = build_prompt("How do I report freelance income?", chunks, chat_history=[], query_type="scenario")
    system_content = next(m["content"] for m in messages if m["role"] == "system")
    assert "clarifying question" in system_content.lower() or "follow-up" in system_content.lower()


def test_build_prompt_uses_default_template_when_no_query_type():
    chunks = [
        {"text": "Deadline info.", "source_url": "https://irs.gov/d", "title": "Deadlines"},
    ]
    messages = build_prompt("When is the deadline?", chunks, chat_history=[])
    system_content = next(m["content"] for m in messages if m["role"] == "system")
    assert "IRS assistant" in system_content
