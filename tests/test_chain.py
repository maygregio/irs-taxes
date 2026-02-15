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
from src.chain import ask


def test_ask_calls_claude_and_returns_response():
    mock_chunks = [
        {"text": "Deadline is April 15.", "source_url": "https://irs.gov/d", "title": "Deadlines"}
    ]

    with patch("src.chain.retrieve_relevant_chunks", return_value=mock_chunks), \
         patch("src.chain.Anthropic"):
        sources, stream = ask("When is the deadline?", chat_history=[])

    assert len(sources) == 1
    assert sources[0]["source_url"] == "https://irs.gov/d"
    assert sources[0]["text"] == "Deadline is April 15."
    assert callable(stream)
