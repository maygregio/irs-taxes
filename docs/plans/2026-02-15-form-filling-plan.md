# Form-Filling Query Handling — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Detect form-filling questions via LLM classification, then boost retrieval and use a specialized prompt to give structured form guidance with proactive follow-ups.

**Architecture:** A Haiku classifier call runs before retrieval. Its output determines whether to augment the search query with form names, increase top_k, and swap to a form-filling system prompt. Non-form questions flow through unchanged.

**Tech Stack:** Python, Anthropic SDK (Haiku for classification, Sonnet for answers), pytest

---

### Task 1: Add `classify_query()` with tests

**Files:**
- Modify: `src/chain.py` (add function after imports)
- Test: `tests/test_chain.py`

**Step 1: Write the failing test**

Add to `tests/test_chain.py`:

```python
from unittest.mock import patch, MagicMock
from src.chain import classify_query


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
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_chain.py::test_classify_query_detects_form_question tests/test_chain.py::test_classify_query_detects_non_form_question tests/test_chain.py::test_classify_query_handles_malformed_response -v`
Expected: FAIL with `ImportError: cannot import name 'classify_query'`

**Step 3: Write minimal implementation**

Add to `src/chain.py` after the existing imports:

```python
import json


CLASSIFIER_PROMPT = """Classify this user question about IRS taxes. Return ONLY a JSON object with these fields:
- "is_form_question": true if the user is asking about how to fill out, complete, or report something on a specific IRS form or schedule. false otherwise.
- "forms": list of form names mentioned or implied (e.g. ["Form 1040", "Schedule C"]). Empty list if none.
- "query_type": "line_specific" if asking about specific lines/fields/boxes on a form. "scenario" if describing a situation and wanting guidance on which parts of the form apply.

User question: {question}"""


def classify_query(question: str) -> dict:
    """Use Claude Haiku to classify whether a question is about filling out a form."""
    default = {"is_form_question": False, "forms": [], "query_type": "scenario"}
    try:
        client = Anthropic()
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=150,
            messages=[{"role": "user", "content": CLASSIFIER_PROMPT.format(question=question)}],
        )
        raw = response.content[0].text.strip()
        parsed = json.loads(raw)
        return {
            "is_form_question": bool(parsed.get("is_form_question", False)),
            "forms": list(parsed.get("forms", [])),
            "query_type": parsed.get("query_type", "scenario"),
        }
    except (json.JSONDecodeError, KeyError, IndexError):
        return default
    except Exception:
        return default
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_chain.py::test_classify_query_detects_form_question tests/test_chain.py::test_classify_query_detects_non_form_question tests/test_chain.py::test_classify_query_handles_malformed_response -v`
Expected: 3 PASSED

**Step 5: Commit**

```bash
git add src/chain.py tests/test_chain.py
git commit -m "feat: add LLM-based query classifier for form-filling detection"
```

---

### Task 2: Add form-filling system prompt

**Files:**
- Modify: `src/chain.py` (add new prompt template)
- Test: `tests/test_chain.py`

**Step 1: Write the failing test**

Add to `tests/test_chain.py`:

```python
from src.chain import build_prompt, FORM_FILLING_PROMPT_TEMPLATE


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
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_chain.py::test_build_prompt_uses_form_template_for_line_specific tests/test_chain.py::test_build_prompt_uses_form_template_for_scenario tests/test_chain.py::test_build_prompt_uses_default_template_when_no_query_type -v`
Expected: FAIL (build_prompt doesn't accept query_type yet, FORM_FILLING_PROMPT_TEMPLATE doesn't exist)

**Step 3: Write minimal implementation**

Add `FORM_FILLING_PROMPT_TEMPLATE` to `src/chain.py`:

```python
FORM_FILLING_PROMPT_TEMPLATE = """You are an IRS form-filling assistant that helps users complete IRS forms using only the official IRS source documents provided below.

Goal:
Help the user fill out their IRS form correctly by providing {guidance_style}.

Core rules:
- Use only the provided sources. Do not add outside facts.
- Never guess at values, thresholds, or line numbers not in the sources.
- If the question could vary by filing status, tax year, income level, or circumstance, call that out clearly.

Proactive clarification:
- Before answering, check if the user's situation is ambiguous (e.g., filing status unknown, income type unclear, dependents not mentioned).
- If ambiguous, ask 1-2 specific clarifying questions before providing form guidance. For example: "Before I walk you through this, are you filing as single or married filing jointly?" or "Is this income from self-employment or a side job with a W-2?"
- If the user has already provided enough context (or answered previous clarifying questions in the chat), proceed directly with the guidance.

Citation rules:
- Cite factual statements inline as [Source: URL].
- Use only URLs from the provided sources.

{response_structure}

IRS Source Documents:
{sources}"""

FORM_LINE_SPECIFIC_STRUCTURE = """Response structure:
1) Confirm which form and line(s) you're addressing
2) Line-by-line walkthrough: For each relevant line, explain what value to enter, where to find it, and any conditions
3) Common mistakes or exceptions for these lines
4) Source links used"""

FORM_SCENARIO_STRUCTURE = """Response structure:
1) Ask clarifying questions if the scenario is ambiguous (then wait for the user's response)
2) Once you have enough context: identify which form sections/lines apply to this scenario
3) Walk through the relevant lines in order, explaining what to enter and why
4) Flag anything that depends on details the user hasn't shared
5) Source links used"""
```

Update `build_prompt` signature to accept optional `query_type`:

```python
def build_prompt(
    question: str,
    chunks: list[dict],
    chat_history: list[dict],
    query_type: str | None = None,
) -> list[dict]:
    """Build the message list for the Claude API call."""
    sources_text = ""
    for i, chunk in enumerate(chunks, 1):
        sources_text += f"### Source {i}: {chunk['title']}\nURL: {chunk['source_url']}\n\n{chunk['text']}\n\n---\n\n"

    if query_type == "line_specific":
        system_content = FORM_FILLING_PROMPT_TEMPLATE.format(
            sources=sources_text,
            guidance_style="a line-by-line walkthrough of the specific lines or fields they are asking about",
            response_structure=FORM_LINE_SPECIFIC_STRUCTURE,
        )
    elif query_type == "scenario":
        system_content = FORM_FILLING_PROMPT_TEMPLATE.format(
            sources=sources_text,
            guidance_style="scenario-based guidance explaining which parts of the form apply to their situation",
            response_structure=FORM_SCENARIO_STRUCTURE,
        )
    else:
        system_content = SYSTEM_PROMPT_TEMPLATE.format(sources=sources_text)

    messages = [{"role": "system", "content": system_content}]

    for msg in chat_history[-10:]:
        messages.append({"role": msg["role"], "content": msg["content"]})

    messages.append({"role": "user", "content": question})

    return messages
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_chain.py -v`
Expected: All tests PASS (including existing ones — `build_prompt` without `query_type` still uses the default template)

**Step 5: Commit**

```bash
git add src/chain.py tests/test_chain.py
git commit -m "feat: add form-filling system prompt with line-specific and scenario modes"
```

---

### Task 3: Wire classification into `ask()` with retrieval boost

**Files:**
- Modify: `src/chain.py` (update `ask()`)
- Test: `tests/test_chain.py`

**Step 1: Write the failing test**

Add to `tests/test_chain.py`:

```python
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
    assert call_args[1].get("top_k", 5) == 5
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_chain.py::test_ask_uses_classifier_and_boosts_retrieval_for_form_questions tests/test_chain.py::test_ask_uses_default_flow_for_non_form_questions -v`
Expected: FAIL (ask() doesn't call classify_query yet)

**Step 3: Write minimal implementation**

Replace the `ask()` function in `src/chain.py`:

```python
def ask(question: str, chat_history: list[dict]) -> tuple[list[dict], callable]:
    """
    Answer a question using RAG: classify, retrieve relevant IRS chunks, send to Claude.
    For form-filling questions, boosts retrieval and uses a specialized prompt.
    Returns (source_chunks, stream_generator_function).
    """
    classification = classify_query(question)

    if classification["is_form_question"]:
        # Augment query with form names for better retrieval
        form_prefix = " ".join(classification["forms"])
        augmented_query = f"{form_prefix} {question}"
        chunks = retrieve_relevant_chunks(augmented_query, top_k=10)
        query_type = classification["query_type"]
    else:
        chunks = retrieve_relevant_chunks(question, top_k=5)
        query_type = None

    messages = build_prompt(question, chunks, chat_history, query_type=query_type)

    system_msg = messages[0]["content"]
    conversation = messages[1:]

    client = Anthropic()

    def stream():
        with client.messages.stream(
            model="claude-sonnet-4-5-20250929",
            max_tokens=4096,
            system=system_msg,
            messages=conversation,
        ) as response:
            for text in response.text_stream:
                yield text

    return chunks, stream
```

**Step 4: Run all tests to verify they pass**

Run: `pytest tests/test_chain.py -v`
Expected: All tests PASS

**Step 5: Commit**

```bash
git add src/chain.py tests/test_chain.py
git commit -m "feat: wire query classifier into ask() with retrieval boost for form questions"
```

---

### Task 4: Run full test suite and verify no regressions

**Files:**
- No new files

**Step 1: Run full test suite**

Run: `pytest tests/ -v`
Expected: All tests PASS

**Step 2: Commit (if any fixups needed)**

Only if fixes were required in previous steps.
