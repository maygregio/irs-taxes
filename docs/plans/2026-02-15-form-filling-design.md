# Form-Filling Query Handling — Design

## Problem

When users ask for help filling out a specific IRS form, the system treats it like any other Q&A question. This produces generic answers that lack the structured, line-by-line or scenario-based guidance users need.

## Solution

Add LLM-based query classification before retrieval. When a form-filling question is detected, boost retrieval and swap to a form-specific system prompt with proactive follow-up behavior.

## Query Classifier

A `classify_query()` function in `chain.py` makes a fast Claude Haiku call to determine:

1. **is_form_question** (boolean) — Is this about filling out a form?
2. **forms** (list of strings) — Which form(s) are relevant (e.g., "Schedule C", "Form 1040")
3. **query_type** ("line_specific" | "scenario") — Is the user asking about specific lines, or describing a broad situation?

Returns JSON: `{"is_form_question": true, "forms": ["Schedule C"], "query_type": "scenario"}`

Runs before retrieval so output shapes both search and prompt.

## Retrieval Boost

When `is_form_question` is true:

- Increase `top_k` from 5 to 10
- Prepend detected form name(s) to the search query to improve chunk relevance for both vector and BM25 search

No changes to `retriever.py` — just different arguments passed from `chain.py`.

## Form-Filling System Prompt

When `is_form_question` is true, use `FORM_FILLING_PROMPT_TEMPLATE` instead of the general prompt:

- **line_specific query_type**: Line-by-line walkthrough format
- **scenario query_type**: Which sections/lines apply, then walk through relevant ones
- **Proactive follow-ups**: Ask 1-2 clarifying questions when the user's scenario is ambiguous (filing status, income type, dependents, etc.)
- Same citation rules and safety guardrails as existing prompt

When `is_form_question` is false, existing prompt and flow are unchanged.

## Data Flow

```
User question
    |
    v
classify_query(question)  <- Haiku call
    |
    +-- is_form_question=false -> existing flow unchanged
    |
    +-- is_form_question=true
          |
          v
        Augment query with form names
        retrieve_relevant_chunks(augmented_query, top_k=10)
          |
          v
        build_prompt() with FORM_FILLING_PROMPT_TEMPLATE + query_type
          |
          v
        Stream response from Claude Sonnet
```

## Files Changed

- **src/chain.py**: Add `classify_query()`, `FORM_FILLING_PROMPT_TEMPLATE`, update `ask()` to route based on classification
- **tests/test_chain.py**: Add tests for classification and prompt selection

No changes to retriever.py, indexer.py, scraper.py, or app.py.
