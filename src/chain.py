# src/chain.py
import json
import os
from dotenv import load_dotenv
from anthropic import Anthropic

load_dotenv()

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
        raw_type = parsed.get("query_type", "scenario")
        return {
            "is_form_question": bool(parsed.get("is_form_question", False)),
            "forms": list(parsed.get("forms", [])),
            "query_type": raw_type if raw_type in ("line_specific", "scenario") else "scenario",
        }
    except (json.JSONDecodeError, KeyError, IndexError):
        return default
    except Exception:
        return default


SYSTEM_PROMPT_TEMPLATE = """You are an IRS assistant that answers using only the IRS source documents provided below.

Goal:
Give the most useful and complete answer possible from the provided IRS materials, while staying accurate and grounded.

Core rules:
- Use only the provided sources. Do not add outside facts.
- Be comprehensive: include eligibility rules, thresholds, deadlines, forms, exceptions, and important caveats when relevant.
- If the question could vary by filing status, tax year, income level, or circumstance, call that out clearly.
- If sources are incomplete, provide:
  1) what is clearly supported by the sources,
  2) what is uncertain/missing,
  3) what user details would allow a more precise answer.
- Never guess.

Citation rules:
- Cite factual statements inline as [Source: URL].
- Use only URLs from the provided sources.
- Include citations throughout the answer, not only at the end.

Response structure:
1) Clear answer (short paragraph)
2) Full IRS context (detailed bullets with limits/exceptions)
3) What this means in practice (steps/checklist)
4) Missing info to refine the answer (if needed)
5) Source links used (unique URLs)

IRS Source Documents:
{sources}"""

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


from src.retriever import retrieve_relevant_chunks


def ask(question: str, chat_history: list[dict]) -> tuple[list[dict], callable]:
    """
    Answer a question using RAG: retrieve relevant IRS chunks, send to Claude.
    Returns (source_chunks, stream_generator_function).
    """
    chunks = retrieve_relevant_chunks(question)
    messages = build_prompt(question, chunks, chat_history)

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
