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


def build_prompt(
    question: str,
    chunks: list[dict],
    chat_history: list[dict],
) -> list[dict]:
    """Build the message list for the Claude API call."""
    sources_text = ""
    for i, chunk in enumerate(chunks, 1):
        sources_text += f"### Source {i}: {chunk['title']}\nURL: {chunk['source_url']}\n\n{chunk['text']}\n\n---\n\n"

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
