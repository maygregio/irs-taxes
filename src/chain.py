# src/chain.py
import os
from dotenv import load_dotenv
from anthropic import Anthropic

load_dotenv()

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
