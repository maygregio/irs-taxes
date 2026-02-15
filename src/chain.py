# src/chain.py
import os
from dotenv import load_dotenv
from anthropic import Anthropic

load_dotenv()

SYSTEM_PROMPT_TEMPLATE = """You are an IRS tax assistant. Answer questions ONLY using the provided IRS source documents below. If the answer is not in the sources, say "I don't have enough information from IRS sources to answer that."

Always cite the source URL for each fact you reference. Format citations as [Source: URL] after the relevant statement.

Do not use any outside knowledge. Only use information from the sources below.

## Sources

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


def ask(question: str, chat_history: list[dict]) -> tuple[str, list[dict]]:
    """
    Answer a question using RAG: retrieve relevant IRS chunks, send to Claude.
    Returns (answer_text, source_chunks).
    """
    chunks = retrieve_relevant_chunks(question)
    messages = build_prompt(question, chunks, chat_history)

    system_msg = messages[0]["content"]
    conversation = messages[1:]

    client = Anthropic()
    response = client.messages.create(
        model="claude-sonnet-4-5-20250929",
        max_tokens=2048,
        system=system_msg,
        messages=conversation,
    )

    answer = response.content[0].text
    return answer, chunks
