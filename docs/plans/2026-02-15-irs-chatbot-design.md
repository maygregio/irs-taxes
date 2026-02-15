# IRS Tax Chatbot — Design Document

**Date:** 2026-02-15
**Status:** Approved

## Summary

A personal, locally-run chatbot that answers tax questions using only official IRS documentation. Uses RAG (Retrieval-Augmented Generation) to ground Claude's responses in scraped IRS.gov content.

## Requirements

- **Audience:** Personal use, running locally
- **Content scope:** Broad IRS.gov content — forms, instructions, publications, FAQs, tax topics
- **Interface:** Web UI (Streamlit)
- **LLM:** Claude via Anthropic API
- **Freshness:** Periodic automatic refresh via cron

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌───────────┐
│  IRS.gov    │────▶│   Scraper    │────▶│  Chunker  │
│  (source)   │     │ (BeautifulSoup│     │ (text     │
│             │     │  + requests)  │     │  splitter)│
└─────────────┘     └──────────────┘     └─────┬─────┘
                                               │
                                               ▼
┌─────────────┐     ┌──────────────┐     ┌───────────┐
│  Streamlit  │◀───▶│  RAG Chain   │◀───▶│ ChromaDB  │
│  Chat UI    │     │ (LangChain + │     │ (vectors) │
│             │     │  Claude API) │     │           │
└─────────────┘     └──────────────┘     └───────────┘
```

### Components

1. **Scraper** — Crawls IRS.gov target pages, extracts text from HTML and PDFs, stores as structured JSON.
2. **Indexer** — Chunks content (~1,000 tokens, 200 overlap), generates embeddings with `all-MiniLM-L6-v2`, stores in ChromaDB.
3. **RAG Chain** — Embeds user query, retrieves top 5 chunks, sends to Claude with a system prompt restricting answers to IRS sources only.
4. **Streamlit UI** — Chat interface with conversation history and source citations.

## Scraper Design

### Target content

| Content type | Starting URL | What we extract |
|---|---|---|
| Forms & instructions | `/forms-instructions` | Form names, numbers, PDFs, instruction text |
| Publications | `/publications` | Full publication text |
| Tax topics | `/taxtopics` | Topic pages |
| FAQs | `/faqs` | Question/answer pairs |

### Behavior

- Start from each index page, follow links to individual pages
- Extract main content area only (skip nav, footer, sidebars)
- PDFs: download and extract text using `pdfplumber`
- Save each page as JSON: `{ url, title, content, content_type, last_scraped }`
- Respect `robots.txt`, 1-2 second delay between requests
- Store under `data/raw/`
- Skip interactive tools and image-only content

## Indexing & Retrieval

### Chunking

- `RecursiveCharacterTextSplitter` (~1,000 tokens, 200 overlap)
- Metadata per chunk: `{ source_url, title, content_type, chunk_index }`

### Vector store

- ChromaDB, persistent storage at `data/chroma/`
- Collection: `irs_documents`
- On re-index: delete and rebuild the collection

### Retrieval

- Embed query with `all-MiniLM-L6-v2`
- Top 5 chunks by cosine similarity
- Last 10 messages of conversation history included

### Claude prompt

```
System: You are an IRS tax assistant. Answer questions ONLY using
the provided IRS source documents below. If the answer is not in
the sources, say "I don't have enough information from IRS sources
to answer that." Always cite the source URL for each fact.

Sources:
{retrieved chunks with URLs}
```

## Project Structure

```
irs-taxes/
├── app.py                  # Streamlit entry point
├── src/
│   ├── scraper.py          # IRS.gov scraper
│   ├── indexer.py          # Chunking + ChromaDB indexing
│   ├── retriever.py        # Query embedding + retrieval
│   └── chain.py            # RAG chain (Claude + retrieved context)
├── data/
│   ├── raw/                # Scraped JSON files
│   └── chroma/             # ChromaDB persistent storage
├── scripts/
│   └── refresh.py          # Cron-friendly script: scrape + re-index
├── requirements.txt
├── .env                    # ANTHROPIC_API_KEY
└── .gitignore              # Ignores data/, .env
```

## Dependencies

- `streamlit`
- `langchain`, `langchain-anthropic`, `langchain-community`
- `chromadb`
- `sentence-transformers`
- `beautifulsoup4`, `requests`
- `pdfplumber`
- `python-dotenv`

## Embedding model

`sentence-transformers/all-MiniLM-L6-v2` — local, free, no API key needed.

## Periodic refresh

Cron job running `scripts/refresh.py` weekly (e.g., Sunday 2am). Re-scrapes all target pages and rebuilds the ChromaDB index.
