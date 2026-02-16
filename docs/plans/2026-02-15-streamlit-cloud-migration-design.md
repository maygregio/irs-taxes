# Streamlit Cloud Migration Design

## Summary

Migrate the IRS Tax Assistant from a localhost-only app to Streamlit Community Cloud by replacing local-heavy resources (embedded ML model, local ChromaDB, BM25) with cloud-hosted services (Voyage AI embeddings, Pinecone vector DB).

## Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Vector DB | Pinecone (free tier) | 2GB free storage, simple API, no persistent filesystem on Streamlit Cloud |
| Embeddings | Voyage AI API | Eliminates PyTorch/sentence-transformers (~2GB), Anthropic ecosystem |
| BM25 keyword search | Drop entirely | Reduces memory, simplifies architecture, Pinecone vector search is sufficient |
| Data refresh | Local script only | Run `scripts/refresh.py` locally to scrape + push to Pinecone |
| Secrets | `st.secrets` | Streamlit Cloud native secrets management |

## Architecture Changes

### src/retriever.py (major rewrite)

**Remove:** `SentenceTransformer`, `chromadb`, `rank_bm25`, BM25 search, RRF fusion

**New flow:** Embed query via Voyage AI API → query Pinecone → filter by score → deduplicate → return

**Public API unchanged:** `retrieve_relevant_chunks(query, top_k) -> list[dict]`

### src/indexer.py (major rewrite)

**Remove:** `SentenceTransformer`, `chromadb`, BM25 corpus generation

**New flow:** Load docs from `data/raw/` → chunk → batch embed via Voyage AI → upsert to Pinecone

**Runs locally only** — never on Streamlit Cloud.

Creates Pinecone index if it doesn't exist. Voyage embedding dimension (1024 for `voyage-3`). Metadata: `source_url`, `title`, `content_type`, `chunk_index`.

### app.py (moderate changes)

- Replace `os.environ.get("APP_PASSWORD")` with `st.secrets["APP_PASSWORD"]`
- Remove "Re-index IRS Data" sidebar button
- Remove `data/raw` document count badge (or replace with Pinecone vector count)
- Keep all chat UI, streaming, sources, styling

### src/chain.py (minimal changes)

- Remove `from dotenv import load_dotenv` and `load_dotenv()` call
- `st.secrets` auto-populates `os.environ`, so Anthropic SDK picks up the key automatically

### scripts/refresh.py (no changes)

Same structure: scrape then index. "Index" now means pushing to Pinecone.

### Configuration

**New secrets:** `VOYAGE_API_KEY`, `PINECONE_API_KEY`, `PINECONE_INDEX_NAME`

**New files:**
- `.streamlit/secrets.toml` (git-ignored) for local dev

**`.gitignore` additions:** `secrets.toml`

### requirements.txt

**Remove:** `sentence-transformers`, `rank-bm25`, `chromadb`, `python-dotenv`

**Add:** `voyageai`, `pinecone-client`

**Keep:** `streamlit`, `langchain`, `langchain-anthropic`, `langchain-community`, `langchain-text-splitters`, `beautifulsoup4`, `requests`, `pdfplumber`, `pytest`

### Error handling

No new error handling. Existing try/except in `app.py` surfaces service errors in chat. Pinecone/Voyage failures propagate naturally.

### Tests

Update existing test mocks to target Pinecone/Voyage instead of ChromaDB/SentenceTransformer. No new test files.
