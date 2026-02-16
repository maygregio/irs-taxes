# Streamlit Cloud Migration Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Migrate the IRS Tax Assistant from localhost to Streamlit Community Cloud by replacing local ChromaDB/sentence-transformers/BM25 with Pinecone + Voyage AI APIs.

**Architecture:** The app retrieves IRS document chunks via Pinecone vector search (replacing local ChromaDB + BM25 hybrid). Embeddings are generated via Voyage AI API (replacing local sentence-transformers). Indexing runs locally via `scripts/refresh.py` which pushes to Pinecone. The deployed app is read-only (no re-indexing).

**Tech Stack:** Streamlit, Pinecone (`pinecone-client`), Voyage AI (`voyageai`), Anthropic Claude API, LangChain (text splitters only)

---

### Task 1: Update requirements.txt

**Files:**
- Modify: `requirements.txt`

**Step 1: Update requirements**

Replace the contents of `requirements.txt` with:

```
streamlit==1.41.1
langchain==0.3.18
langchain-anthropic==0.3.7
langchain-community==0.3.17
langchain-text-splitters==0.3.6
beautifulsoup4==4.12.3
requests==2.32.3
pdfplumber==0.11.4
voyageai>=0.3.0
pinecone-client>=5.0.0
markdown
pytest==8.3.4
```

Removed: `chromadb`, `sentence-transformers`, `rank-bm25`, `python-dotenv`
Added: `voyageai`, `pinecone-client`, `markdown` (was an implicit dependency used in `app.py:427`)

**Step 2: Install new dependencies**

Run: `pip install -r requirements.txt`
Expected: Installs successfully without PyTorch

**Step 3: Commit**

```bash
git add requirements.txt
git commit -m "chore: update dependencies for Streamlit Cloud migration"
```

---

### Task 2: Rewrite src/retriever.py for Pinecone + Voyage AI

**Files:**
- Modify: `src/retriever.py`
- Modify: `tests/test_retriever.py`

**Step 1: Write the failing tests**

Replace `tests/test_retriever.py` with:

```python
# tests/test_retriever.py
from unittest.mock import MagicMock, patch
from src.retriever import (
    retrieve_relevant_chunks,
    _deduplicate,
    _text_overlap_ratio,
)


def test_retrieve_relevant_chunks_returns_formatted_results():
    mock_query_result = {
        "matches": [
            {
                "id": "chunk_0",
                "score": 0.85,
                "metadata": {
                    "text": "Tax filing deadline is April 15.",
                    "source_url": "https://www.irs.gov/page1",
                    "title": "Filing Deadlines",
                },
            },
            {
                "id": "chunk_1",
                "score": 0.75,
                "metadata": {
                    "text": "Standard deduction is $14,600.",
                    "source_url": "https://www.irs.gov/page2",
                    "title": "Deductions",
                },
            },
        ]
    }

    mock_index = MagicMock()
    mock_index.query.return_value = mock_query_result

    mock_voyage = MagicMock()
    mock_voyage.embed.return_value.embeddings = [[0.1] * 1024]

    with patch("src.retriever._get_pinecone_index", return_value=mock_index), \
         patch("src.retriever._get_voyage_client", return_value=mock_voyage):
        results = retrieve_relevant_chunks("When is the tax deadline?")

    assert len(results) == 2
    assert results[0]["text"] == "Tax filing deadline is April 15."
    assert results[0]["source_url"] == "https://www.irs.gov/page1"
    assert results[1]["text"] == "Standard deduction is $14,600."


def test_score_threshold_filters_irrelevant():
    mock_query_result = {
        "matches": [
            {
                "id": "chunk_0",
                "score": 0.85,
                "metadata": {
                    "text": "Relevant chunk.",
                    "source_url": "https://irs.gov/1",
                    "title": "A",
                },
            },
            {
                "id": "chunk_1",
                "score": 0.15,
                "metadata": {
                    "text": "Irrelevant chunk.",
                    "source_url": "https://irs.gov/2",
                    "title": "B",
                },
            },
        ]
    }

    mock_index = MagicMock()
    mock_index.query.return_value = mock_query_result

    mock_voyage = MagicMock()
    mock_voyage.embed.return_value.embeddings = [[0.1] * 1024]

    with patch("src.retriever._get_pinecone_index", return_value=mock_index), \
         patch("src.retriever._get_voyage_client", return_value=mock_voyage):
        results = retrieve_relevant_chunks("test query")

    assert len(results) == 1
    assert results[0]["text"] == "Relevant chunk."


def test_text_overlap_ratio():
    assert _text_overlap_ratio("the cat sat on the mat", "the cat sat on the mat") == 1.0
    assert _text_overlap_ratio("hello world", "goodbye moon") == 0.0
    assert _text_overlap_ratio("", "anything") == 0.0


def test_deduplicate_removes_overlapping():
    chunks = [
        {"text": "The tax filing deadline for individual returns is April 15 each year"},
        {"text": "The tax filing deadline for individual returns is April 15 each year and extensions"},
        {"text": "Standard deduction amounts vary by filing status"},
    ]
    result = _deduplicate(chunks)
    assert len(result) == 2
    assert result[0]["text"] == chunks[0]["text"]
    assert result[1]["text"] == chunks[2]["text"]
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_retriever.py -v`
Expected: FAIL — imports like `_get_pinecone_index` don't exist yet

**Step 3: Write the implementation**

Replace `src/retriever.py` with:

```python
# src/retriever.py
import os

import voyageai
from pinecone import Pinecone

VOYAGE_MODEL = "voyage-3"
SCORE_THRESHOLD = 0.3
TOP_K_FETCH = 10

_voyage_client = None
_pinecone_index = None


def _get_voyage_client():
    global _voyage_client
    if _voyage_client is None:
        _voyage_client = voyageai.Client(
            api_key=os.environ.get("VOYAGE_API_KEY", ""),
        )
    return _voyage_client


def _get_pinecone_index():
    global _pinecone_index
    if _pinecone_index is None:
        pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY", ""))
        index_name = os.environ.get("PINECONE_INDEX_NAME", "irs-documents")
        _pinecone_index = pc.Index(index_name)
    return _pinecone_index


def _text_overlap_ratio(text_a: str, text_b: str) -> float:
    """Return the fraction of text_a's words that appear in text_b."""
    words_a = set(text_a.lower().split())
    words_b = set(text_b.lower().split())
    if not words_a:
        return 0.0
    return len(words_a & words_b) / len(words_a)


def _deduplicate(chunks: list[dict]) -> list[dict]:
    """Remove chunks where >50% of words overlap with an already-selected chunk."""
    selected = []
    for chunk in chunks:
        is_dup = False
        for kept in selected:
            if _text_overlap_ratio(chunk["text"], kept["text"]) > 0.5:
                is_dup = True
                break
        if not is_dup:
            selected.append(chunk)
    return selected


def retrieve_relevant_chunks(query: str, top_k: int = 5) -> list[dict]:
    """
    Retrieve relevant chunks using Pinecone vector search.
    Embeds the query via Voyage AI, queries Pinecone, filters by score,
    deduplicates, and returns top_k results.
    Returns a list of dicts with 'text', 'source_url', 'title'.
    """
    client = _get_voyage_client()
    result = client.embed([query], model=VOYAGE_MODEL, input_type="query")
    query_embedding = result.embeddings[0]

    index = _get_pinecone_index()
    query_result = index.query(
        vector=query_embedding,
        top_k=TOP_K_FETCH,
        include_metadata=True,
    )

    chunks = []
    for match in query_result["matches"]:
        if match["score"] < SCORE_THRESHOLD:
            continue
        meta = match["metadata"]
        chunks.append(
            {
                "text": meta.get("text", ""),
                "source_url": meta.get("source_url", ""),
                "title": meta.get("title", ""),
            }
        )

    deduped = _deduplicate(chunks)
    return deduped[:top_k]
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_retriever.py -v`
Expected: All 4 tests PASS

**Step 5: Commit**

```bash
git add src/retriever.py tests/test_retriever.py
git commit -m "feat: replace ChromaDB/BM25/sentence-transformers with Pinecone + Voyage AI"
```

---

### Task 3: Rewrite src/indexer.py for Pinecone + Voyage AI

**Files:**
- Modify: `src/indexer.py`
- Modify: `tests/test_indexer.py`

**Step 1: Write the failing test**

Add to `tests/test_indexer.py`:

```python
# tests/test_indexer.py
import json
from unittest.mock import MagicMock, patch, call
from src.indexer import load_documents, chunk_documents


def test_load_documents_reads_json_files(tmp_path):
    doc = {
        "url": "https://www.irs.gov/test",
        "title": "Test Page",
        "content": "This is test content about taxes.",
        "content_type": "forms",
        "last_scraped": "2026-01-01T00:00:00Z",
    }
    with open(tmp_path / "test.json", "w") as f:
        json.dump(doc, f)

    docs = load_documents(str(tmp_path))
    assert len(docs) == 1
    assert docs[0]["content"] == "This is test content about taxes."
    assert docs[0]["url"] == "https://www.irs.gov/test"


def test_load_documents_skips_non_json(tmp_path):
    (tmp_path / "readme.txt").write_text("not a doc")
    docs = load_documents(str(tmp_path))
    assert len(docs) == 0


def test_chunk_documents_splits_long_content():
    docs = [
        {
            "url": "https://www.irs.gov/test",
            "title": "Test",
            "content": "Word " * 500,
            "content_type": "forms",
        }
    ]
    chunks = chunk_documents(docs, chunk_size=500, chunk_overlap=100)
    assert len(chunks) > 1
    assert all("source_url" in c["metadata"] for c in chunks)
    assert all("title" in c["metadata"] for c in chunks)


def test_chunk_documents_preserves_metadata():
    docs = [
        {
            "url": "https://www.irs.gov/form-1040",
            "title": "Form 1040",
            "content": "Short content.",
            "content_type": "forms",
        }
    ]
    chunks = chunk_documents(docs, chunk_size=500, chunk_overlap=100)
    assert len(chunks) == 1
    assert chunks[0]["metadata"]["source_url"] == "https://www.irs.gov/form-1040"
    assert chunks[0]["metadata"]["content_type"] == "forms"


def test_build_index_upserts_to_pinecone(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    doc = {
        "url": "https://www.irs.gov/test",
        "title": "Test",
        "content": "Some tax content for testing.",
        "content_type": "forms",
    }
    with open(raw_dir / "doc.json", "w") as f:
        json.dump(doc, f)

    mock_index = MagicMock()
    mock_pc = MagicMock()
    mock_pc.list_indexes.return_value.names.return_value = ["irs-documents"]
    mock_pc.Index.return_value = mock_index

    mock_voyage = MagicMock()
    mock_voyage.embed.return_value.embeddings = [[0.1] * 1024]

    with patch("src.indexer.Pinecone", return_value=mock_pc), \
         patch("src.indexer.voyageai") as mock_voyageai_mod:
        mock_voyageai_mod.Client.return_value = mock_voyage
        from src.indexer import build_index
        count = build_index(raw_dir=str(raw_dir))

    assert count > 0
    mock_index.upsert.assert_called()
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_indexer.py -v`
Expected: FAIL — `Pinecone` import doesn't exist in indexer yet

**Step 3: Write the implementation**

Replace `src/indexer.py` with:

```python
# src/indexer.py
import json
import os

import voyageai
from pinecone import Pinecone, ServerlessSpec
from langchain_text_splitters import RecursiveCharacterTextSplitter

VOYAGE_MODEL = "voyage-3"
VOYAGE_DIMENSION = 1024
PINECONE_INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME", "irs-documents")
PINECONE_CLOUD = "aws"
PINECONE_REGION = "us-east-1"


def load_documents(raw_dir: str) -> list[dict]:
    """Load all JSON documents from the raw data directory."""
    docs = []
    for filename in os.listdir(raw_dir):
        if not filename.endswith(".json"):
            continue
        filepath = os.path.join(raw_dir, filename)
        with open(filepath) as f:
            doc = json.load(f)
            docs.append(doc)
    return docs


def chunk_documents(
    docs: list[dict], chunk_size: int = 1000, chunk_overlap: int = 200
) -> list[dict]:
    """Split documents into chunks with metadata."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = []
    for doc in docs:
        texts = splitter.split_text(doc["content"])
        for i, text in enumerate(texts):
            chunks.append(
                {
                    "text": text,
                    "metadata": {
                        "source_url": doc["url"],
                        "title": doc.get("title", ""),
                        "content_type": doc.get("content_type", ""),
                        "chunk_index": i,
                    },
                }
            )
    return chunks


def build_index(
    raw_dir: str = "data/raw",
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> int:
    """
    Load documents, chunk them, embed via Voyage AI, and upsert to Pinecone.
    Returns the number of chunks indexed.
    """
    print("Loading documents...")
    docs = load_documents(raw_dir)
    if not docs:
        print("No documents found.")
        return 0

    print(f"Loaded {len(docs)} documents. Chunking...")
    chunks = chunk_documents(docs, chunk_size, chunk_overlap)
    print(f"Created {len(chunks)} chunks. Embedding via Voyage AI...")

    # Initialize clients
    voyage_client = voyageai.Client(
        api_key=os.environ.get("VOYAGE_API_KEY", ""),
    )
    pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY", ""))

    # Create index if it doesn't exist
    existing_indexes = pc.list_indexes().names()
    if PINECONE_INDEX_NAME not in existing_indexes:
        print(f"Creating Pinecone index '{PINECONE_INDEX_NAME}'...")
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=VOYAGE_DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION),
        )

    index = pc.Index(PINECONE_INDEX_NAME)

    # Clear existing vectors
    index.delete(delete_all=True)

    # Embed and upsert in batches
    batch_size = 96  # Voyage AI batch limit
    texts = [c["text"] for c in chunks]

    for i in range(0, len(chunks), batch_size):
        batch_end = min(i + batch_size, len(chunks))
        batch_texts = texts[i:batch_end]

        print(f"  Embedding batch {i // batch_size + 1} ({i}–{batch_end})...")
        result = voyage_client.embed(
            batch_texts, model=VOYAGE_MODEL, input_type="document"
        )
        embeddings = result.embeddings

        vectors = []
        for j, (emb, chunk) in enumerate(
            zip(embeddings, chunks[i:batch_end])
        ):
            vectors.append(
                {
                    "id": f"chunk_{i + j}",
                    "values": emb,
                    "metadata": {
                        "text": chunk["text"],
                        "source_url": chunk["metadata"]["source_url"],
                        "title": chunk["metadata"]["title"],
                        "content_type": chunk["metadata"]["content_type"],
                        "chunk_index": chunk["metadata"]["chunk_index"],
                    },
                }
            )
        index.upsert(vectors=vectors)

    print(f"Indexed {len(chunks)} chunks into Pinecone.")
    return len(chunks)
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_indexer.py -v`
Expected: All 5 tests PASS

**Step 5: Commit**

```bash
git add src/indexer.py tests/test_indexer.py
git commit -m "feat: rewrite indexer to use Pinecone + Voyage AI"
```

---

### Task 4: Update src/chain.py (remove dotenv)

**Files:**
- Modify: `src/chain.py`

**Step 1: Update chain.py**

Remove these two lines at the top of `src/chain.py`:

```python
from dotenv import load_dotenv
```

and:

```python
load_dotenv()
```

No other changes needed. The Anthropic SDK reads `ANTHROPIC_API_KEY` from `os.environ`, and `st.secrets` auto-populates `os.environ` on Streamlit Cloud.

**Step 2: Run existing tests to verify nothing breaks**

Run: `pytest tests/test_chain.py -v`
Expected: All tests PASS (they mock Anthropic, don't depend on dotenv)

**Step 3: Commit**

```bash
git add src/chain.py
git commit -m "refactor: remove dotenv dependency from chain"
```

---

### Task 5: Update app.py for Streamlit Cloud

**Files:**
- Modify: `app.py`

**Step 1: Update app.py**

Make these changes:

1. Remove the `from src.scraper import scrape_irs` and `from src.indexer import build_index` imports (no longer used in the app).

2. Replace the password check to support both `st.secrets` and `os.environ`:

```python
def check_password():
    """Prompt for a password and return True if correct."""
    if st.session_state.get("authenticated"):
        return True

    password = st.text_input("Password", type="password", placeholder="Enter password to continue")
    if password:
        try:
            correct = st.secrets["APP_PASSWORD"]
        except (KeyError, FileNotFoundError):
            correct = os.environ.get("APP_PASSWORD", "")
        if not correct:
            st.error("APP_PASSWORD is not configured.")
            return False
        if hmac.compare_digest(password, correct):
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("Incorrect password.")
    return False
```

3. Replace the sidebar section. Remove the re-index button and raw file count. Replace with a simple info block:

```python
with st.sidebar:
    st.markdown("### Settings")

    st.markdown(
        '<div class="sidebar-badge">IRS Tax Assistant</div>',
        unsafe_allow_html=True,
    )

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.markdown("---")
    st.markdown(
        "<p style='font-size:0.8rem;color:#9ca3af;text-align:center;'>"
        "Data source: <a href='https://www.irs.gov' target='_blank'>IRS.gov</a>"
        "</p>",
        unsafe_allow_html=True,
    )

    st.markdown(
        "<p style='font-size:0.72rem;color:#b0b0b0;text-align:center;line-height:1.4;'>"
        "⚠️ This tool is for informational purposes only and does not constitute "
        "tax, legal, or financial advice. Consult a qualified tax professional "
        "for guidance on your specific situation."
        "</p>",
        unsafe_allow_html=True,
    )
```

4. Remove the `import os` inside the sidebar block (line 370). The top-level `import os` on line 2 is sufficient.

**Step 2: Verify the app still runs locally**

Run: `streamlit run app.py`
Expected: App loads, login works, chat works (assuming Pinecone is populated)

**Step 3: Commit**

```bash
git add app.py
git commit -m "feat: update app.py for Streamlit Cloud deployment"
```

---

### Task 6: Update integration test

**Files:**
- Modify: `tests/test_integration.py`

**Step 1: Rewrite the integration test**

Replace `tests/test_integration.py` with:

```python
# tests/test_integration.py
"""
Integration test: verifies the retrieve -> ask pipeline works end-to-end.
Mocks Pinecone, Voyage AI, and Anthropic — no external calls.
"""
from unittest.mock import patch, MagicMock

from src.retriever import retrieve_relevant_chunks
from src.chain import ask


def test_retrieve_and_ask_pipeline():
    """Test the full query pipeline: embed -> search -> build prompt -> stream."""
    mock_query_result = {
        "matches": [
            {
                "id": "chunk_0",
                "score": 0.9,
                "metadata": {
                    "text": "The tax filing deadline for individual returns is April 15. "
                    "If April 15 falls on a weekend, the deadline is the next business day.",
                    "source_url": "https://www.irs.gov/filing/deadline",
                    "title": "Tax Filing Deadline",
                },
            },
            {
                "id": "chunk_1",
                "score": 0.7,
                "metadata": {
                    "text": "You can request an automatic extension to October 15 by filing Form 4868.",
                    "source_url": "https://www.irs.gov/filing/deadline",
                    "title": "Tax Filing Deadline",
                },
            },
        ]
    }

    mock_index = MagicMock()
    mock_index.query.return_value = mock_query_result

    mock_voyage = MagicMock()
    mock_voyage.embed.return_value.embeddings = [[0.1] * 1024]

    with patch("src.retriever._get_pinecone_index", return_value=mock_index), \
         patch("src.retriever._get_voyage_client", return_value=mock_voyage):
        results = retrieve_relevant_chunks("What is the tax filing deadline?", top_k=2)

    assert len(results) >= 1
    assert any("April 15" in r["text"] for r in results)

    # Now test the full ask() pipeline
    with patch("src.retriever._get_pinecone_index", return_value=mock_index), \
         patch("src.retriever._get_voyage_client", return_value=mock_voyage), \
         patch("src.chain.classify_query", return_value={"is_form_question": False, "forms": [], "query_type": "scenario"}), \
         patch("src.chain.Anthropic"):
        sources, stream = ask("What is the tax filing deadline?", chat_history=[])

    assert len(sources) >= 1
    assert any("April 15" in s["text"] for s in sources)
    assert callable(stream)
```

**Step 2: Run all tests**

Run: `pytest tests/ -v`
Expected: All tests PASS

**Step 3: Commit**

```bash
git add tests/test_integration.py
git commit -m "test: update integration test for Pinecone + Voyage AI"
```

---

### Task 7: Add Streamlit Cloud configuration files

**Files:**
- Create: `.streamlit/secrets.toml` (git-ignored, for local dev)
- Modify: `.gitignore`

**Step 1: Update .gitignore**

Add these lines to `.gitignore`:

```
.streamlit/secrets.toml
```

**Step 2: Create local secrets template**

Create `.streamlit/secrets.toml`:

```toml
ANTHROPIC_API_KEY = "sk-ant-..."
VOYAGE_API_KEY = "pa-..."
PINECONE_API_KEY = "..."
PINECONE_INDEX_NAME = "irs-documents"
APP_PASSWORD = "changeme"
```

**Step 3: Commit**

```bash
git add .gitignore
git commit -m "chore: add Streamlit secrets config to gitignore"
```

---

### Task 8: Update scripts/refresh.py

**Files:**
- Modify: `scripts/refresh.py`

**Step 1: Update refresh.py**

Replace `scripts/refresh.py` with:

```python
#!/usr/bin/env python3
"""Scrape IRS.gov and push to Pinecone. Run locally for periodic refresh."""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

load_dotenv()

from src.scraper import scrape_irs
from src.indexer import build_index


def main():
    print("=== IRS Data Refresh ===")
    print("Step 1: Scraping IRS.gov...")
    doc_count = scrape_irs()
    print(f"Scraped {doc_count} documents.\n")

    print("Step 2: Embedding and pushing to Pinecone...")
    chunk_count = build_index()
    print(f"Indexed {chunk_count} chunks.\n")

    print("=== Refresh complete ===")


if __name__ == "__main__":
    main()
```

Note: This script keeps `python-dotenv` for local use. Since it only runs locally (not on Streamlit Cloud), we add `python-dotenv` as a dev dependency. Add to requirements.txt under a comment or keep it as a local-only install.

**Step 2: Commit**

```bash
git add scripts/refresh.py
git commit -m "refactor: update refresh script for Pinecone indexing"
```

---

### Task 9: Run full test suite and verify

**Step 1: Run all tests**

Run: `pytest tests/ -v`
Expected: All tests PASS

**Step 2: Verify no old imports remain**

Run: `grep -r "chromadb\|sentence_transformers\|SentenceTransformer\|rank_bm25\|BM25Okapi\|from dotenv" src/ app.py --include="*.py"`
Expected: No matches (only `scripts/refresh.py` should have dotenv, which is outside `src/`)

**Step 3: Final commit if any fixes needed**

---

### Task 10: Populate Pinecone and test locally

**Step 1: Set up Pinecone account**

- Sign up at pinecone.io (free tier)
- Create an API key
- Note the API key

**Step 2: Set up Voyage AI account**

- Sign up at voyageai.com (free tier)
- Create an API key

**Step 3: Configure local secrets**

Add real keys to `.streamlit/secrets.toml` (or `.env` for the refresh script).

**Step 4: Run the refresh script to populate Pinecone**

Run: `python scripts/refresh.py`
Expected: Scrapes IRS.gov docs, embeds via Voyage AI, upserts to Pinecone

**Step 5: Test the app locally**

Run: `streamlit run app.py`
Expected: App loads, can ask tax questions, gets answers with sources

**Step 6: Commit any final tweaks**

---

### Task 11: Deploy to Streamlit Community Cloud

**Step 1: Push repo to GitHub** (if not already)

**Step 2: Go to share.streamlit.io**

- Connect your GitHub repo
- Set main file path: `app.py`
- Add secrets in the Streamlit Cloud dashboard (same keys as `.streamlit/secrets.toml`)

**Step 3: Verify the deployed app works**

- Navigate to the deployed URL
- Log in with password
- Ask a tax question
- Verify sources appear

**Step 4: Done!**
