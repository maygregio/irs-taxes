# IRS Tax Chatbot Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a local RAG chatbot that answers tax questions using only official IRS.gov documentation, with a Streamlit web UI.

**Architecture:** Scrape IRS.gov pages and PDFs, chunk and embed them into ChromaDB, then use LangChain + Claude to answer questions grounded in retrieved IRS content. Streamlit provides the chat interface.

**Tech Stack:** Python 3.12, LangChain, ChromaDB, Streamlit, Anthropic Claude API, BeautifulSoup, pdfplumber, sentence-transformers

**Design doc:** `docs/plans/2026-02-15-irs-chatbot-design.md`

---

### Task 1: Project Setup

**Files:**
- Create: `requirements.txt`
- Create: `.env`
- Create: `.gitignore`
- Create: `src/__init__.py`
- Create: `tests/__init__.py`

**Step 1: Create requirements.txt**

```txt
streamlit==1.41.1
langchain==0.3.18
langchain-anthropic==0.3.7
langchain-community==0.3.17
langchain-text-splitters==0.3.6
chromadb==0.6.3
sentence-transformers==3.4.1
beautifulsoup4==4.12.3
requests==2.32.3
pdfplumber==0.11.4
python-dotenv==1.0.1
pytest==8.3.4
```

**Step 2: Create .env template**

```
ANTHROPIC_API_KEY=your-key-here
```

**Step 3: Create .gitignore**

```
data/
.env
__pycache__/
*.pyc
.venv/
```

**Step 4: Create empty __init__.py files**

Create `src/__init__.py` and `tests/__init__.py` (empty files).

**Step 5: Install dependencies**

Run: `python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`
Expected: All packages install successfully

**Step 6: Create data directories**

Run: `mkdir -p data/raw data/chroma`

**Step 7: Commit**

```bash
git add requirements.txt .gitignore src/__init__.py tests/__init__.py
git commit -m "feat: project setup with dependencies and structure"
```

---

### Task 2: IRS Scraper

**Files:**
- Create: `src/scraper.py`
- Create: `tests/test_scraper.py`

**Step 1: Write the failing test for HTML content extraction**

```python
# tests/test_scraper.py
from src.scraper import extract_page_content


def test_extract_page_content_returns_title_and_text():
    html = """
    <html>
    <head><title>Form 1040 Instructions</title></head>
    <body>
        <nav>Skip this nav</nav>
        <div id="main-content">
            <h1>Form 1040 Instructions</h1>
            <p>Use Form 1040 to file your individual tax return.</p>
        </div>
        <footer>Skip this footer</footer>
    </body>
    </html>
    """
    result = extract_page_content(html, "https://www.irs.gov/forms-instructions/form-1040")
    assert result["title"] == "Form 1040 Instructions"
    assert "Form 1040" in result["content"]
    assert "Skip this nav" not in result["content"]
    assert "Skip this footer" not in result["content"]
    assert result["url"] == "https://www.irs.gov/forms-instructions/form-1040"


def test_extract_page_content_fallback_to_body():
    html = """
    <html>
    <head><title>Some Page</title></head>
    <body>
        <p>Main content here.</p>
    </body>
    </html>
    """
    result = extract_page_content(html, "https://www.irs.gov/some-page")
    assert "Main content here" in result["content"]
```

**Step 2: Run test to verify it fails**

Run: `source .venv/bin/activate && pytest tests/test_scraper.py -v`
Expected: FAIL with "cannot import name 'extract_page_content'"

**Step 3: Implement extract_page_content**

```python
# src/scraper.py
import json
import os
import time
from datetime import datetime, timezone
from urllib.parse import urljoin, urlparse

import pdfplumber
import requests
from bs4 import BeautifulSoup


def extract_page_content(html: str, url: str) -> dict:
    """Extract title and main content from an IRS.gov HTML page."""
    soup = BeautifulSoup(html, "html.parser")

    title_tag = soup.find("title")
    title = title_tag.get_text(strip=True) if title_tag else ""

    # Try to find the main content area
    main = (
        soup.find("div", id="main-content")
        or soup.find("main")
        or soup.find("article")
        or soup.find("div", class_="field--name-body")
    )
    if main is None:
        main = soup.find("body") or soup

    # Remove nav and footer elements from the content
    for tag in main.find_all(["nav", "footer", "header", "script", "style"]):
        tag.decompose()

    content = main.get_text(separator="\n", strip=True)

    return {
        "url": url,
        "title": title,
        "content": content,
        "last_scraped": datetime.now(timezone.utc).isoformat(),
    }
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_scraper.py -v`
Expected: PASS

**Step 5: Write failing test for PDF text extraction**

Add to `tests/test_scraper.py`:

```python
import os
import tempfile
from src.scraper import extract_pdf_text


def test_extract_pdf_text(tmp_path):
    # Create a minimal test by checking the function handles a non-PDF gracefully
    bad_file = tmp_path / "not_a_pdf.pdf"
    bad_file.write_text("this is not a pdf")
    result = extract_pdf_text(str(bad_file))
    assert result == ""
```

**Step 6: Run test to verify it fails**

Run: `pytest tests/test_scraper.py::test_extract_pdf_text -v`
Expected: FAIL with "cannot import name 'extract_pdf_text'"

**Step 7: Implement extract_pdf_text**

Add to `src/scraper.py`:

```python
def extract_pdf_text(pdf_path: str) -> str:
    """Extract text from a PDF file."""
    try:
        text_parts = []
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)
        return "\n".join(text_parts)
    except Exception:
        return ""
```

**Step 8: Run test to verify it passes**

Run: `pytest tests/test_scraper.py -v`
Expected: All PASS

**Step 9: Write failing test for link discovery**

Add to `tests/test_scraper.py`:

```python
from src.scraper import discover_links


def test_discover_links_finds_irs_links():
    html = """
    <html><body>
        <a href="/forms-pubs/about-form-1040">Form 1040</a>
        <a href="/forms-pubs/about-form-w-2">Form W-2</a>
        <a href="https://external.com/page">External</a>
    </body></html>
    """
    links = discover_links(html, "https://www.irs.gov/forms-instructions")
    assert "https://www.irs.gov/forms-pubs/about-form-1040" in links
    assert "https://www.irs.gov/forms-pubs/about-form-w-2" in links
    assert "https://external.com/page" not in links
```

**Step 10: Run test to verify it fails**

Run: `pytest tests/test_scraper.py::test_discover_links_finds_irs_links -v`
Expected: FAIL

**Step 11: Implement discover_links**

Add to `src/scraper.py`:

```python
def discover_links(html: str, base_url: str) -> list[str]:
    """Find all internal IRS.gov links on a page."""
    soup = BeautifulSoup(html, "html.parser")
    links = []
    for a_tag in soup.find_all("a", href=True):
        href = a_tag["href"]
        full_url = urljoin(base_url, href)
        parsed = urlparse(full_url)
        if parsed.hostname and "irs.gov" in parsed.hostname:
            # Remove fragments and query strings for dedup
            clean_url = f"{parsed.scheme}://{parsed.hostname}{parsed.path}"
            if clean_url not in links:
                links.append(clean_url)
    return links
```

**Step 12: Run test to verify it passes**

Run: `pytest tests/test_scraper.py -v`
Expected: All PASS

**Step 13: Implement the main scrape functions (not TDD — integration code)**

Add to `src/scraper.py`:

```python
# Target IRS.gov starting pages
SCRAPE_TARGETS = [
    {"url": "https://www.irs.gov/forms-instructions", "content_type": "forms"},
    {"url": "https://www.irs.gov/publications", "content_type": "publications"},
    {"url": "https://www.irs.gov/taxtopics", "content_type": "taxtopics"},
    {"url": "https://www.irs.gov/faqs", "content_type": "faqs"},
]

HEADERS = {
    "User-Agent": "IRS-Tax-Chatbot/1.0 (personal research tool)"
}


def fetch_page(url: str) -> str | None:
    """Fetch a page from IRS.gov with polite delay."""
    try:
        resp = requests.get(url, headers=HEADERS, timeout=30)
        resp.raise_for_status()
        return resp.text
    except requests.RequestException as e:
        print(f"Failed to fetch {url}: {e}")
        return None


def download_pdf(url: str, save_dir: str) -> str | None:
    """Download a PDF and return the local file path."""
    try:
        resp = requests.get(url, headers=HEADERS, timeout=60)
        resp.raise_for_status()
        filename = os.path.basename(urlparse(url).path)
        filepath = os.path.join(save_dir, filename)
        with open(filepath, "wb") as f:
            f.write(resp.content)
        return filepath
    except requests.RequestException as e:
        print(f"Failed to download PDF {url}: {e}")
        return None


def save_document(doc: dict, output_dir: str) -> None:
    """Save a scraped document as JSON."""
    # Create a safe filename from the URL
    parsed = urlparse(doc["url"])
    safe_name = parsed.path.strip("/").replace("/", "_") or "index"
    filepath = os.path.join(output_dir, f"{safe_name}.json")
    with open(filepath, "w") as f:
        json.dump(doc, f, indent=2)


def scrape_irs(output_dir: str = "data/raw", max_pages_per_target: int = 50) -> int:
    """
    Scrape IRS.gov target pages and save content as JSON files.
    Returns the number of documents saved.
    """
    os.makedirs(output_dir, exist_ok=True)
    pdf_dir = os.path.join(output_dir, "pdfs")
    os.makedirs(pdf_dir, exist_ok=True)

    visited = set()
    doc_count = 0

    for target in SCRAPE_TARGETS:
        print(f"\nScraping {target['content_type']}: {target['url']}")
        html = fetch_page(target["url"])
        if not html:
            continue

        # Get links from the index page
        links = discover_links(html, target["url"])
        # Also process the index page itself
        links = [target["url"]] + links

        for i, link in enumerate(links[:max_pages_per_target]):
            if link in visited:
                continue
            visited.add(link)

            # Check if it's a PDF
            if link.lower().endswith(".pdf"):
                pdf_path = download_pdf(link, pdf_dir)
                if pdf_path:
                    text = extract_pdf_text(pdf_path)
                    if text.strip():
                        doc = {
                            "url": link,
                            "title": os.path.basename(link),
                            "content": text,
                            "content_type": target["content_type"],
                            "last_scraped": datetime.now(timezone.utc).isoformat(),
                        }
                        save_document(doc, output_dir)
                        doc_count += 1
                        print(f"  [{doc_count}] PDF: {link}")
            else:
                page_html = fetch_page(link)
                if page_html:
                    doc = extract_page_content(page_html, link)
                    doc["content_type"] = target["content_type"]
                    save_document(doc, output_dir)
                    doc_count += 1
                    print(f"  [{doc_count}] Page: {link}")

            # Polite delay
            time.sleep(1.5)

    print(f"\nDone. Scraped {doc_count} documents.")
    return doc_count
```

**Step 14: Run all tests**

Run: `pytest tests/test_scraper.py -v`
Expected: All PASS

**Step 15: Commit**

```bash
git add src/scraper.py tests/test_scraper.py
git commit -m "feat: add IRS.gov scraper with HTML/PDF extraction"
```

---

### Task 3: Indexer (Chunking + ChromaDB)

**Files:**
- Create: `src/indexer.py`
- Create: `tests/test_indexer.py`

**Step 1: Write failing test for document loading**

```python
# tests/test_indexer.py
import json
import os
from src.indexer import load_documents


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
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_indexer.py -v`
Expected: FAIL

**Step 3: Implement load_documents**

```python
# src/indexer.py
import json
import os

from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb
from sentence_transformers import SentenceTransformer


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
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_indexer.py -v`
Expected: PASS

**Step 5: Write failing test for chunking**

Add to `tests/test_indexer.py`:

```python
from src.indexer import chunk_documents


def test_chunk_documents_splits_long_content():
    docs = [
        {
            "url": "https://www.irs.gov/test",
            "title": "Test",
            "content": "Word " * 500,  # ~2500 chars
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
```

**Step 6: Run test to verify it fails**

Run: `pytest tests/test_indexer.py::test_chunk_documents_splits_long_content -v`
Expected: FAIL

**Step 7: Implement chunk_documents**

Add to `src/indexer.py`:

```python
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
```

**Step 8: Run test to verify it passes**

Run: `pytest tests/test_indexer.py -v`
Expected: All PASS

**Step 9: Implement build_index (integration — embeds and stores in ChromaDB)**

Add to `src/indexer.py`:

```python
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
COLLECTION_NAME = "irs_documents"


def build_index(
    raw_dir: str = "data/raw",
    chroma_dir: str = "data/chroma",
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> int:
    """
    Load documents, chunk them, embed, and store in ChromaDB.
    Returns the number of chunks indexed.
    """
    print("Loading documents...")
    docs = load_documents(raw_dir)
    if not docs:
        print("No documents found.")
        return 0

    print(f"Loaded {len(docs)} documents. Chunking...")
    chunks = chunk_documents(docs, chunk_size, chunk_overlap)
    print(f"Created {len(chunks)} chunks. Embedding...")

    # Load embedding model
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    # Embed all chunks
    texts = [c["text"] for c in chunks]
    embeddings = model.encode(texts, show_progress_bar=True).tolist()

    # Store in ChromaDB
    client = chromadb.PersistentClient(path=chroma_dir)

    # Delete existing collection if it exists (full rebuild)
    try:
        client.delete_collection(COLLECTION_NAME)
    except ValueError:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    # ChromaDB has a batch size limit, add in batches
    batch_size = 500
    for i in range(0, len(chunks), batch_size):
        batch_end = min(i + batch_size, len(chunks))
        collection.add(
            ids=[f"chunk_{j}" for j in range(i, batch_end)],
            embeddings=embeddings[i:batch_end],
            documents=texts[i:batch_end],
            metadatas=[c["metadata"] for c in chunks[i:batch_end]],
        )

    print(f"Indexed {len(chunks)} chunks into ChromaDB.")
    return len(chunks)
```

**Step 10: Run all tests**

Run: `pytest tests/ -v`
Expected: All PASS

**Step 11: Commit**

```bash
git add src/indexer.py tests/test_indexer.py
git commit -m "feat: add document chunking and ChromaDB indexer"
```

---

### Task 4: Retriever

**Files:**
- Create: `src/retriever.py`
- Create: `tests/test_retriever.py`

**Step 1: Write failing test for retriever**

```python
# tests/test_retriever.py
from unittest.mock import MagicMock, patch
from src.retriever import retrieve_relevant_chunks


def test_retrieve_relevant_chunks_returns_formatted_results():
    # Mock ChromaDB collection query results
    mock_results = {
        "documents": [["Tax filing deadline is April 15.", "Standard deduction is $14,600."]],
        "metadatas": [
            [
                {"source_url": "https://www.irs.gov/page1", "title": "Filing Deadlines"},
                {"source_url": "https://www.irs.gov/page2", "title": "Deductions"},
            ]
        ],
        "distances": [[0.2, 0.4]],
    }

    with patch("src.retriever._get_collection") as mock_col, \
         patch("src.retriever._get_embedding_model") as mock_model:
        mock_col.return_value.query.return_value = mock_results
        mock_model.return_value.encode.return_value = [[0.1] * 384]

        results = retrieve_relevant_chunks("When is the tax deadline?")

    assert len(results) == 2
    assert results[0]["text"] == "Tax filing deadline is April 15."
    assert results[0]["source_url"] == "https://www.irs.gov/page1"
    assert results[1]["text"] == "Standard deduction is $14,600."
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_retriever.py -v`
Expected: FAIL

**Step 3: Implement retriever**

```python
# src/retriever.py
import chromadb
from sentence_transformers import SentenceTransformer

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
COLLECTION_NAME = "irs_documents"
CHROMA_DIR = "data/chroma"

_model = None
_collection = None


def _get_embedding_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _model


def _get_collection():
    global _collection
    if _collection is None:
        client = chromadb.PersistentClient(path=CHROMA_DIR)
        _collection = client.get_collection(COLLECTION_NAME)
    return _collection


def retrieve_relevant_chunks(query: str, top_k: int = 5) -> list[dict]:
    """
    Embed the query and retrieve the top_k most relevant chunks from ChromaDB.
    Returns a list of dicts with 'text', 'source_url', 'title'.
    """
    model = _get_embedding_model()
    query_embedding = model.encode([query]).tolist()

    collection = _get_collection()
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k,
    )

    chunks = []
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        chunks.append(
            {
                "text": doc,
                "source_url": meta.get("source_url", ""),
                "title": meta.get("title", ""),
            }
        )
    return chunks
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_retriever.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/retriever.py tests/test_retriever.py
git commit -m "feat: add retriever for querying ChromaDB"
```

---

### Task 5: RAG Chain (Claude Integration)

**Files:**
- Create: `src/chain.py`
- Create: `tests/test_chain.py`

**Step 1: Write failing test for prompt building**

```python
# tests/test_chain.py
from src.chain import build_prompt


def test_build_prompt_includes_sources_and_question():
    chunks = [
        {"text": "The deadline is April 15.", "source_url": "https://www.irs.gov/p1", "title": "Deadlines"},
        {"text": "Extensions available.", "source_url": "https://www.irs.gov/p2", "title": "Extensions"},
    ]
    messages = build_prompt("When is the deadline?", chunks, chat_history=[])

    # System message should mention IRS
    assert any("IRS" in m["content"] for m in messages if m["role"] == "system")
    # Sources should be included
    system_content = next(m["content"] for m in messages if m["role"] == "system")
    assert "April 15" in system_content
    assert "https://www.irs.gov/p1" in system_content
    # User question should be the last message
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
    # Should have: system, user (history), assistant (history), user (new question)
    assert roles == ["system", "user", "assistant", "user"]
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_chain.py -v`
Expected: FAIL

**Step 3: Implement build_prompt**

```python
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
    # Format sources
    sources_text = ""
    for i, chunk in enumerate(chunks, 1):
        sources_text += f"### Source {i}: {chunk['title']}\nURL: {chunk['source_url']}\n\n{chunk['text']}\n\n---\n\n"

    system_content = SYSTEM_PROMPT_TEMPLATE.format(sources=sources_text)

    messages = [{"role": "system", "content": system_content}]

    # Add chat history (last 10 messages max)
    for msg in chat_history[-10:]:
        messages.append({"role": msg["role"], "content": msg["content"]})

    # Add current question
    messages.append({"role": "user", "content": question})

    return messages
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_chain.py -v`
Expected: PASS

**Step 5: Write failing test for ask function**

Add to `tests/test_chain.py`:

```python
from unittest.mock import patch, MagicMock
from src.chain import ask


def test_ask_calls_claude_and_returns_response():
    mock_chunks = [
        {"text": "Deadline is April 15.", "source_url": "https://irs.gov/d", "title": "Deadlines"}
    ]

    mock_response = MagicMock()
    mock_response.content = [MagicMock(text="The tax deadline is April 15. [Source: https://irs.gov/d]")]

    with patch("src.chain.retrieve_relevant_chunks", return_value=mock_chunks), \
         patch("src.chain.Anthropic") as mock_anthropic_cls:
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client
        mock_client.messages.create.return_value = mock_response

        answer, sources = ask("When is the deadline?", chat_history=[])

    assert "April 15" in answer
    assert len(sources) == 1
    assert sources[0]["source_url"] == "https://irs.gov/d"
```

**Step 6: Run test to verify it fails**

Run: `pytest tests/test_chain.py::test_ask_calls_claude_and_returns_response -v`
Expected: FAIL

**Step 7: Implement ask function**

Add to `src/chain.py`:

```python
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
```

**Step 8: Run test to verify it passes**

Run: `pytest tests/test_chain.py -v`
Expected: All PASS

**Step 9: Commit**

```bash
git add src/chain.py tests/test_chain.py
git commit -m "feat: add RAG chain with Claude API integration"
```

---

### Task 6: Streamlit Chat UI

**Files:**
- Create: `app.py`

**Step 1: Implement the Streamlit app**

```python
# app.py
import streamlit as st
from src.chain import ask
from src.scraper import scrape_irs
from src.indexer import build_index

st.set_page_config(page_title="IRS Tax Assistant", page_icon="📋", layout="centered")
st.title("IRS Tax Assistant")
st.caption("Answers grounded in official IRS.gov documentation")

# Sidebar
with st.sidebar:
    st.header("Settings")
    if st.button("🔄 Re-index IRS Data"):
        with st.spinner("Scraping IRS.gov..."):
            count = scrape_irs()
            st.success(f"Scraped {count} documents.")
        with st.spinner("Building index..."):
            chunks = build_index()
            st.success(f"Indexed {chunks} chunks.")

    st.markdown("---")
    st.markdown("Data source: [IRS.gov](https://www.irs.gov)")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander("📄 Sources"):
                for src in message["sources"]:
                    st.markdown(f"- [{src['title']}]({src['source_url']})")

# Chat input
if prompt := st.chat_input("Ask a tax question..."):
    # Display user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get response
    with st.chat_message("assistant"):
        with st.spinner("Searching IRS documents..."):
            try:
                chat_history = [
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages[:-1]  # exclude current question
                ]
                answer, sources = ask(prompt, chat_history)
                st.markdown(answer)

                if sources:
                    with st.expander("📄 Sources"):
                        seen = set()
                        for src in sources:
                            if src["source_url"] not in seen:
                                st.markdown(f"- [{src['title']}]({src['source_url']})")
                                seen.add(src["source_url"])

                st.session_state.messages.append(
                    {"role": "assistant", "content": answer, "sources": sources}
                )
            except Exception as e:
                st.error(f"Error: {e}. Make sure you've indexed IRS data first (use the sidebar button).")
```

**Step 2: Test manually**

Run: `source .venv/bin/activate && streamlit run app.py`
Expected: Browser opens with the chat UI. The "Re-index IRS Data" button should be visible in the sidebar.

**Step 3: Commit**

```bash
git add app.py
git commit -m "feat: add Streamlit chat UI with source citations"
```

---

### Task 7: Refresh Script + Cron

**Files:**
- Create: `scripts/refresh.py`

**Step 1: Create the refresh script**

```python
#!/usr/bin/env python3
"""Scrape IRS.gov and rebuild the ChromaDB index. Run via cron for periodic refresh."""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.scraper import scrape_irs
from src.indexer import build_index


def main():
    print("=== IRS Data Refresh ===")
    print("Step 1: Scraping IRS.gov...")
    doc_count = scrape_irs()
    print(f"Scraped {doc_count} documents.\n")

    print("Step 2: Building index...")
    chunk_count = build_index()
    print(f"Indexed {chunk_count} chunks.\n")

    print("=== Refresh complete ===")


if __name__ == "__main__":
    main()
```

**Step 2: Make it executable**

Run: `chmod +x scripts/refresh.py`

**Step 3: Test it runs (will actually scrape — only run when ready)**

Run: `source .venv/bin/activate && python scripts/refresh.py`
Expected: Scrapes IRS.gov pages and builds the index. Takes several minutes.

**Step 4: Set up cron (manual step — provide instructions)**

Print cron setup instructions for the user:

```bash
# Edit crontab
crontab -e

# Add this line (runs every Sunday at 2am):
0 2 * * 0 cd /Users/mayara/Documents/irs-taxes && /Users/mayara/Documents/irs-taxes/.venv/bin/python scripts/refresh.py >> data/refresh.log 2>&1
```

**Step 5: Commit**

```bash
git add scripts/refresh.py
git commit -m "feat: add cron-friendly refresh script"
```

---

### Task 8: End-to-End Test

**Files:**
- Create: `tests/test_integration.py`

**Step 1: Write integration test**

```python
# tests/test_integration.py
"""
Integration test: indexes sample documents and queries them.
Does NOT hit IRS.gov or Claude API — uses mocks for external services.
"""
import json
import os
from unittest.mock import patch, MagicMock

from src.indexer import load_documents, chunk_documents, build_index
from src.retriever import retrieve_relevant_chunks


def test_index_and_retrieve(tmp_path):
    """Test the full pipeline: load docs -> chunk -> index -> retrieve."""
    # Create sample documents
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    chroma_dir = tmp_path / "chroma"

    doc1 = {
        "url": "https://www.irs.gov/filing/deadline",
        "title": "Tax Filing Deadline",
        "content": "The tax filing deadline for individual returns is April 15. "
        "If April 15 falls on a weekend or holiday, the deadline is the next business day. "
        "You can request an automatic extension to October 15 by filing Form 4868.",
        "content_type": "forms",
    }
    doc2 = {
        "url": "https://www.irs.gov/deductions/standard",
        "title": "Standard Deduction",
        "content": "The standard deduction for single filers is $14,600 for 2024. "
        "For married filing jointly, the standard deduction is $29,200. "
        "Taxpayers who are 65 or older get an additional standard deduction.",
        "content_type": "publications",
    }

    for i, doc in enumerate([doc1, doc2]):
        with open(raw_dir / f"doc{i}.json", "w") as f:
            json.dump(doc, f)

    # Build the index
    count = build_index(
        raw_dir=str(raw_dir),
        chroma_dir=str(chroma_dir),
        chunk_size=500,
        chunk_overlap=50,
    )
    assert count > 0

    # Query it (patch the retriever's globals to use our test ChromaDB)
    import src.retriever as retriever_mod
    import chromadb
    from sentence_transformers import SentenceTransformer

    old_collection = retriever_mod._collection
    old_model = retriever_mod._model

    try:
        client = chromadb.PersistentClient(path=str(chroma_dir))
        retriever_mod._collection = client.get_collection("irs_documents")
        retriever_mod._model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

        results = retrieve_relevant_chunks("What is the tax filing deadline?", top_k=2)
        assert len(results) == 2
        # The deadline document should be the most relevant
        assert any("April 15" in r["text"] for r in results)
    finally:
        retriever_mod._collection = old_collection
        retriever_mod._model = old_model
```

**Step 2: Run integration test**

Run: `pytest tests/test_integration.py -v`
Expected: PASS (this test uses real embeddings but no external API calls)

**Step 3: Commit**

```bash
git add tests/test_integration.py
git commit -m "test: add end-to-end integration test"
```

---

## Summary

| Task | Description | Est. Steps |
|------|-------------|-----------|
| 1 | Project setup | 7 |
| 2 | IRS scraper | 15 |
| 3 | Indexer (chunking + ChromaDB) | 11 |
| 4 | Retriever | 5 |
| 5 | RAG chain (Claude) | 9 |
| 6 | Streamlit UI | 3 |
| 7 | Refresh script + cron | 5 |
| 8 | End-to-end test | 3 |
