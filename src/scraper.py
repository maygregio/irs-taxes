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


def discover_links(html: str, base_url: str) -> list[str]:
    """Find all internal IRS.gov links on a page."""
    soup = BeautifulSoup(html, "html.parser")
    links = []
    for a_tag in soup.find_all("a", href=True):
        href = a_tag["href"]
        full_url = urljoin(base_url, href)
        parsed = urlparse(full_url)
        if parsed.hostname and "irs.gov" in parsed.hostname:
            clean_url = f"{parsed.scheme}://{parsed.hostname}{parsed.path}"
            if clean_url not in links:
                links.append(clean_url)
    return links


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

        links = discover_links(html, target["url"])
        links = [target["url"]] + links

        for link in links[:max_pages_per_target]:
            if link in visited:
                continue
            visited.add(link)

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

            time.sleep(1.5)

    print(f"\nDone. Scraped {doc_count} documents.")
    return doc_count
