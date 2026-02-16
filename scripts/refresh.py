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
