#!/usr/bin/env python3
"""Scrape IRS.gov and push to Pinecone. Run locally for periodic refresh."""

import sys
import os
import argparse

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

load_dotenv()

from src.scraper import scrape_irs
from src.indexer import build_index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scrape IRS docs and refresh Pinecone index.")
    parser.add_argument(
        "--full-reindex",
        action="store_true",
        help="Delete all vectors and rebuild from all local JSON docs.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("=== IRS Data Refresh ===")
    print("Step 1: Scraping IRS.gov...")
    doc_count, scraped_urls = scrape_irs(return_urls=True)
    print(f"Scraped {doc_count} documents.\n")

    print("Step 2: Embedding and pushing to Pinecone...")
    mode = "full reindex" if args.full_reindex else "incremental"
    print(f"Index mode: {mode}")
    chunk_count = build_index(
        full_reindex=args.full_reindex,
        changed_urls=None if args.full_reindex else scraped_urls,
    )
    print(f"Indexed {chunk_count} chunks.\n")

    print("=== Refresh complete ===")


if __name__ == "__main__":
    main()
