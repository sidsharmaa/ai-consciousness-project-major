"""Data processing pipeline for fetching and filtering arXiv papers.

This script defines a pipeline that performs the following steps:
1.  Fetches academic papers from the arXiv API based on a predefined query
    related to AI consciousness.
2.  Filters the fetched papers based on a list of relevant keywords.
3.  Saves the filtered papers to a CSV file.
4.  Generates and saves a plot showing the number of papers published per
    month.

Examples:
    To run the entire data pipeline:
    $ python pipeline.py

"""

import os
from datetime import datetime
from typing import Dict, List

import arxiv
import matplotlib.pyplot as plt
import pandas as pd

# --- Constants ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")

# --- arXiv Query Configuration ---
SEARCH_QUERY = (
    '("AI consciousness" OR "machine consciousness" OR "synthetic consciousness" OR ' 
    '"digital consciousness" OR "artificial consciousness" OR "sentient AI" OR ' 
    '"conscious AI" OR "subjective experience" OR "qualia")'
)
CATEGORY_QUERY = "cat:cs.AI OR cat:phil.CO OR cat:q-bio.NC OR cat:cs.LG OR cs.CL"
FULL_QUERY = f"{SEARCH_QUERY} AND ({CATEGORY_QUERY})"

# --- Filtering Keywords ---
KEYWORDS = [
    "consciousness",
    "sentience",
    "awareness",
    "self-awareness",
    "mind",
    "experience",
]


def fetch_arxiv_papers(query: str) -> List[Dict]:
    """Fetches paper metadata from the arXiv API.

    Args:
        query: The search query string for the arXiv API.

    Returns:
        A list of dictionaries, where each dictionary represents a paper.
    """
    print("Fetching papers from arXiv...")
    search = arxiv.Search(
        query=query,
        max_results=float("inf"),
        sort_by=arxiv.SortCriterion.SubmittedDate,
    )
    client = arxiv.Client(page_size=100, delay_seconds=3)

    results = []
    try:
        for result in client.results(search):
            results.append(
                {
                    "Title": result.title,
                    "Summary": result.summary,
                    "Authors": [a.name for a in result.authors],
                    "Published": result.published,
                    "Updated": result.updated,
                    "PDF_URL": result.pdf_url,
                    "Primary_Category": result.primary_category,
                    "Categories": result.categories,
                }
            )
    except arxiv.UnexpectedEmptyPageError:
        print("No more results found. Ending pagination.")
    
    print(f"Total papers fetched: {len(results)}")
    return results


def filter_papers(papers: pd.DataFrame, keywords: List[str]) -> pd.DataFrame:
    """Filters papers based on keywords in the title and summary.

    Args:
        papers: A DataFrame of papers to filter.
        keywords: A list of keywords to search for.

    Returns:
        A DataFrame containing only the papers that match the keywords.
    """
    print("Filtering papers...")
    papers["text"] = (
        papers["Title"].fillna("") + " " + papers["Summary"].fillna("")
    ).str.lower()
    
    mask = papers["text"].apply(lambda x: any(kw in x for kw in keywords))
    filtered = papers[mask].copy()
    filtered.drop(columns=["text"], inplace=True)
    
    print(f"{len(filtered)} papers remaining after filtering.")
    return filtered


def main() -> None:
    """Main function to run the data pipeline."""
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    # --- Fetch and Save Raw Data ---
    raw_papers_list = fetch_arxiv_papers(FULL_QUERY)
    if not raw_papers_list:
        print("No papers fetched. Exiting pipeline.")
        return
        
    raw_df = pd.DataFrame(raw_papers_list)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    raw_csv_path = os.path.join(RAW_DIR, f"arxiv_papers_{timestamp}.csv")
    raw_df.to_csv(raw_csv_path, index=False)
    print(f"Raw data saved to {raw_csv_path}")

    # --- Filter and Save Processed Data ---
    filtered_df = filter_papers(raw_df, KEYWORDS)
    filtered_csv_path = os.path.join(PROCESSED_DIR, "filtered_papers.csv")
    filtered_df.to_csv(filtered_csv_path, index=False)
    print(f"Filtered data saved to {filtered_csv_path}")

if __name__ == "__main__":
    main()