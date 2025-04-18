import arxiv
import pandas as pd
from datetime import datetime

# -------- SETTINGS --------
SEARCH_QUERY = "AI consciousness"
RESULT_LIMIT = 50  # Number of papers to fetch
SORT_BY = arxiv.SortCriterion.SubmittedDate  # Most recent papers
OUTPUT_CSV = f"papers_{datetime.now().strftime('%Y-%m-%d')}.csv"
# --------------------------

def fetch_arxiv_papers(query, limit=50, sort_by=arxiv.SortCriterion.SubmittedDate):
    search = arxiv.Search(
        query=query,
        max_results=limit,
        sort_by=sort_by,
        sort_order=arxiv.SortOrder.Descending,
    )
    
    results = []
    for result in search.results():
        results.append({
            "Title": result.title,
            "Authors": ', '.join([author.name for author in result.authors]),
            "Summary": result.summary,
            "Published": result.published.strftime('%Y-%m-%d'),
            "Updated": result.updated.strftime('%Y-%m-%d'),
            "URL": result.entry_id,
            "Primary Category": result.primary_category,
        })
    
    return results

def save_to_csv(data, filename):
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
    print(f"✅ Saved {len(data)} papers to {filename}")

if __name__ == "__main__":
    print(f"🔍 Searching arXiv for: '{SEARCH_QUERY}'...")
    papers = fetch_arxiv_papers(SEARCH_QUERY, RESULT_LIMIT, SORT_BY)
    save_to_csv(papers, OUTPUT_CSV)
