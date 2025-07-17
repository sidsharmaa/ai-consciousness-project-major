import os
import pandas as pd
import arxiv
import time
from datetime import datetime
import matplotlib.pyplot as plt

# -----------------------
# Set base directories
# -----------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")

os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

# -----------------------
# Fetch all matching papers
# -----------------------
print(" Fetching all AI consciousness-related papers from arXiv...")

query_str = '"AI consciousness" OR "machine consciousness" OR "synthetic consciousness" OR "digital consciousness" OR "artificial consciousness" OR "sentient AI" OR "conscious AI" OR "subjective experience" OR "qualia"'
category_query = 'cat:cs.AI OR cat:phil.CO OR cat:q-bio.NC OR cat:cs.LG OR cat:cs.CL'
query_str = f'({query_str}) AND ({category_query})'


search = arxiv.Search(
    query=query_str,
    max_results=float('inf'),
    sort_by=arxiv.SortCriterion.SubmittedDate
)

client = arxiv.Client(
  page_size = 100,
  delay_seconds = 3
)

all_results = []
try:
    for result in client.results(search):
        all_results.append({
            "Title": result.title,
            "Summary": result.summary,
            "Authors": [a.name for a in result.authors],
            "Published": result.published,
            "Updated": result.updated,
            "PDF_URL": result.pdf_url,
            "Primary_Category": result.primary_category,
            "Categories": result.categories
        })
except arxiv.UnexpectedEmptyPageError:
    print("No more results found. Ending pagination.")

print(f" Total papers fetched: {len(all_results)}")

# Convert to DataFrame
df = pd.DataFrame(all_results)


# -----------------------
# Filter papers
# -----------------------
print(" Filtering papers with relevant keywords...")
KEYWORDS = ["consciousness", "sentience", "awareness", "self-awareness", "mind", "experience"]

df["text"] = df["Title"].fillna("") + " " + df["Summary"].fillna("")
df["text"] = df["text"].str.lower()

filtered_df = df[df["text"].apply(lambda x: any(kw in x for kw in KEYWORDS))]
filtered_file = os.path.join(PROCESSED_DIR, "filtered_papers.csv")
filtered_df.to_csv(filtered_file, index=False)
print(f" Saved filtered papers to {filtered_file}")

# -----------------------
# Plot papers per day
# -----------------------
