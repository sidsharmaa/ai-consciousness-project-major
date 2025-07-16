#THIS SCRIPT IS NOT BEING USED IN THE CURRENT EXECUTION
import arxiv
import pandas as pd
from datetime import datetime
import os

# Search Query
query = "AI consciousness OR artificial intelligence consciousness OR machine consciousness"

# Search Config
search = arxiv.Search(
    query=query,
    max_results=50,
    sort_by=arxiv.SortCriterion.SubmittedDate
)

# Fetch and structure results
results = []
for result in search.results():
    results.append({
        "title": result.title,
        "summary": result.summary,
        "published": result.published,
        "updated": result.updated,
        "authors": [author.name for author in result.authors],
        "pdf_url": result.pdf_url,
        "primary_category": result.primary_category,
        "categories": result.categories
    })

'''This will create the directory if it doesn't exist
created it due to an error of non existent directory'''

output_dir = os.path.join("data", "raw")
os.makedirs(output_dir, exist_ok=True)

# Save to CSV
df = pd.DataFrame(results)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_path = os.path.join(output_dir, f"arxiv_ai_consciousness_{timestamp}.csv")
df.to_csv(output_path, index=False)

print(f"✅ {len(df)} papers saved to {output_path}")
