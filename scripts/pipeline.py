import os
import pandas as pd
import arxiv
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
# Fetch papers
# -----------------------
print("Fetching papers from arXiv...")
search = arxiv.Search(
    query="AI consciousness OR machine consciousness",
    max_results=50,
    sort_by=arxiv.SortCriterion.SubmittedDate
)

results = []
for result in search.results():
    results.append({
        "Title": result.title,
        "Summary": result.summary,
        "Authors": [a.name for a in result.authors],
        "Published": result.published,
        "Updated": result.updated,
        "PDF_URL": result.pdf_url,
        "Primary_Category": result.primary_category,
        "Categories": result.categories
    })

df = pd.DataFrame(results)
today = datetime.today().strftime("%Y-%m-%d")
raw_file = os.path.join(RAW_DIR, f"papers_{today}.csv")
df.to_csv(raw_file, index=False)
print(f"Saved raw papers to {raw_file}")

# -----------------------
# Filter papers
# -----------------------
print("Filtering...")
KEYWORDS = ["consciousness", "sentience", "awareness", "self-awareness", "mind", "experience"]

df["text"] = df["Title"].fillna("") + " " + df["Summary"].fillna("")
df["text"] = df["text"].str.lower()

filtered_df = df[df["text"].apply(lambda x: any(kw in x for kw in KEYWORDS))]
filtered_file = os.path.join(PROCESSED_DIR, "filtered_papers.csv")
filtered_df.to_csv(filtered_file, index=False)
print(f"Saved filtered papers to {filtered_file}")

# -----------------------
# Plot papers per month
# -----------------------
df['Published'] = pd.to_datetime(df['Published'])
monthly_counts = df.set_index('Published').resample('M').size()

plt.figure(figsize=(10, 6))
ax = monthly_counts.plot(kind='bar', color='skyblue')
ax.set_title("Number of Papers per Month")
ax.set_xlabel("Month")
ax.set_ylabel("Number of Papers")
ax.set_xticklabels([ts.strftime('%b %Y') for ts in monthly_counts.index], rotation=45, ha='right')
plt.tight_layout()

plot_file = os.path.join(PROCESSED_DIR, "papers_per_month.png")
plt.savefig(plot_file)
plt.show()
print(f"Plot saved to {plot_file}")
