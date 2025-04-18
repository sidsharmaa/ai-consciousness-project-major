import pandas as pd

# Define keyword list
KEYWORDS = ["consciousness", "sentience", "awareness", "self-awareness", "mind", "experience"]

# Load CSV
df = pd.read_csv("data/raw/papers_2025-04-18.csv")

# Clean column names
df.columns = df.columns.str.strip()

# Combine title + summary for filtering
df["text"] = df["Title"].fillna("") + " " + df["Summary"].fillna("")
df["text"] = df["text"].str.lower()

# Filter rows based on keywords
filtered_df = df[df["text"].apply(lambda x: any(kw in x for kw in KEYWORDS))]

# Save filtered results
filtered_df.to_csv("data/processed/filtered_papers.csv", index=False)

print(f"Filtered {len(filtered_df)} papers containing keywords.")
