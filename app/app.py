import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Load data
filtered_df = pd.read_csv("data/processed/filtered_papers.csv")

st.title(" AI Consciousness Papers Dashboard")

# 1️ Show total count
st.write(f"**Total Filtered Papers:** {len(filtered_df)}")

# 2️ Keyword search box
query = st.text_input("Search in Titles or Summaries:", "")

if query:
    filtered_view = filtered_df[
        filtered_df["Title"].str.contains(query, case=False, na=False) |
        filtered_df["Summary"].str.contains(query, case=False, na=False)
    ]
else:
    filtered_view = filtered_df

st.write(f"**Showing {len(filtered_view)} results:**")
st.dataframe(filtered_view[["Title", "Summary", "PDF_URL"]])

# 3️ Show plot
st.write("---")
st.subheader("Number of Papers per Month")

# Load your saved plot or recreate
try:
    img = plt.imread("data/processed/papers_per_month.png")
    st.image(img, caption="Papers per Month")
except:
    st.write("No plot found. Please run pipeline.py first.")

# 4️ Download option
st.write("---")
st.download_button(
    label="Download Filtered Papers CSV",
    data=filtered_df.to_csv(index=False).encode('utf-8'),
    file_name="filtered_papers.csv",
    mime='text/csv',
)
