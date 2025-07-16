import os
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
import pandas as pd

# -----------------------
# Config & Paths
# -----------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
CSV_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "filtered_papers.csv")
FAISS_INDEX = os.path.join(BASE_DIR, "faiss_index")

print("CSV_PATH:", CSV_PATH)
print("FAISS_INDEX:", FAISS_INDEX)

# -----------------------
# Load data
# -----------------------
df = pd.read_csv(CSV_PATH)
docs = []
for _, row in df.iterrows():
    content = f"Title: {row['Title']}\n\nSummary: {row['Summary']}"
    metadata = {
        "title": row['Title'],
        "published": row.get('Published', 'unknown')
    }
    docs.append(Document(page_content=content, metadata=metadata))

# -----------------------
# Split & Embed
# -----------------------
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
split_docs = splitter.split_documents(docs)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

db = FAISS.from_documents(split_docs, embeddings)
db.save_local(FAISS_INDEX)

print(f" FAISS index saved to {FAISS_INDEX}")
