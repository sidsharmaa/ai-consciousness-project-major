"""Loads paper data from a CSV and embeds it into a FAISS vector store.

This script reads a CSV file containing processed paper information, converts
each paper into a LangChain Document, splits the documents into smaller
chunks, and then embeds them using a sentence-transformer model. The
resulting vector store is saved to a local FAISS index.

Examples:
    To run the embedding process:
    $ python load_and_embed.py

"""

import os
from typing import List

import pandas as pd
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# --- Constants ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
CSV_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "filtered_papers.csv")
FAISS_INDEX_PATH = os.path.join(BASE_DIR, "faiss_index")
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def create_documents_from_csv(file_path: str) -> List[Document]:
    """Reads a CSV and converts each row into a Document object.

    Args:
        file_path: The path to the input CSV file.

    Returns:
        A list of Document objects with content and metadata.

    Raises:
        FileNotFoundError: If the specified CSV file does not exist.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"CSV file not found at: {file_path}")

    df = pd.read_csv(file_path)
    docs: List[Document] = []
    for _, row in df.iterrows():
        content = f"Title: {row.get('Title', '')}\n\nSummary: {row.get('Summary', '')}"
        metadata = {
            "title": row.get("Title", "Unknown Title"),
            "published": row.get("Published", "Unknown Date"),
            "primary_category": row.get("Primary_Category", "N/A"),
            "source_type": "arxiv_paper",
        }
        docs.append(Document(page_content=content, metadata=metadata))
    return docs


def main() -> None:
    """Main function to load, process, and embed paper data."""
    print(f"Loading data from: {CSV_PATH}")
    try:
        documents = create_documents_from_csv(CSV_PATH)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    if not documents:
        print("No documents found to process.")
        return

    print(f"Loaded {len(documents)} documents.")

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    split_docs = splitter.split_documents(documents)
    print(f"Split documents into {len(split_docs)} chunks.")

    print(f"Initializing embedding model: {EMBEDDING_MODEL_NAME}")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

    print("Creating FAISS index from documents...")
    db = FAISS.from_documents(split_docs, embeddings)
    
    db.save_local(FAISS_INDEX_PATH)
    print(f"FAISS index successfully saved to {FAISS_INDEX_PATH}")


if __name__ == "__main__":
    main()