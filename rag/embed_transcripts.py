"""Embeds text transcripts into a FAISS vector store.

This script reads all .txt files from a specified transcripts directory,
processes them into documents with metadata, splits them into manageable
chunks, and embeds them using a sentence-transformer model. The resulting
vectors are stored in a FAISS index, which is either created anew or
updated if it already exists.

Examples:
    To run the embedding process:
    $ python embed_transcripts.py

"""

import os
from typing import List

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# --- Constants ---
FAISS_INDEX_PATH = "rag/faiss_index"
TRANSCRIPTS_DIR = "data/external/transcripts"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def load_transcripts(directory: str) -> List[Document]:
    """Loads all .txt files from a directory into a list of Documents.

    Args:
        directory: The path to the directory containing transcript files.

    Returns:
        A list of Document objects, each with content and metadata.
    """
    all_docs: List[Document] = []
    transcript_files = [f for f in os.listdir(directory) if f.endswith(".txt")]

    for transcript_file in transcript_files:
        file_path = os.path.join(directory, transcript_file)
        loader = TextLoader(file_path)
        raw_docs = loader.load()

        title = os.path.splitext(transcript_file)[0].replace("_", " ").title()
        docs = [
            Document(
                page_content=doc.page_content,
                metadata={"title": title, "source_type": "transcript"},
            )
            for doc in raw_docs
        ]
        all_docs.extend(docs)
    
    return all_docs


def main() -> None:
    """Main function to run the transcript embedding process."""
    documents = load_transcripts(TRANSCRIPTS_DIR)
    if not documents:
        print("No transcripts found to embed.")
        return

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200
    )
    split_docs = text_splitter.split_documents(documents)

    embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

    if os.path.exists(FAISS_INDEX_PATH):
        print(f"Loading existing FAISS index from {FAISS_INDEX_PATH}.")
        db = FAISS.load_local(
            FAISS_INDEX_PATH,
            embedding_model,
            allow_dangerous_deserialization=True,
        )
        print(f"Adding {len(split_docs)} new document chunks to the index.")
        db.add_documents(split_docs)
    else:
        print("Creating new FAISS index.")
        db = FAISS.from_documents(split_docs, embedding_model)

    db.save_local(FAISS_INDEX_PATH)
    print(
        f"Successfully embedded {len(documents)} transcript(s) into "
        f"{FAISS_INDEX_PATH}."
    )


if __name__ == "__main__":
    main()