"""CLI RAG chatbot for answering questions related to AI consciousness."""

import os
from typing import List

from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document

# ------------------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FAISS_INDEX_PATH = os.path.join(BASE_DIR, "faiss_index")
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL_NAME = "mistral"

# ------------------------------------------------------------------------------
# Utility Functions
# ------------------------------------------------------------------------------


def format_source(doc: Document) -> str:
    """Format metadata from a Document object for display.

    Args:
        doc: A LangChain Document with metadata.

    Returns:
        A formatted string describing the source.

    Examples:
        >>> doc = Document(page_content="...", metadata={"title": "My Paper", "source_type": "arxiv_paper", "primary_category": "cs.AI"})
        >>> format_source(doc)
        'My Paper (cs.AI)'

        >>> doc = Document(page_content="...", metadata={"title": "My Transcript", "source_type": "transcript"})
        >>> format_source(doc)
        'My Transcript'
    """
    title = doc.metadata.get("title", "Unknown Title")
    source_type = doc.metadata.get("source_type", "Unknown")

    if source_type == "arxiv_paper":
        category = doc.metadata.get("primary_category", "N/A")
        return f"{title} ({category})"

    return title


# ------------------------------------------------------------------------------
# Main Function
# ------------------------------------------------------------------------------


def main() -> None:
    """Initialize and launch the RAG chatbot CLI.

    This function sets up the environment, loads the FAISS index and embeddings,
    initializes the local LLM, and runs a command-line interaction loop for
    answering questions related to AI consciousness.

    Raises:
        SystemExit: If the FAISS index cannot be loaded.

    Example:
        Run the chatbot:
            $ python query_bot.py
    """
    load_dotenv()

    print(f"Loading FAISS index from: {FAISS_INDEX_PATH}")
    try:
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
        db = FAISS.load_local(
            FAISS_INDEX_PATH,
            embeddings,
            allow_dangerous_deserialization=True,
        )
    except Exception as e:
        print(f"Error loading FAISS index: {e}")
        print("Please run one of the following to build the index:")
        print("  python rag/load_and_embed.py")
        print("  python rag/embed_transcripts.py")
        raise SystemExit(1)

    llm = Ollama(model=LLM_MODEL_NAME)
    retriever = db.as_retriever()

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
    )

    print("\nAsk your questions (Type 'exit' to quit):\n")

    while True:
        query = input("You: ").strip()

        if query.lower() in {"exit", "quit"}:
            break

        if not query:
            continue

        result = qa_chain.invoke(query)
        answer = result.get("result", "No answer found.")
        sources: List[Document] = result.get("source_documents", [])

        print(f"\nAnswer:\n{answer}")
        if sources:
            print("\nSources:")
            for doc in sources:
                print(f" - {format_source(doc)}")

        print("\n" + "-" * 50)


if __name__ == "__main__":
    main()
