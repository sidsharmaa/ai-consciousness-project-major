"""
Core logic for the RAG (Retrieval-Augmented Generation) chatbot.

This module defines the QueryBot class, which encapsulates the entire
RAG chain, including the vector store, retriever, and language model.
"""
import logging
from typing import Dict, Any, List

from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM

from src.config import RAGApplicationConfig

logger = logging.getLogger(__name__)

class QueryBot:
    """Encapsulates the RAG chain for querying the consciousness knowledge base."""

    def __init__(self, config: RAGApplicationConfig):
        """
        Initializes the QueryBot with its dependencies and configuration.

        Args:
            config: A Pydantic model containing all necessary configuration.
        """
        self.config = config
        self.chain = self._initialize_chain()

    def _initialize_chain(self) -> RetrievalQA:
        """Builds and returns the fully configured RetrievalQA chain."""
        logger.info(f"Loading FAISS index from: {self.config.faiss_index_path}")
        try:
            embeddings = HuggingFaceEmbeddings(model_name=self.config.embedding_model)
            db = FAISS.load_local(
                folder_path=str(self.config.faiss_index_path),
                embeddings=embeddings,
                allow_dangerous_deserialization=True,
            )
        except Exception as e:
            logger.error(f"Fatal error loading FAISS index: {e}")
            raise SystemExit(1) from e

        llm = OllamaLLM(
            model=self.config.llm.model_name,
            base_url=self.config.llm.base_url,
            # Placeholder, will be updated per-query
            num_predict=self.config.answer_length_map["medium"],
        )
        retriever = db.as_retriever()
        prompt = PromptTemplate(
            template=self.config.prompt_template,
            input_variables=["context", "question"],
        )

        return RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt},
        )

    def ask(self, query: str, num_predict_tokens: int) -> Dict[str, Any]:
        """

        Asks a question to the RAG chain.

        Args:
            query: The user's question.
            num_predict_tokens: The max number of tokens for the LLM response.

        Returns:
            The raw dictionary response from the LangChain chain.
        """
        logger.info(f"Received query: '{query}'")
        # Update the LLM's token count for this specific query
        self.chain.combine_documents_chain.llm_chain.llm.num_predict = num_predict_tokens
        
        return self.chain.invoke(query)

def format_response(result: Dict[str, Any]) -> str:
    """
    Formats the raw RAG output into a user-friendly string.

    Args:
        result: The raw dictionary response from the LangChain chain.

    Returns:
        A formatted string containing the answer and its sources.
    """
    answer = result.get("result", "No answer found.")
    sources = result.get("source_documents", [])
    
    response = f"\nAnswer:\n{answer}"
    
    if sources:
        response += "\n\nSources:"
        unique_sources = { # Remove duplicate sources
            format_source_doc(doc) for doc in sources
        }
        for source_str in sorted(list(unique_sources)):
            response += f"\n - {source_str}"
            
    return response

def format_source_doc(doc: Document) -> str:
    """Helper function to format a single source document."""
    title = doc.metadata.get("title", "Unknown Title")
    if doc.metadata.get("source_type") == "arxiv_paper":
        category = doc.metadata.get("primary_category", "N/A")
        return f"{title} ({category})"
    return title