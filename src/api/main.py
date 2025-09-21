"""
FastAPI server for the AI Consciousness RAG chatbot.

This module creates a web server that exposes the QueryBot's functionality
via a REST API endpoint. It allows other applications to get answers from
the RAG system over the network.
"""
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.config import config
from src.rag_pipeline.bot import QueryBot, format_source_doc 

# Initialize the FastAPI app
app = FastAPI(
    title="AI Consciousness Research Assistant API",
    description="Query a RAG pipeline on academic texts about consciousness.",
    version="0.1.0",
)

# Set up logging
logger = logging.getLogger(__name__)

# --- Pydantic Models for API Data Validation ---

class AskRequest(BaseModel):
    """Defines the schema for a request to the /ask endpoint."""
    query: str
    length: str = "medium"  # Default answer length

class AskResponse(BaseModel):
    """Defines the schema for a response from the /ask endpoint."""
    answer: str
    sources: list[str]

# --- Application Startup ---

# We use a placeholder for the bot to be loaded on startup.
bot: QueryBot | None = None

@app.on_event("startup")
def startup_event():
    """Initializes the QueryBot when the API server starts."""
    global bot
    logger.info("Initializing QueryBot for the API...")
    bot = QueryBot(config.rag_application)
    logger.info("QueryBot initialized successfully.")

# --- API Endpoints ---

@app.post("/ask", response_model=AskResponse)
def ask_question(request: AskRequest):
    """
    Receives a question, gets an answer from the QueryBot, and returns it.
    """
    if not bot:
        raise HTTPException(
            status_code=503, detail="Bot is not initialized. Please wait."
        )

    length_map = config.rag_application.answer_length_map
    if request.length not in length_map:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid length. Choose from: {', '.join(length_map.keys())}",
        )
    
    num_tokens = length_map[request.length]
    
    # Get the raw result from the bot
    result = bot.ask(query=request.query, num_predict_tokens=num_tokens)

    # Format the response for the API
    answer = result.get("result", "No answer found.")
    source_docs = result.get("source_documents", [])
    unique_sources = sorted(list({format_source_doc(doc) for doc in source_docs}))

    return AskResponse(answer=answer, sources=unique_sources)