import pytest
from unittest.mock import MagicMock, patch
from langchain.schema import Document

from src.rag_pipeline.bot import format_response

def test_format_response_with_sources():
    """
    Tests if the format_response function correctly formats an answer
    with both arXiv and transcript sources.
    """
    # 1. Arrange: Create a fake RAG result dictionary
    fake_result = {
        "result": "Consciousness is a complex phenomenon.",
        "source_documents": [
            Document(
                page_content="...",
                metadata={
                    "title": "A Paper on AI",
                    "source_type": "arxiv_paper",
                    "primary_category": "cs.AI",
                },
            ),
            Document(
                page_content="...",
                metadata={
                    "title": "An Expert Transcript",
                    "source_type": "transcript",
                },
            ),
        ],
    }

    # 2. Act: Call the function we are testing
    formatted_string = format_response(fake_result)

    # 3. Assert: Check if the output is what we expect
    assert "Answer:" in formatted_string
    assert "Consciousness is a complex phenomenon." in formatted_string
    assert "Sources:" in formatted_string
    assert "A Paper on AI (cs.AI)" in formatted_string
    assert "An Expert Transcript" in formatted_string

def test_format_response_no_sources():
    """
    Tests if the format_response function works correctly when no sources
    are returned.
    """
    # 1. Arrange
    fake_result = {
        "result": "I don't know.",
        "source_documents": [],
    }

    # 2. Act
    formatted_string = format_response(fake_result)

    # 3. Assert
    assert "Answer:" in formatted_string
    assert "I don't know." in formatted_string
    assert "Sources:" not in formatted_string