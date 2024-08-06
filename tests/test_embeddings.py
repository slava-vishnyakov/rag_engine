import pytest
import os
import sys

# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from rag_engine.embeddings import get_embeddings

@pytest.mark.asyncio
async def test_get_embeddings():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        pytest.skip("OPENAI_API_KEY not set")
    
    texts = ["This is a test", "Another test"]
    embeddings = await get_embeddings(texts, api_key)
    
    assert len(embeddings) == 2
    assert len(embeddings[0]) == 1536  # OpenAI's text-embedding-ada-002 model returns 1536-dimensional vectors
