import pytest
import os
import sys
from dotenv import load_dotenv

# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from rag_engine.embeddings import get_embeddings_async, ADA_002, SMALL_3, LARGE_3

load_dotenv()

@pytest.mark.asyncio
async def test_get_embeddings_ada_002():
    api_key = os.getenv("OPENAI_API_KEY")
    texts = ["This is a test", "Another test"]
    embeddings = await get_embeddings_async(texts, api_key, ADA_002)
    
    assert len(embeddings) == 2
    assert len(embeddings[0][0]) == 1536  # ADA_002 returns 1536-dimensional vectors
    assert embeddings[0][1] == ADA_002
    assert embeddings[0][2] == 1536

@pytest.mark.asyncio
async def test_get_embeddings_small_3():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        pytest.skip("OPENAI_API_KEY not set")
    
    texts = ["This is a test", "Another test"]
    embeddings = await get_embeddings_async(texts, api_key, SMALL_3, size=512)
    
    assert len(embeddings) == 2
    assert len(embeddings[0][0]) == 512
    assert embeddings[0][1] == SMALL_3
    assert embeddings[0][2] == 512

@pytest.mark.asyncio
async def test_get_embeddings_large_3():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        pytest.skip("OPENAI_API_KEY not set")
    
    texts = ["This is a test", "Another test"]
    embeddings = await get_embeddings_async(texts, api_key, LARGE_3)
    
    assert len(embeddings) == 2
    assert len(embeddings[0][0]) == 3072  # LARGE_3 returns 3072-dimensional vectors by default
    assert embeddings[0][1] == LARGE_3
    assert embeddings[0][2] == 3072

@pytest.mark.asyncio
async def test_invalid_model():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        pytest.skip("OPENAI_API_KEY not set")
    
    with pytest.raises(ValueError):
        await get_embeddings_async(["Test"], api_key, "invalid_model")

@pytest.mark.asyncio
async def test_invalid_size():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        pytest.skip("OPENAI_API_KEY not set")
    
    with pytest.raises(ValueError):
        await get_embeddings_async(["Test"], api_key, SMALL_3, size=5000)
