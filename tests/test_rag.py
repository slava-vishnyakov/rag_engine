import pytest
import os
import sys

# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from rag_engine.rag import RAGEngine

@pytest.fixture
def rag():
    db_file = "test_rag.sqlite"
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        pytest.skip("OPENAI_API_KEY not set")
    rag = RAGEngine(db_file, api_key)
    yield rag
    os.remove(db_file)

@pytest.mark.asyncio
async def test_add_and_search(rag):
    sentences = ["This is a test sentence.", "Another example sentence."]
    await rag.add(sentences)
    
    results = await rag.search("test sentence", n=2)
    assert len(results) == 2
    assert results[0]['text'] == "This is a test sentence."

@pytest.mark.asyncio
async def test_delete_ids(rag):
    sentences = [
        {"id": 1, "text": "Delete me"},
        {"id": 2, "text": "Keep me"}
    ]
    await rag.add(sentences)
    
    rag.delete_ids([1])
    
    results = await rag.search("Delete", n=2)
    assert len(results) == 1
    assert results[0]['text'] == "Keep me"
