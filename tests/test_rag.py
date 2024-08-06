import pytest
import os
import sys
from dotenv import load_dotenv

# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from rag_engine.rag import RAGEngine
from rag_engine.embeddings import ADA_002, SMALL_3, LARGE_3

load_dotenv()

@pytest.fixture
def rag():
    db_file = "test_rag.sqlite"
    api_key = os.getenv("OPENAI_API_KEY")
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

@pytest.mark.asyncio
async def test_different_models():
    db_file1 = "test_rag_ada.sqlite"
    db_file2 = "test_rag_small.sqlite"
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        pytest.skip("OPENAI_API_KEY not set")
    
    rag_ada = RAGEngine(db_file1, api_key, model=ADA_002)
    rag_small = RAGEngine(db_file2, api_key, model=SMALL_3, size=512)
    
    sentences = ["This is a test sentence."]
    
    await rag_ada.add(sentences)
    await rag_small.add(sentences)
    
    results_ada = await rag_ada.search("test", n=1)
    results_small = await rag_small.search("test", n=1)
    
    assert len(results_ada) == 1
    assert len(results_small) == 1
    assert results_ada[0]['text'] == sentences[0]
    assert results_small[0]['text'] == sentences[0]
    
    os.remove(db_file1)
    os.remove(db_file2)

@pytest.mark.asyncio
async def test_model_consistency():
    db_file = "test_rag_consistency.sqlite"
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        pytest.skip("OPENAI_API_KEY not set")
    
    rag = RAGEngine(db_file, api_key, model=ADA_002)
    
    sentences = ["This is a test sentence."]
    await rag.add(sentences)
    
    # Try to create a new RAGEngine with a different model on the same database
    with pytest.raises(ValueError):
        rag_small = RAGEngine(db_file, api_key, model=SMALL_3)
        await rag_small.add(sentences)
    
    os.remove(db_file)
