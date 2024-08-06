import pytest
import os
import sys
from dotenv import load_dotenv

# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from rag_engine.database import Database
from rag_engine.embeddings import ADA_002, SMALL_3

load_dotenv()

@pytest.fixture
def db():
    db_file = "test_database.sqlite"
    db = Database(db_file)
    yield db
    db.conn.close()
    os.remove(db_file)

def test_insert_and_search(db):
    texts = ["This is a test", "Another test"]
    embeddings = [([0.1, 0.2, 0.3], ADA_002, 3), ([0.4, 0.5, 0.6], ADA_002, 3)]
    ids = [None, None]
    
    inserted_ids = db.insert_embeddings(texts, embeddings, ids)
    assert len(inserted_ids) == 2
    
    results = db.search_similar([0.1, 0.2, 0.3], 2)
    assert len(results) == 2
    assert results[0]['text'] == "This is a test"
    assert 'similarity' in results[0]

def test_delete_embeddings(db):
    texts = ["Delete me", "Keep me"]
    embeddings = [([0.1, 0.2, 0.3], ADA_002, 3), ([0.4, 0.5, 0.6], ADA_002, 3)]
    ids = [None, None]
    
    inserted_ids = db.insert_embeddings(texts, embeddings, ids)
    db.delete_embeddings([inserted_ids[0]])
    
    results = db.search_similar([0.1, 0.2, 0.3], 2)
    assert len(results) == 1
    assert results[0]['text'] == "Keep me"

def test_embedding_model_consistency(db):
    texts = ["First insertion"]
    embeddings = [([0.1, 0.2, 0.3], ADA_002, 3)]
    ids = [None]
    
    db.insert_embeddings(texts, embeddings, ids)
    
    # Attempt to insert with a different model
    texts2 = ["Second insertion"]
    embeddings2 = [([0.4, 0.5, 0.6], SMALL_3, 3)]
    ids2 = [None]
    
    with pytest.raises(ValueError):
        db.insert_embeddings(texts2, embeddings2, ids2)

def test_get_embedding_model(db):
    texts = ["Test insertion"]
    embeddings = [([0.1, 0.2, 0.3], ADA_002, 3)]
    ids = [None]
    
    db.insert_embeddings(texts, embeddings, ids)
    
    model, size = db.get_embedding_model()
    assert model == ADA_002
    assert size == 3
