import pytest
import os
import sys

# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from rag_engine.database import Database

@pytest.fixture
def db():
    db_file = "test_database.sqlite"
    db = Database(db_file)
    yield db
    db.conn.close()
    os.remove(db_file)

def test_insert_and_search(db):
    texts = ["This is a test", "Another test"]
    embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    ids = [None, None]
    
    inserted_ids = db.insert_embeddings(texts, embeddings, ids)
    assert len(inserted_ids) == 2
    
    results = db.search_similar([0.1, 0.2, 0.3], 2)
    assert len(results) == 2
    assert results[0]['text'] == "This is a test"
    assert 'similarity' in results[0]

def test_delete_embeddings(db):
    texts = ["Delete me", "Keep me"]
    embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    ids = [None, None]
    
    inserted_ids = db.insert_embeddings(texts, embeddings, ids)
    db.delete_embeddings([inserted_ids[0]])
    
    results = db.search_similar([0.1, 0.2, 0.3], 2)
    assert len(results) == 1
    assert results[0]['text'] == "Keep me"
