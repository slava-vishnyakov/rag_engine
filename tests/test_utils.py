import pytest
import os
import sys
from dotenv import load_dotenv

# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from rag_engine.utils import load_env_var

load_dotenv()

def test_load_env_var(monkeypatch):
    monkeypatch.setenv("TEST_VAR", "test_value")
    assert load_env_var("TEST_VAR") == "test_value"

def test_load_env_var_missing():
    assert load_env_var("NON_EXISTENT_VAR") is None
