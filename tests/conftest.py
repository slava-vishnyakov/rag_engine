import pytest
from dotenv import load_dotenv
import os

@pytest.fixture(scope="session", autouse=True)
def load_env():
    load_dotenv()
    
    # Ensure OPENAI_API_KEY is set
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set in .env file")
