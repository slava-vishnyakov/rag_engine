import os
from dotenv import load_dotenv

def load_env_var(var_name: str) -> str:
    load_dotenv()
    return os.getenv(var_name)
