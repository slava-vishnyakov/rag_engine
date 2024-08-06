import os
from typing import List, Dict, Union
from .database import Database
from .embeddings import get_embeddings, ADA_002, SMALL_3, LARGE_3
from .utils import load_env_var

class RAGEngine:
    def __init__(self, filename: str, api_key: str = None, model: str = ADA_002, size: int = None):
        self.db = Database(filename)
        self.api_key = api_key or load_env_var('OPENAI_API_KEY')
        self.model = model
        self.size = size

    async def add(self, sentences: List[Union[str, Dict[str, Union[int, str]]]]) -> List[int]:
        texts = []
        ids = []
        for item in sentences:
            if isinstance(item, str):
                texts.append(item)
                ids.append(None)
            else:
                texts.append(item['text'])
                ids.append(item['id'])

        embeddings = await get_embeddings(texts, self.api_key, self.model, self.size)
        
        return self.db.insert_embeddings(texts, embeddings, ids)

    async def search(self, query: str, n: int = 5) -> List[Dict[str, Union[str, float]]]:
        query_embedding, model, size = (await get_embeddings([query], self.api_key, self.model, self.size))[0]
        return self.db.search_similar(query_embedding, n)

    def delete_ids(self, ids: List[int]):
        self.db.delete_embeddings(ids)
