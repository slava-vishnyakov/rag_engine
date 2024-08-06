import os
from typing import List, Dict, Union
from .database import Database
from .embeddings import get_embeddings, ADA_002, SMALL_3, LARGE_3
from .utils import load_env_var

class RAGEngine:
    """
    Retrieval-Augmented Generation (RAG) Engine for managing embeddings and similarity search.
    """

    def __init__(self, filename: str, api_key: str = None, model: str = ADA_002, size: int = None):
        """
        Initialize the RAG Engine.

        :param filename: Path to the database file.
        :param api_key: OpenAI API key. If None, it will be loaded from environment variables.
        :param model: Embedding model to use. Default is ADA_002.
        :param size: Size of the embedding vector. Only applicable for non-ADA_002 models.
        """
        self.db = Database(filename)
        self.api_key = api_key or load_env_var('OPENAI_API_KEY')
        self.model = model
        self.size = size if model != ADA_002 else None

    async def add(self, sentences: List[Union[str, Dict[str, Union[int, str]]]]) -> List[int]:
        """
        Add sentences to the database and generate their embeddings.

        :param sentences: List of sentences or dictionaries containing 'text' and 'id'.
        :return: List of inserted IDs.
        """
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
        """
        Search for similar sentences based on the query.

        :param query: The search query.
        :param n: Number of results to return. Default is 5.
        :return: List of dictionaries containing similar sentences and their similarity scores.
        """
        query_embedding, model, size = (await get_embeddings([query], self.api_key, self.model, self.size))[0]
        return self.db.search_similar(query_embedding, n)

    def delete_ids(self, ids: List[int]) -> None:
        """
        Delete embeddings with the given IDs from the database.

        :param ids: List of IDs to delete.
        """
        self.db.delete_embeddings(ids)
