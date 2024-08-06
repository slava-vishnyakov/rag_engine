from openai import AsyncOpenAI
from typing import List
import asyncio

async def get_embedding_async(text: str, api_key: str) -> List[float]:
    client = AsyncOpenAI(api_key=api_key)
    response = await client.embeddings.create(input=text, model="text-embedding-ada-002")
    return response.data[0].embedding

async def get_embeddings(texts: List[str], api_key: str) -> List[List[float]]:
    tasks = [get_embedding_async(text, api_key) for text in texts]
    return await asyncio.gather(*tasks)
