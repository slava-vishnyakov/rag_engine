from openai import AsyncOpenAI
from typing import List, Tuple
import asyncio

# Constants for embedding models
ADA_002 = "text-embedding-ada-002"
SMALL_3 = "text-embedding-3-small"
LARGE_3 = "text-embedding-3-large"

# Dimension sizes for each model
MODEL_DIMENSIONS = {
    ADA_002: 1536,
    SMALL_3: 1536,
    LARGE_3: 3072
}

async def get_embedding_async(text: str, api_key: str, model: str, size: int = None) -> Tuple[List[float], str, int]:
    client = AsyncOpenAI(api_key=api_key)
    
    if model not in MODEL_DIMENSIONS:
        raise ValueError(f"Invalid model: {model}")
    
    if size is not None:
        if model == ADA_002:
            raise ValueError(f"Size parameter not supported for {ADA_002}")
        if size < 1 or size > MODEL_DIMENSIONS[model]:
            raise ValueError(f"Invalid size for {model}. Must be between 1 and {MODEL_DIMENSIONS[model]}")
    else:
        size = MODEL_DIMENSIONS[model]
    
    response = await client.embeddings.create(input=text, model=model, dimensions=size)
    return response.data[0].embedding, model, size

async def get_embeddings(texts: List[str], api_key: str, model: str, size: int = None) -> List[Tuple[List[float], str, int]]:
    tasks = [get_embedding_async(text, api_key, model, size) for text in texts]
    return await asyncio.gather(*tasks)
