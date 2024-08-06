from openai import AsyncOpenAI, OpenAI
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
    """
    Asynchronously get the embedding for a single text using the specified model.

    :param text: The input text to embed.
    :param api_key: OpenAI API key.
    :param model: The embedding model to use.
    :param size: The size of the embedding vector (only for non-ADA_002 models).
    :return: A tuple containing the embedding vector, model name, and vector size.
    """
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
    
    kwargs = {"input": text, "model": model}
    if model != ADA_002:
        kwargs["dimensions"] = size
    
    response = await client.embeddings.create(**kwargs)
    return response.data[0].embedding, model, size

async def get_embeddings_async(texts: List[str], api_key: str, model: str, size: int = None) -> List[Tuple[List[float], str, int]]:
    """
    Asynchronously get embeddings for multiple texts using the specified model.

    :param texts: List of input texts to embed.
    :param api_key: OpenAI API key.
    :param model: The embedding model to use.
    :param size: The size of the embedding vector (only for non-ADA_002 models).
    :return: A list of tuples, each containing an embedding vector, model name, and vector size.
    """
    tasks = [get_embedding_async(text, api_key, model, size) for text in texts]
    return await asyncio.gather(*tasks)

def get_embeddings(texts: List[str], api_key: str, model: str, size: int = None) -> List[Tuple[List[float], str, int]]:
    """
    Synchronously get embeddings for multiple texts using the specified model.

    :param texts: List of input texts to embed.
    :param api_key: OpenAI API key.
    :param model: The embedding model to use.
    :param size: The size of the embedding vector (only for non-ADA_002 models).
    :return: A list of tuples, each containing an embedding vector, model name, and vector size.
    """
    client = OpenAI(api_key=api_key)
    
    if model not in MODEL_DIMENSIONS:
        raise ValueError(f"Invalid model: {model}")
    
    if size is not None:
        if model == ADA_002:
            raise ValueError(f"Size parameter not supported for {ADA_002}")
        if size < 1 or size > MODEL_DIMENSIONS[model]:
            raise ValueError(f"Invalid size for {model}. Must be between 1 and {MODEL_DIMENSIONS[model]}")
    else:
        size = MODEL_DIMENSIONS[model]
    
    kwargs = {"model": model}
    if model != ADA_002:
        kwargs["dimensions"] = size
    
    results = []
    for text in texts:
        response = client.embeddings.create(input=text, **kwargs)
        results.append((response.data[0].embedding, model, size))
    
    return results
