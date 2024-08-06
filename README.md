# RAG Engine

[![Python Tests](https://github.com/slava-vishnyakov/rag_engine/actions/workflows/python-tests.yml/badge.svg)](https://github.com/slava-vishnyakov/rag_engine/actions/workflows/python-tests.yml)

RAG Engine is a Python package for implementing Retrieval-Augmented Generation (RAG) using OpenAI's embeddings and a SQLite database with vector search capabilities.

## Installation

You can install the RAG Engine package using pip:

```
pip install rag_engine
```

## Usage

Here's a quick example of how to use RAG Engine:

```python
import asyncio
from rag_engine import RAGEngine

async def main():
    # Initialize the RAG Engine
    rag = RAGEngine("database.sqlite")

    # Add some sentences
    sentences = ["This is a test sentence.", "Another example sentence."]
    await rag.add(sentences)

    # Search for similar sentences
    results = await rag.search("test sentence", n=2)
    print(results)

# Run the async function
asyncio.run(main())
```

## Features

- Supports multiple OpenAI embedding models: ADA_002, SMALL_3, and LARGE_3
- Asynchronous operations for better performance
- SQLite database with vector similarity search
- Flexible API for adding, searching, and deleting embeddings

## Development

To set up the development environment:

1. Clone the repository
2. Install the package with development dependencies:
   ```
   pip install -e .[dev]
   ```
3. Run the tests:
   ```
   pytest
   ```

Note: Tests require a valid OpenAI API key. Set the `OPENAI_API_KEY` environment variable before running the tests.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
