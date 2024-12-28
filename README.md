# RAG Engine: Powerful Retrieval-Augmented Generation for Python

[![Python Tests](https://github.com/slava-vishnyakov/rag_engine/actions/workflows/python-tests.yml/badge.svg)](https://github.com/slava-vishnyakov/rag_engine/actions/workflows/python-tests.yml)

RAG Engine is a high-performance Python package for implementing Retrieval-Augmented Generation (RAG) using OpenAI's advanced embeddings and a SQLite database with efficient vector search capabilities. Enhance your natural language processing and machine learning projects with state-of-the-art semantic search and text generation.

## Installation

You can install the RAG Engine package using pip:

```
pip install rag_engine
```

## Usage

Here's a quick example of how to use RAG Engine:

```python
from rag_engine import RAGEngine

# Initialize the RAG Engine
rag = RAGEngine("database.sqlite", api_key='...your OpenAI key...')
# or set OPENAI_API_KEY env var

# Add some sentences
sentences = ["This is a test sentence.", "Another example sentence."]
rag.add(sentences)

# Search for similar sentences
results = rag.search("test sentence", n=2)
print(results)

```

## Key Features

- **Advanced Embedding Models**: Supports multiple OpenAI embedding models including ADA_002, SMALL_3, and LARGE_3 for versatile text representation
  `rag_ada = RAGEngine(db_file1, api_key, model=ADA_002)`, `rag_small = RAGEngine(db_file2, api_key, model=SMALL_3, size=512)`
- **High-Performance Asynchronous Operations**: Optimized for speed and efficiency in handling large-scale data
- **Powerful Vector Similarity Search**: Utilizes SQLite database with built-in vector search capabilities for fast and accurate retrieval
- **Flexible and Intuitive API**: Easy-to-use interface for adding, searching, and managing embeddings in your RAG pipeline
- **Seamless Integration**: Designed to work smoothly with existing NLP and machine learning workflows

## Development and Contribution

We welcome contributions to enhance RAG Engine's capabilities. To set up the development environment:

1. Clone the repository: `git clone https://github.com/slava-vishnyakov/rag_engine.git`
2. Install the package with development dependencies:
   ```
   pip install -e .[dev]
   ```
3. Run the comprehensive test suite:
   ```
   pytest
   ```

Note: Running tests requires a valid OpenAI API key. Set the `OPENAI_API_KEY` environment variable before executing the tests.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
