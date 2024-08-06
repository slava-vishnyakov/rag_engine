# RAG Engine

RAG Engine is a Python package for implementing Retrieval-Augmented Generation (RAG) using OpenAI's embeddings and a SQLite database.

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
rag = RAGEngine("database.sqlite")

# Add some sentences
sentences = ["This is a test sentence.", "Another example sentence."]
rag.add(sentences)

# Search for similar sentences
results = rag.search("test sentence", n=2)
print(results)
```

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

Note: Some tests require a valid OpenAI API key. Set the `OPENAI_API_KEY` environment variable before running the tests.

## License

This project is licensed under the MIT License.
