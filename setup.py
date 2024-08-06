from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="rag_engine",
    version="0.1.2",
    author="Slava Vishnyakov",
    author_email="bomboze@gmail.com",
    description="A Retrieval-Augmented Generation (RAG) Engine for managing embeddings and similarity search",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/slava-vishnyakov/rag_engine",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.7",
    install_requires=[
        "openai>=1.0.0",
        "sqlite-vec>=0.1.0",
        "python-dotenv>=0.19.0",
    ],
    extras_require={
        "dev": [
            "bump2version>=1.0.1",
            "pytest>=6.2.5",
            "pytest-asyncio>=0.14.0",
        ],
    },
)
