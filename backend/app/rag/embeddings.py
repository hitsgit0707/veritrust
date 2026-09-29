"""Local embedding function interface using ChromaDB's default ONNX MiniLM model."""

from typing import Any
import chromadb.utils.embedding_functions as embedding_functions


def get_embedding_function() -> Any:
    """Return an embedding function using ChromaDB's local, zero-cost ONNX MiniLM engine."""
    return embedding_functions.DefaultEmbeddingFunction()
