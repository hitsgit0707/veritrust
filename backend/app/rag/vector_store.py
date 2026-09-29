"""ChromaDB vector store integration supporting persistent on-disk indexing and retrieval."""

import os
from typing import Any, Dict, List, Optional
import chromadb
from pydantic import BaseModel, Field
from backend.app.core.config import get_settings
from backend.app.rag.chunker import DocumentChunk
from backend.app.rag.embeddings import get_embedding_function


class RetrievedChunk(BaseModel):
    """A retrieved knowledge-base evidence chunk returned for a customer query."""
    chunk_id: str
    text: str
    document_id: str
    filename: str
    source: str
    similarity_score: float = Field(..., description="Estimated similarity score (0.0 to 1.0)")
    distance: float = Field(..., description="Raw vector distance metric")
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ChromaVectorStore:
    """Manages persistent ChromaDB client, collections, embeddings, and similarity queries."""

    def __init__(
        self,
        persist_dir: Optional[str] = None,
        collection_name: Optional[str] = None,
    ):
        settings = get_settings()
        self.persist_dir = persist_dir or settings.CHROMA_PERSIST_DIR
        self.collection_name = collection_name or settings.CHROMA_COLLECTION_NAME

        # Ensure directory exists on disk
        os.makedirs(self.persist_dir, exist_ok=True)

        self._embedding_fn = get_embedding_function()
        self._client = chromadb.PersistentClient(path=self.persist_dir)
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self._embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )

    @property
    def collection(self):
        """Access underlying ChromaDB collection."""
        return self._collection

    def add_chunks(self, chunks: List[DocumentChunk]) -> int:
        """Add or update document chunks in the vector store."""
        if not chunks:
            return 0

        ids = [c.chunk_id for c in chunks]
        documents = [c.text for c in chunks]
        metadatas = []

        for c in chunks:
            meta = {
                "chunk_id": c.chunk_id,
                "document_id": c.document_id,
                "filename": c.filename,
                "file_type": c.file_type,
                "chunk_index": c.chunk_index,
                "source": c.source,
            }
            # Flatten primitive metadata items
            for k, v in c.metadata.items():
                if isinstance(v, (str, int, float, bool)):
                    meta[k] = v
            metadatas.append(meta)

        self._collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
        )
        return len(chunks)

    def query(
        self,
        query_text: str,
        top_k: int = 4,
        where: Optional[Dict[str, Any]] = None,
    ) -> List[RetrievedChunk]:
        """Perform semantic similarity search for a natural-language query."""
        total_count = self._collection.count()
        if total_count == 0:
            return []

        # Bound top_k to actual indexed elements
        n_results = min(top_k, total_count)

        kwargs: Dict[str, Any] = {
            "query_texts": [query_text],
            "n_results": n_results,
        }
        if where:
            kwargs["where"] = where

        results = self._collection.query(**kwargs)

        retrieved: List[RetrievedChunk] = []
        if not results or not results["ids"] or not results["ids"][0]:
            return []

        ids_list = results["ids"][0]
        docs_list = results["documents"][0] if results.get("documents") else []
        metas_list = results["metadatas"][0] if results.get("metadatas") else []
        dists_list = results["distances"][0] if results.get("distances") else []

        for idx, chunk_id in enumerate(ids_list):
            doc_text = docs_list[idx] if idx < len(docs_list) else ""
            meta = metas_list[idx] if idx < len(metas_list) else {}
            dist = dists_list[idx] if idx < len(dists_list) else 0.0

            # For cosine distance (range [0, 2]), similarity is 1.0 - (dist / 2.0)
            similarity = max(0.0, min(1.0, 1.0 - (float(dist) / 2.0)))

            retrieved.append(
                RetrievedChunk(
                    chunk_id=chunk_id,
                    text=doc_text,
                    document_id=str(meta.get("document_id", "")),
                    filename=str(meta.get("filename", "")),
                    source=str(meta.get("source", meta.get("filename", ""))),
                    similarity_score=round(similarity, 4),
                    distance=round(float(dist), 4),
                    metadata=meta,
                )
            )

        return retrieved

    def count(self) -> int:
        """Return the number of indexed chunks."""
        return self._collection.count()

    def reset(self) -> None:
        """Delete all chunks from the current collection."""
        self._client.delete_collection(self.collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self._embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )

    def get_stats(self) -> Dict[str, Any]:
        """Return diagnostic metrics of the vector store."""
        return {
            "collection_name": self.collection_name,
            "persist_dir": self.persist_dir,
            "total_chunks": self.count(),
        }
