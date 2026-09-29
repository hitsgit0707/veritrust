"""High-level RAG Service orchestrating document parsing, chunking, storage, and retrieval."""

import os
from functools import lru_cache
from typing import Any, BinaryIO, Dict, List, Optional, Union
from pydantic import BaseModel, Field

from backend.app.core.config import get_settings
from backend.app.rag.chunker import DocumentChunk, DocumentChunker
from backend.app.rag.parser import DocumentParser, ParsedDocument
from backend.app.rag.vector_store import ChromaVectorStore, RetrievedChunk


class IngestionResult(BaseModel):
    """Report for an ingested file."""
    document_id: str
    filename: str
    file_type: str
    chunks_created: int
    status: str = Field(default="success", description="success or failed")
    error: Optional[str] = None


class RAGService:
    """Unified service for ingestion and retrieval used across API and Agents."""

    def __init__(
        self,
        vector_store: Optional[ChromaVectorStore] = None,
        parser: Optional[DocumentParser] = None,
        chunker: Optional[DocumentChunker] = None,
    ):
        self.vector_store = vector_store or ChromaVectorStore()
        self.parser = parser or DocumentParser()
        self.chunker = chunker or DocumentChunker()
        self.settings = get_settings()

    def ingest_file(
        self,
        file_input: Union[str, bytes, BinaryIO],
        filename: str,
        document_id: Optional[str] = None,
    ) -> IngestionResult:
        """Parse, chunk, embed, and store a document into ChromaDB."""
        try:
            parsed_doc: ParsedDocument = self.parser.parse(
                file_input=file_input,
                filename=filename,
                document_id=document_id,
            )
            chunks: List[DocumentChunk] = self.chunker.split_document(parsed_doc)
            count = self.vector_store.add_chunks(chunks)

            return IngestionResult(
                document_id=parsed_doc.document_id,
                filename=filename,
                file_type=parsed_doc.file_type,
                chunks_created=count,
                status="success",
            )
        except Exception as e:
            return IngestionResult(
                document_id=document_id or "unknown",
                filename=filename,
                file_type=self.parser.get_file_extension(filename).lstrip("."),
                chunks_created=0,
                status="failed",
                error=str(e),
            )

    def ingest_directory(self, dir_path: str) -> List[IngestionResult]:
        """Batch ingest all supported documents from a directory on disk."""
        results: List[IngestionResult] = []
        if not os.path.exists(dir_path):
            return results

        for root, _, files in os.walk(dir_path):
            for file in sorted(files):
                if self.parser.is_supported(file):
                    full_path = os.path.join(root, file)
                    result = self.ingest_file(full_path, filename=file)
                    results.append(result)

        return results

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        threshold: Optional[float] = None,
    ) -> List[RetrievedChunk]:
        """
        Retrieve the most relevant knowledge-base chunks for a query.
        
        Note: The similarity threshold is treated as an informative reference rather than
        a hard rejection barrier, ensuring the most relevant contextual evidence is delivered.
        """
        k = top_k or 4
        chunks = self.vector_store.query(query_text=query, top_k=k)

        # Optional filtering if caller strictly requests a minimum threshold
        if threshold is not None:
            chunks = [c for c in chunks if c.similarity_score >= threshold]

        return chunks

    def count(self) -> int:
        """Return total indexed chunks."""
        return self.vector_store.count()

    def reset(self) -> None:
        """Reset the vector store index."""
        self.vector_store.reset()

    def get_stats(self) -> Dict[str, Any]:
        """Return diagnostic metrics."""
        return self.vector_store.get_stats()


@lru_cache
def get_rag_service() -> RAGService:
    """Return a cached singleton instance of RAGService."""
    return RAGService()
