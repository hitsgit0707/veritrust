"""Schemas for RAG ingestion and retrieval API endpoints."""

from typing import List, Optional
from pydantic import BaseModel, Field
from backend.app.rag.service import IngestionResult
from backend.app.rag.vector_store import RetrievedChunk


class BatchUploadResponse(BaseModel):
    """Response payload for multi-file document upload and ingestion."""
    total_files: int
    successful: int
    failed: int
    documents: List[IngestionResult]
    errors: List[str] = Field(default_factory=list)


class RetrieveRequest(BaseModel):
    """Payload for querying the RAG knowledge base independently."""
    query: str = Field(..., min_length=1, max_length=1000, description="Customer question or search phrase")
    top_k: Optional[int] = Field(default=4, ge=1, le=20, description="Maximum number of chunks to return")
    threshold: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Optional minimum similarity filter")


class RetrieveResponse(BaseModel):
    """Response payload containing retrieved knowledge chunks."""
    query: str
    total_retrieved: int
    chunks: List[RetrievedChunk]
