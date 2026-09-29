"""API Endpoints for Document Ingestion and RAG Knowledge Base Retrieval."""

from typing import List
from fastapi import APIRouter, File, HTTPException, UploadFile, status

from backend.app.rag.parser import DocumentParser
from backend.app.rag.service import get_rag_service
from backend.app.schemas.rag import (
    BatchUploadResponse,
    RetrieveRequest,
    RetrieveResponse,
)

router = APIRouter(prefix="/api/v1", tags=["Knowledge Base & Documents"])


@router.post(
    "/upload-documents",
    response_model=BatchUploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload and Ingest Knowledge Documents",
    description="Accepts TXT, PDF, DOCX, CSV, and HTML policy files, splits them into chunks, and stores them in ChromaDB.",
)
async def upload_documents(
    files: List[UploadFile] = File(..., description="One or more policy documents to ingest"),
):
    if not files:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No files were provided for upload.",
        )

    rag_service = get_rag_service()
    successful_docs = []
    errors = []

    for file in files:
        filename = file.filename or "unnamed_document"
        if not DocumentParser.is_supported(filename):
            errors.append(
                f"File '{filename}' has an unsupported file format. "
                f"Supported: {', '.join(sorted(DocumentParser.SUPPORTED_EXTENSIONS))}"
            )
            continue

        try:
            content_bytes = await file.read()
            res = rag_service.ingest_file(
                file_input=content_bytes,
                filename=filename,
            )
            if res.status == "success":
                successful_docs.append(res)
            else:
                errors.append(f"Failed to process '{filename}': {res.error}")
        except Exception as e:
            errors.append(f"Unexpected error processing '{filename}': {str(e)}")

    return BatchUploadResponse(
        total_files=len(files),
        successful=len(successful_docs),
        failed=len(errors),
        documents=successful_docs,
        errors=errors,
    )


@router.post(
    "/retrieve",
    response_model=RetrieveResponse,
    summary="Query RAG Knowledge Base Evidence",
    description="Retrieve relevant policy chunks matching a natural-language question for testing or debugging.",
)
async def retrieve_knowledge(payload: RetrieveRequest):
    rag_service = get_rag_service()
    chunks = rag_service.retrieve(
        query=payload.query,
        top_k=payload.top_k,
        threshold=payload.threshold,
    )
    return RetrieveResponse(
        query=payload.query,
        total_retrieved=len(chunks),
        chunks=chunks,
    )


@router.get(
    "/documents/stats",
    summary="Get Knowledge Base Index Metrics",
    description="Returns total indexed chunks, collection name, and persistence directory.",
)
async def get_documents_stats():
    rag_service = get_rag_service()
    return rag_service.get_stats()
