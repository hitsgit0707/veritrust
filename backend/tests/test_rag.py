"""Tests for Stage 3: RAG Knowledge Base, Document Parsers, Chunker, ChromaDB, and Retrieval."""

import io
import os
import shutil
import tempfile
import pytest
from httpx import ASGITransport, AsyncClient

from backend.app.main import app
from backend.app.rag.chunker import DocumentChunker
from backend.app.rag.parser import DocumentParser
from backend.app.rag.service import RAGService
from backend.app.rag.vector_store import ChromaVectorStore


@pytest.fixture
def temp_chroma_dir():
    """Create a temporary directory for isolated ChromaDB tests."""
    temp_dir = tempfile.mkdtemp(prefix="veritrust_test_chroma_")
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


# ==============================================================================
# 1. Multi-Format Parser Tests
# ==============================================================================

def test_parse_txt():
    """Verify TXT parsing."""
    parser = DocumentParser()
    content = b"Company refund policy allows returns within 30 days."
    doc = parser.parse(content, filename="policy.txt")
    assert doc.file_type == "txt"
    assert len(doc.sections) == 1
    assert "30 days" in doc.sections[0].text


def test_parse_html():
    """Verify HTML parsing strips scripts and extracts text."""
    parser = DocumentParser()
    html_content = b"""
    <html>
      <head><style>body { color: red; }</style></head>
      <body>
        <script>alert('malicious');</script>
        <h1>Hardware Warranty</h1>
        <p>1-year limited coverage for manufacturing defects.</p>
      </body>
    </html>
    """
    doc = parser.parse(html_content, filename="warranty.html")
    assert doc.file_type == "html"
    assert "Hardware Warranty" in doc.sections[0].text
    assert "1-year limited coverage" in doc.sections[0].text
    assert "alert" not in doc.sections[0].text


def test_parse_csv():
    """Verify CSV parsing formats rows with header context."""
    parser = DocumentParser()
    csv_content = b"sku,product_name,price\nPROD-01,Webcam,49.99\nPROD-02,Headset,79.99"
    doc = parser.parse(csv_content, filename="catalog.csv")
    assert doc.file_type == "csv"
    assert len(doc.sections) == 2
    assert "sku: PROD-01" in doc.sections[0].text
    assert "product_name: Webcam" in doc.sections[0].text
    assert doc.sections[0].metadata["row_index"] == 1


def test_parse_docx():
    """Verify DOCX parsing extracts paragraphs and tables."""
    import docx
    doc_io = io.BytesIO()
    doc_obj = docx.Document()
    doc_obj.add_paragraph("Shipping takes 3-5 business days.")
    table = doc_obj.add_table(rows=1, cols=2)
    table.rows[0].cells[0].text = "Standard"
    table.rows[0].cells[1].text = "$4.99"
    doc_obj.save(doc_io)
    doc_io.seek(0)

    parser = DocumentParser()
    parsed = parser.parse(doc_io.getvalue(), filename="shipping.docx")
    assert parsed.file_type == "docx"
    assert "3-5 business days" in parsed.sections[0].text
    assert "Standard | $4.99" in parsed.sections[0].text


def test_parse_pdf():
    """Verify PDF parsing extracts text and preserves page metadata."""
    import pypdf
    writer = pypdf.PdfWriter()
    page = writer.add_blank_page(width=72, height=72)
    # Using pypdf's annotation/text features or writer
    pdf_io = io.BytesIO()
    writer.write(pdf_io)
    pdf_io.seek(0)

    parser = DocumentParser()
    parsed = parser.parse(pdf_io.getvalue(), filename="document.pdf")
    assert parsed.file_type == "pdf"
    assert len(parsed.sections) >= 1
    assert "page_number" in parsed.sections[0].metadata


def test_unsupported_file_format():
    """Verify parser rejects unsupported formats cleanly."""
    parser = DocumentParser()
    with pytest.raises(ValueError, match="Unsupported file type"):
        parser.parse(b"binary", filename="script.exe")


# ==============================================================================
# 2. Document Chunking & Metadata Preservation
# ==============================================================================

def test_document_chunker():
    """Verify chunker creates DocumentChunk instances with preserved metadata."""
    parser = DocumentParser()
    text = (
        "Section 1: Warranty covers 1 year.\n\n"
        "Section 2: Refunds allowed within 30 days.\n\n"
        "Section 3: Shipping standard takes 3-5 days."
    )
    doc = parser.parse(text.encode("utf-8"), filename="summary.txt")
    chunker = DocumentChunker(chunk_size=100, chunk_overlap=20)
    chunks = chunker.split_document(doc)

    assert len(chunks) >= 2
    for chunk in chunks:
        assert chunk.document_id == doc.document_id
        assert chunk.filename == "summary.txt"
        assert chunk.file_type == "txt"
        assert chunk.source.startswith("summary.txt#chunk_")
        assert len(chunk.text) > 0


# ==============================================================================
# 3. ChromaDB Persistence & Reopening
# ==============================================================================

def test_chroma_persistence_and_reopen(temp_chroma_dir):
    """Verify that indexed chunks persist on disk and survive reopening the vector store."""
    store1 = ChromaVectorStore(persist_dir=temp_chroma_dir, collection_name="test_persist")
    parser = DocumentParser()
    chunker = DocumentChunker()

    doc = parser.parse(b"Persistent knowledge chunk for verification.", filename="persist.txt")
    chunks = chunker.split_document(doc)
    store1.add_chunks(chunks)
    assert store1.count() == len(chunks)

    # Simulate application restart: instantiate a new ChromaVectorStore pointing to the same directory
    store2 = ChromaVectorStore(persist_dir=temp_chroma_dir, collection_name="test_persist")
    assert store2.count() == len(chunks)

    # Query reopened store
    results = store2.query("verification", top_k=1)
    assert len(results) == 1
    assert "Persistent knowledge" in results[0].text
    assert results[0].filename == "persist.txt"


# ==============================================================================
# 4. Sample Knowledge Base Ingestion & Relevant Evidence Retrieval
# ==============================================================================

def test_sample_knowledge_base_retrieval(temp_chroma_dir):
    """Verify retrieval accuracy against the realistic sample_data policy files."""
    store = ChromaVectorStore(persist_dir=temp_chroma_dir, collection_name="sample_kb")
    service = RAGService(vector_store=store)

    sample_dir = os.path.join(os.path.dirname(__file__), "..", "..", "sample_data")
    assert os.path.exists(sample_dir), f"Sample directory missing: {sample_dir}"

    ingest_results = service.ingest_directory(sample_dir)
    assert len(ingest_results) >= 4
    assert all(r.status == "success" for r in ingest_results)
    assert service.count() >= 4

    # Query 1: Warranty Duration
    warranty_chunks = service.retrieve("What is the warranty coverage period for hardware?", top_k=2)
    assert len(warranty_chunks) > 0
    top_w = warranty_chunks[0]
    assert "warranty_policy.txt" in top_w.filename
    assert "1-year limited warranty" in top_w.text

    # Query 2: Refund Policy
    refund_chunks = service.retrieve("Can I request a refund and what is the return window?", top_k=2)
    assert len(refund_chunks) > 0
    top_r = refund_chunks[0]
    assert "refund_policy.txt" in top_r.filename
    assert "30 days" in top_r.text

    # Query 3: Shipping Policy
    shipping_chunks = service.retrieve("How many days does standard domestic shipping take?", top_k=2)
    assert len(shipping_chunks) > 0
    top_s = shipping_chunks[0]
    assert "shipping_policy.txt" in top_s.filename
    assert "3 to 5 business days" in top_s.text

    # Query 4: Product Catalog
    catalog_chunks = service.retrieve("UltraSound Pro Wireless Headset price and specifications", top_k=2)
    assert len(catalog_chunks) > 0
    top_c = catalog_chunks[0]
    assert "product_catalog.csv" in top_c.filename
    assert "PROD-101" in top_c.text
    assert "99.99" in top_c.text


# ==============================================================================
# 5. API Endpoints: Document Upload & Independent Retrieval
# ==============================================================================

@pytest.mark.asyncio
async def test_upload_documents_api_valid_and_invalid():
    """Verify POST /api/v1/upload-documents ingests valid files and reports invalid ones."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # Prepare valid and invalid files
        files = [
            ("files", ("test_policy.txt", b"Company return policy is 30 days.", "text/plain")),
            ("files", ("bad_script.exe", b"binary content", "application/octet-stream")),
        ]
        response = await client.post("/api/v1/upload-documents", files=files)
        assert response.status_code == 201
        data = response.json()

        assert data["total_files"] == 2
        assert data["successful"] == 1
        assert data["failed"] == 1
        assert len(data["documents"]) == 1
        assert data["documents"][0]["filename"] == "test_policy.txt"
        assert len(data["errors"]) == 1
        assert "unsupported file format" in data["errors"][0].lower()


@pytest.mark.asyncio
async def test_retrieve_api_endpoint():
    """Verify POST /api/v1/retrieve returns structured chunks for testing."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # First upload a policy document
        files = [("files", ("api_shipping.txt", b"Standard ground shipping is 3-5 days.", "text/plain"))]
        await client.post("/api/v1/upload-documents", files=files)

        # Retrieve
        req_payload = {"query": "How many days for shipping?", "top_k": 2}
        response = await client.post("/api/v1/retrieve", json=req_payload)
        assert response.status_code == 200
        data = response.json()

        assert data["query"] == "How many days for shipping?"
        assert data["total_retrieved"] >= 1
        assert "3-5 days" in data["chunks"][0]["text"]
        assert data["chunks"][0]["filename"] == "api_shipping.txt"


@pytest.mark.asyncio
async def test_documents_stats_api():
    """Verify GET /api/v1/documents/stats returns collection metadata."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/api/v1/documents/stats")
        assert response.status_code == 200
        data = response.json()
        assert "collection_name" in data
        assert "total_chunks" in data
