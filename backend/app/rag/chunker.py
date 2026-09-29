"""Document chunking module producing context-rich chunks with metadata."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from backend.app.rag.parser import ParsedDocument


class DocumentChunk(BaseModel):
    """A granular piece of text ready for vector embedding and retrieval."""
    chunk_id: str
    text: str
    document_id: str
    filename: str
    file_type: str
    chunk_index: int
    source: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class DocumentChunker:
    """Splits parsed documents into overlapping chunks while preserving metadata."""

    def __init__(
        self,
        chunk_size: int = 600,
        chunk_overlap: int = 100,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_document(self, parsed_doc: ParsedDocument) -> List[DocumentChunk]:
        """Convert a ParsedDocument into a list of DocumentChunks."""
        chunks: List[DocumentChunk] = []
        global_chunk_idx = 0

        for section in parsed_doc.sections:
            text = section.text.strip()
            if not text:
                continue

            # If section text fits in one chunk, keep it intact
            if len(text) <= self.chunk_size:
                chunk_id = f"{parsed_doc.document_id}#chunk_{global_chunk_idx}"
                meta = dict(section.metadata)
                meta.update({
                    "document_id": parsed_doc.document_id,
                    "filename": parsed_doc.filename,
                    "file_type": parsed_doc.file_type,
                    "chunk_index": global_chunk_idx,
                    "source": f"{parsed_doc.filename}#chunk_{global_chunk_idx}",
                })
                chunks.append(
                    DocumentChunk(
                        chunk_id=chunk_id,
                        text=text,
                        document_id=parsed_doc.document_id,
                        filename=parsed_doc.filename,
                        file_type=parsed_doc.file_type,
                        chunk_index=global_chunk_idx,
                        source=f"{parsed_doc.filename}#chunk_{global_chunk_idx}",
                        metadata=meta,
                    )
                )
                global_chunk_idx += 1
            else:
                # Split large text into overlapping windows
                split_texts = self._split_text(text)
                for split_text in split_texts:
                    chunk_id = f"{parsed_doc.document_id}#chunk_{global_chunk_idx}"
                    meta = dict(section.metadata)
                    meta.update({
                        "document_id": parsed_doc.document_id,
                        "filename": parsed_doc.filename,
                        "file_type": parsed_doc.file_type,
                        "chunk_index": global_chunk_idx,
                        "source": f"{parsed_doc.filename}#chunk_{global_chunk_idx}",
                    })
                    chunks.append(
                        DocumentChunk(
                            chunk_id=chunk_id,
                            text=split_text,
                            document_id=parsed_doc.document_id,
                            filename=parsed_doc.filename,
                            file_type=parsed_doc.file_type,
                            chunk_index=global_chunk_idx,
                            source=f"{parsed_doc.filename}#chunk_{global_chunk_idx}",
                            metadata=meta,
                        )
                    )
                    global_chunk_idx += 1

        return chunks

    def _split_text(self, text: str) -> List[str]:
        """Split text into segments of approximate chunk_size with overlap."""
        paragraphs = text.split("\n\n")
        chunks = []
        current_chunk = ""

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            if len(current_chunk) + len(para) + 2 <= self.chunk_size:
                current_chunk = f"{current_chunk}\n\n{para}".strip()
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                    # Keep overlap from end of current_chunk
                    overlap_start = max(0, len(current_chunk) - self.chunk_overlap)
                    overlap_text = current_chunk[overlap_start:].strip()
                    current_chunk = f"{overlap_text}\n\n{para}".strip()
                else:
                    # Paragraph itself is larger than chunk_size; slice by characters
                    for i in range(0, len(para), self.chunk_size - self.chunk_overlap):
                        slice_chunk = para[i : i + self.chunk_size].strip()
                        if slice_chunk:
                            chunks.append(slice_chunk)
                    current_chunk = ""

        if current_chunk:
            chunks.append(current_chunk)

        return chunks or [text]
