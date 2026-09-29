"""Multi-format document parser supporting TXT, PDF, DOCX, CSV, and HTML."""

import csv
import io
import os
import uuid
from typing import BinaryIO, Dict, List, Optional, Union
from pydantic import BaseModel, Field


class ParsedSection(BaseModel):
    """A logical segment of a parsed document (e.g., page, section, or tabular row)."""
    text: str
    metadata: Dict[str, Union[str, int, float, bool]] = Field(default_factory=dict)


class ParsedDocument(BaseModel):
    """Full parsed representation of an ingested document."""
    document_id: str
    filename: str
    file_type: str
    sections: List[ParsedSection]


class DocumentParser:
    """Parser capable of reading and extracting clean text from multiple file types."""

    SUPPORTED_EXTENSIONS = {".txt", ".pdf", ".docx", ".csv", ".html", ".htm"}

    @classmethod
    def get_file_extension(cls, filename: str) -> str:
        """Extract and lowercase file extension."""
        return os.path.splitext(filename)[1].lower()

    @classmethod
    def is_supported(cls, filename: str) -> bool:
        """Check if file extension is supported."""
        return cls.get_file_extension(filename) in cls.SUPPORTED_EXTENSIONS

    def parse(
        self,
        file_input: Union[str, bytes, BinaryIO],
        filename: str,
        document_id: Optional[str] = None,
    ) -> ParsedDocument:
        """Parse document content into structured sections based on file type."""
        ext = self.get_file_extension(filename)
        if ext not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported file type '{ext}' for file '{filename}'. "
                f"Supported types: {', '.join(sorted(self.SUPPORTED_EXTENSIONS))}"
            )

        doc_id = document_id or f"doc_{uuid.uuid4().hex[:12]}"
        
        # Load content into bytes
        if isinstance(file_input, str):
            with open(file_input, "rb") as f:
                content_bytes = f.read()
        elif isinstance(file_input, bytes):
            content_bytes = file_input
        else:
            content_bytes = file_input.read()

        if ext in (".txt", ".htm", ".html"):
            sections = self._parse_text_or_html(content_bytes, ext)
        elif ext == ".pdf":
            sections = self._parse_pdf(content_bytes)
        elif ext == ".docx":
            sections = self._parse_docx(content_bytes)
        elif ext == ".csv":
            sections = self._parse_csv(content_bytes)
        else:
            sections = []

        return ParsedDocument(
            document_id=doc_id,
            filename=filename,
            file_type=ext.lstrip("."),
            sections=sections,
        )

    def _parse_text_or_html(self, content_bytes: bytes, ext: str) -> List[ParsedSection]:
        """Parse TXT and HTML files."""
        # Try UTF-8 decoding, fallback to latin-1
        try:
            text_content = content_bytes.decode("utf-8")
        except UnicodeDecodeError:
            text_content = content_bytes.decode("latin-1", errors="replace")

        if ext in (".html", ".htm"):
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(text_content, "html.parser")
            # Remove script and style elements
            for element in soup(["script", "style", "nav", "header", "footer"]):
                element.extract()
            text_content = soup.get_text(separator="\n", strip=True)

        # Clean multiple blank lines
        clean_text = "\n".join(line.strip() for line in text_content.splitlines() if line.strip())
        return [ParsedSection(text=clean_text, metadata={"section_type": "full_text"})]

    def _parse_pdf(self, content_bytes: bytes) -> List[ParsedSection]:
        """Parse PDF document with page number tracking."""
        import pypdf
        reader = pypdf.PdfReader(io.BytesIO(content_bytes))
        sections = []

        for page_idx, page in enumerate(reader.pages):
            page_text = page.extract_text() or ""
            clean_page_text = "\n".join(line.strip() for line in page_text.splitlines() if line.strip())
            if clean_page_text:
                sections.append(
                    ParsedSection(
                        text=clean_page_text,
                        metadata={"page_number": page_idx + 1},
                    )
                )

        if not sections:
            sections.append(ParsedSection(text="[Empty PDF document]", metadata={"page_number": 1}))

        return sections

    def _parse_docx(self, content_bytes: bytes) -> List[ParsedSection]:
        """Parse DOCX document paragraphs and tables."""
        import docx
        doc = docx.Document(io.BytesIO(content_bytes))
        parts = []

        # Extract paragraphs
        for para in doc.paragraphs:
            text = para.text.strip()
            if text:
                parts.append(text)

        # Extract tables
        for table in doc.tables:
            for row in table.rows:
                cells = [c.text.strip() for c in row.cells if c.text.strip()]
                if cells:
                    parts.append(" | ".join(cells))

        full_text = "\n\n".join(parts)
        return [ParsedSection(text=full_text, metadata={"section_type": "body"})]

    def _parse_csv(self, content_bytes: bytes) -> List[ParsedSection]:
        """Parse CSV rows into contextual self-contained record statements."""
        try:
            text_content = content_bytes.decode("utf-8")
        except UnicodeDecodeError:
            text_content = content_bytes.decode("latin-1", errors="replace")

        reader = csv.DictReader(io.StringIO(text_content))
        sections = []

        for row_idx, row in enumerate(reader, start=1):
            # Format row nicely: "Column: Value | Column: Value"
            items = [f"{col}: {val.strip()}" for col, val in row.items() if val and val.strip()]
            if items:
                row_text = " | ".join(items)
                sections.append(
                    ParsedSection(
                        text=row_text,
                        metadata={"row_index": row_idx},
                    )
                )

        return sections
