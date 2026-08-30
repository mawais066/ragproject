"""
PDF Processing Module
Handles PDF validation, text extraction, and text chunking for RAG.
"""

import io
from typing import List, Dict, Any, Tuple
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


class PDFProcessingError(Exception):
    """Custom exception for PDF processing issues."""
    pass


def extract_and_chunk_pdf(
    file_bytes: bytes,
    filename: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200
) -> Tuple[List[Document], Dict[str, Any]]:
    """
    Extracts text from a PDF file in memory and splits it into semantic chunks.

    Args:
        file_bytes: Raw binary content of the PDF file.
        filename: Name of the uploaded file.
        chunk_size: Target character length per chunk.
        chunk_overlap: Number of overlapping characters between chunks.

    Returns:
        Tuple of (list of Document objects, dictionary of metadata/stats)

    Raises:
        PDFProcessingError: If the PDF is invalid, empty, or unreadable.
    """
    if not file_bytes or len(file_bytes) == 0:
        raise PDFProcessingError("The uploaded file is completely empty (0 bytes).")

    # Basic PDF magic header check
    if not file_bytes.startswith(b"%PDF"):
        raise PDFProcessingError(
            f"'{filename}' does not appear to be a valid PDF file. "
            "Please ensure you upload a proper .pdf document."
        )

    try:
        pdf_stream = io.BytesIO(file_bytes)
        reader = PdfReader(pdf_stream)
    except Exception as e:
        raise PDFProcessingError(f"Failed to read PDF structure: {str(e)}")

    if reader.is_encrypted:
        try:
            # Attempt empty password decrypt
            decrypt_result = reader.decrypt("")
            if decrypt_result == 0:
                raise PDFProcessingError(
                    f"'{filename}' is password protected / encrypted. "
                    "Please provide an unencrypted PDF."
                )
        except Exception:
            raise PDFProcessingError(
                f"'{filename}' is password protected / encrypted. "
                "Please provide an unencrypted PDF."
            )

    total_pages = len(reader.pages)
    if total_pages == 0:
        raise PDFProcessingError(f"'{filename}' contains 0 pages.")

    raw_documents: List[Document] = []
    total_characters = 0
    pages_with_text = 0

    for page_idx, page in enumerate(reader.pages):
        page_num = page_idx + 1
        try:
            text = page.extract_text() or ""
            text = text.strip()
        except Exception as e:
            text = ""

        if text:
            pages_with_text += 1
            total_characters += len(text)
            raw_documents.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": filename,
                        "page": page_num,
                        "total_pages": total_pages,
                    }
                )
            )

    if not raw_documents or total_characters < 10:
        raise PDFProcessingError(
            f"No readable text could be extracted from '{filename}'. "
            "This PDF might consist entirely of scanned images without text layer. "
            "Please upload a text-based PDF or run OCR first."
        )

    # Split text into chunks using recursive character splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )

    chunks = text_splitter.split_documents(raw_documents)

    if not chunks:
        raise PDFProcessingError(
            f"Could not generate text chunks from '{filename}'."
        )

    # Attach chunk indexing metadata
    for idx, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = idx + 1
        chunk.metadata["total_chunks"] = len(chunks)

    stats = {
        "filename": filename,
        "total_pages": total_pages,
        "pages_with_text": pages_with_text,
        "total_characters": total_characters,
        "total_chunks": len(chunks),
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
    }

    return chunks, stats
