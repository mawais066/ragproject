"""
Unit and Integration Tests for PDF RAG Application
Tests PDF processing, chunking, API routes, and error handling.
"""

import os
import sys
from pathlib import Path

# Add backend directory to sys.path
backend_path = Path(__file__).resolve().parent / "backend"
if str(backend_path) not in sys.path:
    sys.path.insert(0, str(backend_path))

from pdf_processor import extract_and_chunk_pdf, PDFProcessingError
from fastapi.testclient import TestClient
from main import app

def test_pdf_extraction_and_chunking():
    print("[1/4] Testing PDF Extraction & Semantic Chunking...")
    sample_pdf_path = Path(__file__).resolve().parent / "sample_documents" / "Complete_AI_Summary.pdf"
    assert sample_pdf_path.exists(), "Sample PDF not found!"

    with open(sample_pdf_path, "rb") as f:
        pdf_bytes = f.read()

    chunks, stats = extract_and_chunk_pdf(
        file_bytes=pdf_bytes,
        filename="Complete_AI_Summary.pdf",
        chunk_size=1000,
        chunk_overlap=200
    )

    print(f"      Extracted {len(chunks)} chunks across {stats['total_pages']} pages ({stats['total_characters']} characters).")
    assert len(chunks) > 0, "No chunks generated!"
    assert stats["total_pages"] == 10, f"Expected 10 pages, got {stats['total_pages']}"
    assert "source" in chunks[0].metadata
    assert "page" in chunks[0].metadata
    print("      [PASSED] Extraction & Chunking verified.")

def test_pdf_processor_error_cases():
    print("[2/4] Testing Error Handling for Invalid/Corrupt PDFs...")
    # Empty bytes
    try:
        extract_and_chunk_pdf(b"", "empty.pdf")
        assert False, "Should have failed on empty bytes"
    except PDFProcessingError as e:
        print(f"      [PASSED] Empty PDF caught: {e}")

    # Non-PDF file
    try:
        extract_and_chunk_pdf(b"This is just a plain text file, not a PDF", "fake.pdf")
        assert False, "Should have failed on invalid magic bytes"
    except PDFProcessingError as e:
        print(f"      [PASSED] Invalid PDF header caught: {e}")

def test_fastapi_endpoints():
    print("[3/4] Testing FastAPI Endpoints...")
    client = TestClient(app)

    # Test GET /
    res = client.get("/")
    assert res.status_code == 200, f"Root / failed: {res.status_code}"
    assert "PDF RAG Assistant" in res.text or "<!DOCTYPE html>" in res.text
    print("      [PASSED] GET / returned HTML frontend.")

    # Test GET /api/status
    res_status = client.get("/api/status")
    assert res_status.status_code == 200
    status_data = res_status.json()
    assert "has_api_key" in status_data
    assert "is_document_loaded" in status_data
    print(f"      [PASSED] GET /api/status returned valid schema (is_document_loaded={status_data['is_document_loaded']}).")

    # Test POST /ask before uploading PDF
    res_ask_early = client.post("/ask", json={"question": "What is AI?"})
    assert res_ask_early.status_code == 400
    assert "No PDF has been uploaded" in res_ask_early.json()["detail"]
    print("      [PASSED] POST /ask before PDF upload correctly rejected with 400.")

    # Test POST /reset
    res_reset = client.post("/reset")
    assert res_reset.status_code == 200
    print("      [PASSED] POST /reset returned 200.")

def test_static_files():
    print("[4/4] Testing Static Assets...")
    client = TestClient(app)
    res_css = client.get("/static/style.css")
    assert res_css.status_code == 200
    assert "--primary" in res_css.text

    res_js = client.get("/static/script.js")
    assert res_js.status_code == 200
    assert "handleFileUpload" in res_js.text
    print("      [PASSED] CSS & JS static files served correctly.")

if __name__ == "__main__":
    print("=" * 60)
    print(" Running Full Test Suite for RAG PDF System")
    print("=" * 60)
    test_pdf_extraction_and_chunking()
    test_pdf_processor_error_cases()
    test_fastapi_endpoints()
    test_static_files()
    print("=" * 60)
    print(" ALL TESTS PASSED SUCCESSFULLY! ")
    print("=" * 60)
