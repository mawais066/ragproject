"""
FastAPI Application Entrypoint
Serves REST API endpoints for PDF upload, Q&A querying, and serves the Frontend UI.
"""

import os
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from pdf_processor import extract_and_chunk_pdf, PDFProcessingError
from rag import rag_service, RAGError

# Initialize FastAPI App
app = FastAPI(
    title="RAG PDF Question-Answering AI",
    description="Beginner-friendly full-stack RAG application for PDF documents.",
    version="1.0.0",
)

# Enable CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request Models
class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, description="The question to ask about the PDF")


# API Routes

@app.get("/api/status")
async def get_status():
    """
    Returns current configuration state and active PDF status.
    """
    return rag_service.get_config()


@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    """
    Accepts a PDF file upload, extracts text, generates chunks, 
    embeds them into a FAISS vector database, and returns document stats.
    """
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No filename provided with upload."
        )

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"File '{file.filename}' is not a PDF. Please upload a .pdf file."
        )

    try:
        file_bytes = await file.read()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Could not read uploaded file: {str(e)}"
        )

    # 1. Extract text and split into chunks
    try:
        chunks, stats = extract_and_chunk_pdf(
            file_bytes=file_bytes,
            filename=file.filename,
        )
    except PDFProcessingError as pe:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(pe)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected error while extracting PDF text: {str(e)}"
        )

    # 2. Build Vector Embeddings with FAISS
    try:
        rag_service.index_documents(chunks, stats)
    except RAGError as re:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(re)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected error while generating embeddings: {str(e)}"
        )

    return {
        "status": "success",
        "message": f"Successfully processed and indexed '{file.filename}'",
        "stats": stats,
    }


@app.post("/ask")
async def ask_question(request: AskRequest):
    """
    Receives a question, retrieves the most relevant chunks from the indexed PDF,
    and returns a strictly grounded answer from the LLM.
    """
    try:
        response = rag_service.query(request.question)
        return {
            "status": "success",
            "question": request.question,
            "answer": response["answer"],
            "sources": response["sources"],
            "pdf_name": response.get("pdf_name"),
        }
    except RAGError as re:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(re)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while answering your question: {str(e)}"
        )


@app.post("/reset")
async def reset_document():
    """
    Clears the currently indexed document and vector store to allow a fresh upload.
    """
    rag_service.reset()
    return {
        "status": "success",
        "message": "Indexed document has been cleared. You can now upload a new PDF."
    }


# Frontend Static Files Setup
BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"

if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

    @app.get("/style.css")
    async def serve_css():
        return FileResponse(str(FRONTEND_DIR / "style.css"))

    @app.get("/script.js")
    async def serve_js():
        return FileResponse(str(FRONTEND_DIR / "script.js"))

    @app.get("/")
    async def serve_index():
        """Serves the frontend single-page application."""
        index_file = FRONTEND_DIR / "index.html"
        if index_file.exists():
            return FileResponse(str(index_file))
        return JSONResponse({"message": "Frontend index.html not found"}, status_code=404)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
