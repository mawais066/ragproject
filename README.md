# 🤖 Full-Stack RAG PDF Question-Answering AI

A beginner-friendly, production-ready full-stack **RAG (Retrieval-Augmented Generation)** application built with **FastAPI**, **LangChain**, **FAISS**, and a modern **Vanilla HTML/CSS/JS** frontend.

Upload any PDF document, and the system extracts text, splits it into semantic chunks, generates vector embeddings in FAISS, and answers questions strictly based on the content of the PDF.

---

## 🌟 Key Features

- 📄 **PDF Text Extraction & Parsing**: Powered by `pypdf` with page tracking and validation for empty/encrypted files.
- ✂️ **Semantic Chunking**: Configurable `RecursiveCharacterTextSplitter` (1000 chars, 200 overlap).
- ⚡ **Vector Search (FAISS)**: Fast similarity search to retrieve top-k relevant document chunks.
- 🔒 **Strict Grounded QA**: Enforces system instructions to answer *only* from the uploaded PDF without hallucinations.
- 🔎 **Source Citations**: Displays page numbers and extracted chunk snippets with every answer.
- 🌐 **Multi-Provider LLM Support**: Works seamlessly with **OpenAI**, **Groq** (Fast & Free Tier), **OpenRouter**, **DeepSeek**, and local **Ollama**.
- 🎨 **Modern Glassmorphic UI**: Responsive design with drag-and-drop uploads, real-time statistics, loading indicators, and markdown formatting.
- 🚀 **Zero-Config Serving**: FastAPI serves both the REST API and the Frontend together.

---

## 🏗️ Project Architecture & Pipeline

```
[1. User Uploads PDF] ──► [2. Text Extraction (PyPDF)]
                                    │
                                    ▼
                          [3. Recursive Text Splitter]
                                    │
                                    ▼
                          [4. Vector Embeddings]
                                    │
                                    ▼
                          [5. FAISS Vector Database]
                                    │
[User Asks Question]  ──► [6. Top-K Similarity Search]
                                    │
                                    ▼
                          [7. Augmented Strict Prompt]
                                    │
                                    ▼
                          [8. OpenAI-Compatible LLM]
                                    │
                                    ▼
                          [9. Answer + Source Citations in UI]
```

---

## 📁 Project Structure

```
rag-project/
├── backend/
│   ├── main.py              # FastAPI server, REST routes & static mounting
│   ├── rag.py               # Vector store (FAISS), retrieval & LLM chain
│   ├── pdf_processor.py     # PDF validation, text extraction & chunking
│   ├── requirements.txt     # Python package dependencies
│   ├── .env.example         # Environment variable template
│   └── .env                 # (Your local configuration - keep private)
├── frontend/
│   ├── index.html           # Modern chat & upload interface
│   ├── style.css            # Dark glassmorphic design system
│   └── script.js            # Frontend logic, API calls & state management
├── run.py                   # Single-command launcher script
├── .env.example             # Root configuration template
└── README.md                # Documentation & beginner guide
```

---

## 🚀 Quick Start Guide (Step-by-Step)

### 1. Prerequisites
- **Python 3.9+** (Python 3.10, 3.11, 3.12 recommended).
- Internet connection (or a locally running Ollama instance).

---

### 2. Install Dependencies

Open your terminal in the project root directory and run:

```bash
# Optional: Create and activate a virtual environment
python -m venv venv

# On Windows (PowerShell):
venv\Scripts\Activate.ps1
# On macOS / Linux:
# source venv/bin/activate

# Install required Python packages
pip install -r backend/requirements.txt
```

---

### 3. Configure Environment Variables (`.env`)

Create a `.env` file in the `backend/` folder (or copy `.env.example`):

```bash
# Copy example file
cp backend/.env.example backend/.env
```

Open `backend/.env` in your code editor and configure your provider:

#### Option A: OpenAI (Default)
```env
LLM_API_KEY=sk-proj-your-openai-api-key-here
LLM_BASE_URL=https://api.openai.com/v1
LLM_MODEL=gpt-4o-mini
EMBEDDING_MODEL=text-embedding-3-small
TOP_K=4
```

#### Option B: Groq (Ultra-fast & Free Tier Available)
1. Get a free API key at [https://console.groq.com/keys](https://console.groq.com/keys).
2. Set your `.env`:
```env
LLM_API_KEY=gsk_your_groq_api_key_here
LLM_BASE_URL=https://api.groq.com/openai/v1
LLM_MODEL=llama-3.3-70b-versatile
EMBEDDING_MODEL=text-embedding-3-small
TOP_K=4
```
*(Note: If using OpenAI embeddings with Groq chat model, keep an OpenAI key for embeddings or use OpenAI compatible embeddings endpoint).*

#### Option C: OpenRouter
```env
LLM_API_KEY=sk-or-v1-your-openrouter-key-here
LLM_BASE_URL=https://openrouter.ai/api/v1
LLM_MODEL=meta-llama/llama-3.1-8b-instruct:free
EMBEDDING_MODEL=text-embedding-3-small
TOP_K=4
```

#### Option D: Local Ollama (100% Offline & Free)
```env
LLM_API_KEY=ollama
LLM_BASE_URL=http://localhost:11434/v1
LLM_MODEL=llama3:latest
EMBEDDING_MODEL=nomic-embed-text
TOP_K=4
```

---

### 4. Run the Application

Start the server using the single launcher script:

```bash
python run.py
```

Or run via Uvicorn directly:

```bash
cd backend
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

---

### 5. Open and Use the App

1. Open your web browser and navigate to:
   ```
   http://127.0.0.1:8000
   ```
2. **Upload a PDF**: Drag and drop your `.pdf` file into the upload zone or click to select one.
3. **Wait for Processing**: The app extracts the text, builds chunks, and creates the FAISS vector database.
4. **Ask Questions**: Type any question related to the uploaded document in the chat box and press `Enter` or click the Send button.
5. **Inspect Sources**: Click **"Retrieved Sources"** below any answer to see the exact page numbers and text snippets used.
6. **Upload Another PDF**: Click the **"Reset"** button on the active document card to clear and upload a new PDF.

---

## 📡 API Endpoints Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Serves the interactive Frontend UI |
| `GET` | `/api/status` | Returns system config & active document status |
| `POST` | `/upload-pdf` | Uploads, chunks, and indexes a PDF into FAISS |
| `POST` | `/ask` | Queries the vector store & generates a strict answer |
| `POST` | `/reset` | Clears active vector store and document data |
| `GET` | `/docs` | Interactive Swagger / OpenAPI documentation |

---

## 🛠️ Troubleshooting & FAQs

### 1. "LLM_API_KEY is not configured"
- **Fix:** Ensure you created `backend/.env` and replaced `your_api_key_here` with your actual API key. Restart the server after editing `.env`.

### 2. "Authentication Error / 401 Invalid API Key"
- **Fix:** Check that your API key does not have extra spaces or quotation marks. Verify key active status on your provider's dashboard.

### 3. "No readable text could be extracted from PDF"
- **Fix:** The PDF may be a scanned document containing only images. Ensure your PDF has a selectable text layer, or run OCR before uploading.

### 4. "The information is not available in the uploaded PDF"
- **Explanation:** The system operates under strict grounding rules. If the question asks about something not present in the PDF chunks, the assistant will decline to guess or hallucinate.

---

## 📜 License
MIT License. Free for educational and personal use!
