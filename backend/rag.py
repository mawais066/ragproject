"""
RAG (Retrieval-Augmented Generation) Pipeline Module
Manages Vector Store (FAISS), Embeddings, Retrieval, and LLM Q&A.
"""

import os
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.embeddings import Embeddings

# Load environment variables
load_dotenv()


class RAGError(Exception):
    """Custom exception for RAG pipeline errors."""
    pass


class Qwen3EmbeddingModel(Embeddings):
    """
    Lightning-fast embedding model for 'futur/Qwen3-Embedding-0.6B-model2vec-onnx'
    and similar static token embedding models.
    """
    def __init__(self, repo_id: str = "futur/Qwen3-Embedding-0.6B-model2vec-onnx", token: Optional[str] = None):
        from huggingface_hub import hf_hub_download
        from safetensors import safe_open
        from tokenizers import Tokenizer
        
        self.repo_id = repo_id
        tok_path = hf_hub_download(repo_id=repo_id, filename="tokenizer.json", token=token)
        weights_path = hf_hub_download(repo_id=repo_id, filename="model.safetensors", token=token)
        
        self.tokenizer = Tokenizer.from_file(tok_path)
        with safe_open(weights_path, framework="numpy") as f:
            self.embeddings_table = f.get_tensor("embeddings")
        self.vocab_size = self.embeddings_table.shape[0]
        self.dim = self.embeddings_table.shape[1]

    def embed_query(self, text: str) -> List[float]:
        encoded = self.tokenizer.encode(text)
        valid_ids = [i for i in encoded.ids if i < self.vocab_size]
        if not valid_ids:
            return np.zeros((self.dim,), dtype=np.float32).tolist()
        
        vec = np.mean(self.embeddings_table[valid_ids], axis=0)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec.tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self.embed_query(t) for t in texts]


class HuggingFaceInferenceEmbeddings(Embeddings):
    """Hugging Face API Embeddings client using huggingface_hub InferenceClient."""
    def __init__(self, api_key: str, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        from huggingface_hub import InferenceClient
        self.client = InferenceClient(token=api_key)
        self.model_name = model_name

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = []
        for text in texts:
            try:
                vec = self.client.feature_extraction(text, model=self.model_name)
                # If nested array or tensor
                if hasattr(vec, "tolist"):
                    vec = vec.tolist()
                # If 2D array, take first or mean
                if isinstance(vec, list) and len(vec) > 0 and isinstance(vec[0], list):
                    vec = vec[0]
                embeddings.append(vec)
            except Exception as e:
                raise RAGError(f"Hugging Face embedding extraction failed: {str(e)}")
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        vec = self.client.feature_extraction(text, model=self.model_name)
        if hasattr(vec, "tolist"):
            vec = vec.tolist()
        if isinstance(vec, list) and len(vec) > 0 and isinstance(vec[0], list):
            vec = vec[0]
        return vec


class ResilientVectorStore:
    """
    High-performance in-memory Vector Store using FAISS with automatic 
    pure-NumPy cosine similarity fallback for maximum Windows compatibility.
    """
    def __init__(self, documents: List[Document], embeddings_model: Embeddings):
        self.documents = documents
        self.embeddings_model = embeddings_model
        self.doc_embeddings: Optional[np.ndarray] = None
        self.use_faiss = False
        self.faiss_store = None

        # Extract texts
        texts = [doc.page_content for doc in documents]
        raw_embeds = self.embeddings_model.embed_documents(texts)
        self.doc_embeddings = np.array(raw_embeds, dtype=np.float32)

        # Normalize document vectors for fast cosine similarity
        norms = np.linalg.norm(self.doc_embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.normed_doc_embeddings = self.doc_embeddings / norms

        # Try building FAISS if available and permitted
        try:
            from langchain_community.vectorstores import FAISS
            self.faiss_store = FAISS.from_documents(documents, embeddings_model)
            self.use_faiss = True
        except Exception:
            self.use_faiss = False

    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        if self.use_faiss and self.faiss_store is not None:
            try:
                return self.faiss_store.similarity_search(query, k=k)
            except Exception:
                pass

        # High-performance NumPy Cosine Similarity search
        query_vec = np.array(self.embeddings_model.embed_query(query), dtype=np.float32)
        q_norm = np.linalg.norm(query_vec)
        if q_norm == 0:
            q_norm = 1.0
        query_vec_normed = query_vec / q_norm

        # Compute cosine similarity across all chunk vectors: (N, D) dot (D,) -> (N,)
        scores = np.dot(self.normed_doc_embeddings, query_vec_normed)

        # Get top-k indices
        top_k_indices = np.argsort(scores)[::-1][:k]

        return [self.documents[idx] for idx in top_k_indices]


class RAGPipeline:
    """
    Manages active PDF document state, vector embeddings, and RAG answering.
    Supports OpenAI, Groq, OpenRouter, and Hugging Face (Qwen).
    """

    STRICT_SYSTEM_PROMPT = (
        "You are a PDF question-answering assistant. "
        "Answer the user's question only using the provided context retrieved from the uploaded PDF. "
        "Do not invent information. "
        "If the answer cannot be found in the provided PDF context, clearly say that the information is not available in the uploaded PDF."
    )

    def __init__(self):
        self.vector_store: Optional[ResilientVectorStore] = None
        self.active_pdf_stats: Optional[Dict[str, Any]] = None
        self.top_k: int = int(os.getenv("TOP_K", "4"))

    def _is_huggingface(self) -> bool:
        api_key = os.getenv("LLM_API_KEY", "").strip()
        model = os.getenv("LLM_MODEL", "").strip()
        return api_key.startswith("hf_") or "huggingface" in model.lower() or "/" in model

    def get_config(self) -> Dict[str, str]:
        """Returns safe configuration info (excluding private keys)."""
        api_key = os.getenv("LLM_API_KEY", "").strip()
        has_api_key = bool(api_key and api_key != "your_api_key_here")
        return {
            "has_api_key": has_api_key,
            "base_url": os.getenv("LLM_BASE_URL", "https://api.openai.com/v1"),
            "llm_model": os.getenv("LLM_MODEL", "Qwen/Qwen2.5-72B-Instruct" if self._is_huggingface() else "gpt-4o-mini"),
            "embedding_model": os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2" if self._is_huggingface() else "text-embedding-3-small"),
            "top_k": str(self.top_k),
            "is_document_loaded": self.vector_store is not None,
            "loaded_filename": self.active_pdf_stats.get("filename") if self.active_pdf_stats else None,
        }

    def _get_embeddings_model(self) -> Embeddings:
        """Initializes the embeddings client."""
        api_key = os.getenv("LLM_API_KEY", "").strip()
        if not api_key or api_key == "your_api_key_here":
            raise RAGError(
                "LLM_API_KEY is not configured. Please set your API key in the .env file."
            )

        embed_model = os.getenv("EMBEDDING_MODEL", "futur/Qwen3-Embedding-0.6B-model2vec-onnx").strip()

        # 1. Custom fast Qwen static embedding model
        if "futur" in embed_model.lower() or "qwen3-embedding" in embed_model.lower() or "model2vec" in embed_model.lower():
            try:
                return Qwen3EmbeddingModel(repo_id=embed_model, token=api_key)
            except Exception as e:
                # Fallback to HF Inference embeddings if local load fails
                return HuggingFaceInferenceEmbeddings(api_key=api_key, model_name="sentence-transformers/all-MiniLM-L6-v2")

        # 2. Standard Hugging Face Inference embeddings
        if self._is_huggingface():
            try:
                return HuggingFaceInferenceEmbeddings(api_key=api_key, model_name=embed_model)
            except Exception as e:
                raise RAGError(f"Failed to initialize Hugging Face embeddings ({embed_model}): {str(e)}")

        # 3. OpenAI / compatible embeddings
        from langchain_openai import OpenAIEmbeddings
        base_url = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")
        model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

        try:
            return OpenAIEmbeddings(
                model=model,
                openai_api_key=api_key,
                openai_api_base=base_url,
                check_embedding_ctx_length=False,
            )
        except Exception as e:
            raise RAGError(f"Failed to initialize OpenAI embedding model ({model}): {str(e)}")

    def _generate_answer_llm(self, formatted_context: str, question: str) -> str:
        """Invokes the configured LLM (Hugging Face Qwen or OpenAI-compatible)."""
        api_key = os.getenv("LLM_API_KEY", "").strip()
        model_name = os.getenv("LLM_MODEL", "Qwen/Qwen2.5-72B-Instruct")

        prompt = (
            f"Context from the uploaded PDF:\n\"\"\"\n{formatted_context}\n\"\"\"\n\n"
            f"User Question:\n{question}\n\n"
            f"Answer strictly using only the context above:"
        )

        if self._is_huggingface():
            from huggingface_hub import InferenceClient
            try:
                client = InferenceClient(token=api_key)
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": self.STRICT_SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    max_tokens=600,
                    temperature=0.1,
                )
                return response.choices[0].message.content or ""
            except Exception as e:
                err_msg = str(e)
                if "401" in err_msg or "Invalid username or password" in err_msg or "token" in err_msg.lower():
                    raise RAGError("Authentication Error: The provided Hugging Face API key is invalid.")
                else:
                    # Fallback to recommended Qwen 72B
                    try:
                        fallback_client = InferenceClient(token=api_key)
                        res = fallback_client.chat.completions.create(
                            model="Qwen/Qwen2.5-72B-Instruct",
                            messages=[
                                {"role": "system", "content": self.STRICT_SYSTEM_PROMPT},
                                {"role": "user", "content": prompt},
                            ],
                            max_tokens=600,
                            temperature=0.1,
                        )
                        return res.choices[0].message.content or ""
                    except Exception as fe:
                        raise RAGError(f"Hugging Face LLM generation failed: {str(fe)}")

        # OpenAI / Groq / OpenRouter path
        from langchain_openai import ChatOpenAI
        base_url = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")

        try:
            llm = ChatOpenAI(
                model=model_name,
                openai_api_key=api_key,
                openai_api_base=base_url,
                temperature=0.0,
            )
            prompt_template = ChatPromptTemplate.from_messages([
                ("system", self.STRICT_SYSTEM_PROMPT),
                ("human", prompt),
            ])
            chain = prompt_template | llm | StrOutputParser()
            return chain.invoke({})
        except Exception as e:
            err_msg = str(e)
            if "401" in err_msg or "Incorrect API key" in err_msg:
                raise RAGError("Authentication Error: The provided LLM_API_KEY is invalid or expired.")
            elif "429" in err_msg or "quota" in err_msg.lower():
                raise RAGError("API Rate limit or quota exceeded. Please check your provider billing/limits.")
            else:
                raise RAGError(f"LLM generation failed: {err_msg}")

    def index_documents(self, chunks: List[Document], stats: Dict[str, Any]) -> None:
        """
        Generates embeddings for the chunks and builds the vector database.
        """
        if not chunks:
            raise RAGError("No chunks provided to build vector index.")

        embeddings = self._get_embeddings_model()

        try:
            self.vector_store = ResilientVectorStore(chunks, embeddings)
            self.active_pdf_stats = stats
        except Exception as e:
            err_msg = str(e)
            if "401" in err_msg or "Invalid username or password" in err_msg:
                raise RAGError("Authentication Error: The provided API key is invalid.")
            else:
                raise RAGError(f"Failed while generating vector embeddings: {err_msg}")

    def query(self, question: str) -> Dict[str, Any]:
        """
        Retrieves relevant PDF chunks and answers the user question using strict context.
        """
        question = question.strip()
        if not question:
            raise RAGError("Question cannot be empty.")

        if self.vector_store is None:
            raise RAGError(
                "No PDF has been uploaded and indexed yet. Please upload a PDF before asking questions."
            )

        # 1. Retrieve top-k relevant document chunks
        try:
            relevant_docs = self.vector_store.similarity_search(question, k=self.top_k)
        except Exception as e:
            raise RAGError(f"Vector search failed: {str(e)}")

        if not relevant_docs:
            return {
                "answer": "The information is not available in the uploaded PDF.",
                "sources": [],
            }

        # 2. Format context from retrieved chunks
        context_blocks = []
        sources = []
        for doc in relevant_docs:
            page = doc.metadata.get("page", 1)
            chunk_id = doc.metadata.get("chunk_id", "?")
            content = doc.page_content.strip()

            context_blocks.append(f"[Page {page} - Chunk {chunk_id}]\n{content}")
            sources.append({
                "page": page,
                "chunk_id": chunk_id,
                "content": content[:300] + ("..." if len(content) > 300 else ""),
            })

        formatted_context = "\n\n---\n\n".join(context_blocks)

        # 3. Generate Answer
        answer = self._generate_answer_llm(formatted_context, question)

        return {
            "answer": answer.strip(),
            "sources": sources,
            "pdf_name": self.active_pdf_stats.get("filename") if self.active_pdf_stats else "Uploaded PDF",
        }

    def reset(self) -> None:
        """Clears active vector store and stats."""
        self.vector_store = None
        self.active_pdf_stats = None


# Global singleton instance for the application
rag_service = RAGPipeline()
