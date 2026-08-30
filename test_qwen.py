import sys
from pathlib import Path

backend_path = Path('.').resolve() / 'backend'
sys.path.insert(0, str(backend_path))

from pdf_processor import extract_and_chunk_pdf
from rag import rag_service

def test_qwen_rag():
    pdf_path = Path('.').resolve() / 'sample_documents' / 'Complete_AI_Summary.pdf'
    with open(pdf_path, 'rb') as f:
        pdf_bytes = f.read()

    print("1. Extracting and chunking PDF...")
    chunks, stats = extract_and_chunk_pdf(pdf_bytes, 'Complete_AI_Summary.pdf')
    print(f"   Generated {len(chunks)} chunks across {stats['total_pages']} pages.")

    print("2. Indexing into Vector Store with 'futur/Qwen3-Embedding-0.6B-model2vec-onnx'...")
    rag_service.index_documents(chunks, stats)
    print("   Qwen3 Embedding Index built successfully!")

    print("3. Querying Qwen (Qwen/Qwen2.5-72B-Instruct)...")
    question = "What are the common tasks of Natural Language Processing (NLP) listed in the PDF?"
    res = rag_service.query(question)

    print("\n================ ANSWER FROM QWEN ================")
    print(res['answer'])
    print("==================================================\n")

    print("Sources retrieved:")
    for s in res['sources']:
        print(f" - Page {s['page']} (Chunk {s['chunk_id']})")

if __name__ == '__main__':
    test_qwen_rag()
