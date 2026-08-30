"""
Generates the 10-page 'Complete AI Summary' PDF study notes document
for immediate testing with the RAG Question-Answering application.
"""

import os
from pathlib import Path

# PDF Content for each of the 10 pages matching the user study guide
PAGES_CONTENT = [
    # Page 1
    """Complete AI Summary
Beginner to Practical AI — Study Notes

A structured summary covering Artificial Intelligence, Machine Learning, Deep Learning, Generative AI, LLMs, RAG, AI Agents, LangChain, APIs, embeddings, vector databases, deployment, and responsible AI.

Prepared as a beginner-friendly revision guide
Page 1""",

    # Page 2
    """1. Artificial Intelligence (AI)

What is AI?
Artificial Intelligence is the field of building computer systems that can perform tasks that normally require human intelligence, such as understanding language, recognizing patterns, making predictions, planning, and solving problems.

Main areas of AI
Machine Learning, Deep Learning, Natural Language Processing (NLP), Computer Vision, Robotics, Expert Systems, Speech Processing, Generative AI, and AI Agents.

Types of AI
Narrow AI is designed for specific tasks. General AI refers to a hypothetical system with broad human-level intelligence. Superintelligence is a theoretical concept in which an AI would exceed human intelligence across many areas.

2. Machine Learning (ML)

Definition
Machine Learning allows systems to learn patterns from data and use those patterns to make predictions or decisions without being explicitly programmed for every individual case.

Supervised Learning
The model learns from labeled examples. Common tasks are classification and regression. Examples include spam detection, house-price prediction, and image classification.

Unsupervised Learning
The model works with data without target labels. Common tasks include clustering and dimensionality reduction.

Reinforcement Learning
An agent learns through interaction with an environment using rewards and penalties. It is commonly discussed in robotics, games, and sequential decision problems.

Basic workflow
Collect data -> clean data -> split data -> train model -> validate/test -> evaluate -> deploy -> monitor.

3. Deep Learning

Definition
Deep Learning uses multi-layer neural networks to learn complex representations from large datasets.
Page 2""",

    # Page 3
    """Neural network basics
A neural network contains input layers, hidden layers, and output layers. Neurons apply weighted inputs, a bias, and an activation function.

Important architectures
CNNs are widely used for visual data. RNNs process sequences. Transformers use attention mechanisms and are central to modern language models.

Training concepts
Important terms include loss function, optimizer, learning rate, epochs, batch size, backpropagation, overfitting, and regularization.

4. Natural Language Processing (NLP)

Definition
NLP focuses on enabling computers to process and generate human language.

Common tasks
Text classification, sentiment analysis, translation, summarization, question answering, named-entity recognition, information extraction, and text generation.

Tokenization
Text is divided into tokens that a language model can process. A token may be a word, part of a word, punctuation mark, or another unit depending on the tokenizer.

Important idea
Modern NLP systems commonly represent language using learned numerical representations called embeddings.

5. Generative AI

Definition
Generative AI creates new content such as text, images, audio, video, or code based on learned patterns.

Examples
Text generation, image generation, code assistants, document summarization, and conversational systems.

Prompting
A good prompt clearly states the task, context, constraints, expected format, and important input data.

Limitations
Generative models can produce incorrect information, misunderstand ambiguous prompts, or generate plausible but unsupported answers. Outputs should be verified for important tasks.

6. Large Language Models (LLMs)
Page 3""",

    # Page 4
    """What is an LLM?
An LLM is a neural network trained on large amounts of text to model language and generate responses.

Transformer
Transformers use attention to determine which parts of an input are relevant to other parts. This architecture made large-scale language modeling much more effective.

Inference
Inference is the process of using a trained model to produce an output for a new input.

Context window
The context window is the amount of information a model can consider in one interaction. Larger context does not automatically guarantee factual accuracy.

7. Embeddings and Vector Search

Embeddings
An embedding converts text or another object into a numerical vector that captures useful semantic information.

Similarity
Vectors can be compared using measures such as cosine similarity to find semantically related content.

Vector database
A vector database stores embeddings and supports similarity search. It is useful for semantic retrieval.

Examples
Common technologies include FAISS, Chroma, Pinecone, Weaviate, Milvus, and pgvector.

8. RAG — Retrieval-Augmented Generation

Definition
RAG combines retrieval with generation. Instead of relying only on the model's internal knowledge, the system retrieves relevant information from an external knowledge source and provides it to the model.

Pipeline
Documents -> load -> split into chunks -> create embeddings -> store in vector database -> retrieve relevant chunks -> send context to LLM -> generate answer.

Why use RAG?
It is useful for private documents, frequently changing information, company knowledge bases, manuals, PDFs, and domain-specific question answering.

Key issue
Page 4""",

    # Page 5
    """Retrieval quality strongly affects answer quality. Poor chunking, weak embeddings, or irrelevant retrieval can lead to poor responses.

9. AI Agents

Definition
An AI agent is a system that can reason about a task, use tools, observe results, and continue until it reaches an appropriate outcome.

Typical components
LLM + instructions + tools + memory/state + execution loop + optional retrieval.

Tools
Tools may include calculators, search, weather APIs, databases, file readers, or business APIs.

Agent loop
Understand goal -> decide action -> call tool -> inspect result -> decide next step -> produce final response.

Agent vs chatbot
A basic chatbot mainly responds to messages. An agent can take actions and use external tools as part of completing a task.

10. LangChain

Purpose
LangChain is a framework for building applications around language models, including chains, tool use, retrieval, agents, and integrations.

Common concepts
Models, prompts, messages, tools, retrievers, vector stores, parsers, memory/state, and agents.

Typical RAG structure
Document loader -> text splitter -> embeddings -> vector store -> retriever -> prompt -> chat model -> answer.

Good practice
Keep API keys in environment variables, separate configuration from application logic, validate tool inputs, and log useful non-secret information.

11. LangGraph

Definition
Page 5""",

    # Page 6
    """LangGraph is designed for building stateful, multi-step agent workflows as graphs.

Graph concepts
Nodes perform work, edges define transitions, and state carries information between steps.

Why use it?
It is useful when an agent requires predictable workflows, branching, loops, persistence, human approval, or multiple specialized steps.

12. AI APIs

What is an API?
An Application Programming Interface lets one software system communicate with another using defined requests and responses.

LLM API flow
Application sends a request containing model information and messages -> API processes it -> application receives a response.

Security
Never hard-code private API keys into public source code. Store secrets in environment variables or a secure secret manager. Do not expose them in screenshots, Git repositories, logs, or frontend code.

13. Python for AI

Important skills
Variables, functions, classes, lists, dictionaries, tuples, exceptions, file handling, modules, virtual environments, and package management.

Useful libraries
NumPy for numerical computing, pandas for data manipulation, matplotlib for visualization, scikit-learn for classical ML, PyTorch/TensorFlow for deep learning, and LangChain/LangGraph for LLM applications.

Environment
A virtual environment keeps project dependencies isolated. A requirements.txt file can record packages needed to reproduce an environment.

14. Data and Model Evaluation

Data quality
AI performance depends heavily on the quality, relevance, coverage, and correctness of data.

Classification metrics
Page 6""",

    # Page 7
    """Accuracy measures overall correctness; precision measures how many predicted positives are correct; recall measures how many actual positives are found; F1 balances precision and recall.

Regression metrics
MAE measures average absolute error. MSE measures average squared error. RMSE is the square root of MSE.

LLM evaluation
Check factuality, relevance, completeness, instruction following, safety, latency, cost, and consistency. For RAG, also evaluate retrieval quality and whether answers are grounded in retrieved evidence.

15. Fine-Tuning vs RAG

Fine-tuning
Fine-tuning changes model parameters using task-specific training data. It can be useful for behavior, style, or specialized task adaptation.

RAG
RAG supplies external information at inference time. It is often preferable when knowledge changes frequently or comes from private documents.

Simple rule
If the main problem is missing or changing knowledge, consider RAG. If the main problem is behavior or task-specific adaptation, fine-tuning may be useful. Some systems use both.

16. AI Project Architecture

Typical application
Frontend -> backend/API -> AI orchestration -> model provider -> tools/retrieval -> data sources -> response.

Production concerns
Authentication, authorization, secret management, rate limits, validation, monitoring, error handling, caching, cost control, testing, and privacy.

Deployment
A project can be deployed on a cloud platform after adding the correct build/start commands, dependencies, environment variables, and secure configuration.

17. Responsible AI

Core principles
Safety, privacy, transparency, fairness, reliability, accountability, and human oversight.
Page 7""",

    # Page 8
    """Privacy
Do not collect or expose unnecessary personal information. Protect credentials and sensitive data.

Human oversight
High-impact decisions should have appropriate human review and verification rather than blindly trusting model output.

18. Quick Revision — Important Terms

AI: Broad field of intelligent computer systems.
ML: Learning patterns from data.
Deep Learning: Machine learning using multi-layer neural networks.
NLP: Processing and generating human language.
LLM: Large language model for language understanding/generation.
Embedding: Numerical vector representation used for semantic comparison.
RAG: Retrieve external information and use it to generate an answer.
Agent: AI system that can reason and use tools to complete tasks.
API: Interface for communication between software systems.
Vector Database: Database optimized for storing and searching vector representations.

19. Beginner AI Roadmap

Step 1: Learn Python fundamentals and basic Git/GitHub.
Page 8""",

    # Page 9
    """Step 2: Learn NumPy, pandas, data visualization, and basic statistics.
Step 3: Learn machine learning with scikit-learn.
Step 4: Learn neural networks and deep learning concepts.
Step 5: Learn NLP, transformers, prompting, and LLM APIs.
Step 6: Build a RAG PDF/document chatbot.
Step 7: Build tool-using AI agents.
Step 8: Learn LangChain/LangGraph and build structured workflows.
Step 9: Deploy projects and learn monitoring/security.
Step 10: Build a portfolio with several practical projects.
Page 9""",

    # Page 10
    """AI Cheat Sheet

Concept | One-line meaning
AI | Machines performing tasks associated with intelligence
ML | Learning patterns from data
DL | Neural-network-based machine learning
NLP | Working with human language
LLM | Large neural language model
Embedding | Vector representation of information
RAG | Retrieve context before generating an answer
Agent | LLM-based system that can use tools/actions
API | Programmatic interface between systems
Vector DB | Stores and searches embeddings
Prompt | Instructions/context given to an AI model

Note: This PDF is a study/revision guide. AI tools, models, APIs, and frameworks change over time, so current official documentation should be checked when implementing a real project.
Page 10"""
]

def build_pdf():
    """Builds the sample PDF with 10 pages using pure Python/pypdf or reportlab."""
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.pdfgen import canvas
        from reportlab.lib.units import inch

        out_dir = Path(__file__).resolve().parent / "sample_documents"
        out_dir.mkdir(exist_ok=True)
        pdf_path = out_dir / "Complete_AI_Summary.pdf"

        c = canvas.Canvas(str(pdf_path), pagesize=letter)
        width, height = letter

        for page_num, text in enumerate(PAGES_CONTENT, 1):
            c.setFont("Helvetica-Bold", 14)
            lines = text.strip().split("\n")
            
            # Title
            c.drawString(54, height - 54, lines[0])
            
            c.setFont("Helvetica", 10)
            y = height - 80
            for line in lines[1:]:
                if line.strip().startswith(("1.", "2.", "3.", "4.", "5.", "6.", "7.", "8.", "9.", "10.", "11.", "12.", "13.", "14.", "15.", "16.", "17.", "18.", "19.", "AI Cheat Sheet", "Concept")):
                    c.setFont("Helvetica-Bold", 11)
                    y -= 6
                else:
                    c.setFont("Helvetica", 10)

                c.drawString(54, y, line.strip())
                y -= 15
                if y < 50:
                    break

            # Footer
            c.setFont("Helvetica", 8)
            c.drawRightString(width - 54, 30, f"Complete AI Summary • Page {page_num}")
            c.showPage()

        c.save()
        print(f"[OK] Generated sample 10-page PDF at: {pdf_path}")
        return True
    except ImportError:
        # If reportlab is not installed, install it or write via pure PDF generator
        import subprocess
        subprocess.run(["pip", "install", "reportlab"], check=True)
        return build_pdf()

if __name__ == "__main__":
    build_pdf()
