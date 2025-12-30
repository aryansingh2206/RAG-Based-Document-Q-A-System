
# RAG-Based Document Q&A System

An end-to-end **Retrieval-Augmented Generation (RAG)** application that enables users to ask natural-language questions over real-world PDF documents such as **company policies, employee handbooks, and reports**.

The system retrieves relevant document chunks using semantic search and generates **grounded, hallucination-safe answers** using an open-source language model — all with a fully free tech stack.

---

## Key Features

-  Question answering over PDF documents  
-  Semantic search using embeddings + FAISS  
-  Context-grounded answer generation (RAG)  
-  Hallucination reduction via strict context-only prompting  
-  Fast local inference using an open-source LLM  
-  Interactive and explainable Streamlit UI  
-  Fully free & open-source (no paid APIs)

---

##  What is RAG?

**Retrieval-Augmented Generation (RAG)** combines:
- **Information Retrieval** (to fetch relevant context)
- **Text Generation** (to produce fluent answers)

Instead of relying solely on the LLM’s internal knowledge, this system:
1. Retrieves the most relevant document chunks
2. Injects them into the prompt
3. Generates answers strictly from that context

This approach significantly **reduces hallucinations** and allows the knowledge base to be updated **without retraining** the model.

---

##  System Architecture

```

PDF Documents
→ Text Extraction
→ Chunking with Overlap
→ Embedding Generation
→ FAISS Vector Store
→ User Query
→ Similarity Search (Top-K)
→ Context Injection
→ LLM Answer

```

---

## 🛠️ Tech Stack

| Component | Technology |
|---------|-----------|
| Language Model | Google FLAN-T5 (open-source) |
| Embeddings | Sentence-Transformers |
| Vector Database | FAISS |
| Orchestration | LangChain |
| Frontend | Streamlit |
| Language | Python |

---

## Project Structure

```

rag-document-qa/
│
├── app.py                # Streamlit UI
├── ingest.py             # Document ingestion & FAISS indexing
├── rag_pipeline.py       # Retrieval + generation logic
├── data/
│   └── documents/        # Input PDF documents
├── vectorstore/          # FAISS index (generated locally)
├── requirements.txt
└── README.md

````

---

## ▶️ Running the Project Locally

### 1️⃣ Install dependencies
```bash
pip install -r requirements.txt
````

### 2️⃣ Add documents

Place your PDF files inside:

```
data/documents/
```

### 3️⃣ Ingest & index documents

```bash
python ingest.py
```

This step:

* Loads PDFs
* Chunks text with overlap
* Generates embeddings
* Stores them in a FAISS vector database

### 4️⃣ Run the Streamlit app

```bash
streamlit run app.py
```

---

## Example Questions

* What happens if an employee resigns?
* Is leave encashment allowed?
* What is the notice period?
* How many days of annual leave are permitted?

If a question cannot be answered from the documents, the system explicitly responds with:

> *"I don't know based on the provided documents."*

---

## Hallucination Control Strategies

* Context-only prompt design
* Explicit refusal instructions
* Limited top-k retrieval
* Source chunk visibility in the UI
* Low-temperature generation

The system prioritizes **faithfulness over creativity**.

---

## UI Highlights

* Clean, minimal Streamlit interface
* Configurable retrieval parameters (Top-K, context length)
* Example prompts for better UX
* Expandable source chunks for transparency
* Confidence indicator for user trust

---

## Future Improvements

* Upload PDFs and re-index directly from the UI
* Hybrid search (BM25 + embeddings)
* Reranking for better retrieval quality
* Cloud deployment (Streamlit Cloud / Hugging Face Spaces)
* Evaluation metrics for RAG quality

---


Just say the word 👍
```
