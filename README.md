# Engineering RAG Assistant — V0

A local Retrieval-Augmented Generation prototype designed to answer questions from engineering and technical PDF documents.

The goal of this project is not to build a generic PDF chatbot, but to explore the core components of a document-grounded RAG pipeline:

- PDF ingestion
- text chunking
- local embedding generation
- vector search with ChromaDB
- source-grounded retrieval
- optional query rewriting
- local LLM answer generation with Ollama
- transparent retrieval debugging with source pages, chunk IDs and similarity scores

This is a first working version intended as a portfolio project. The current focus is on building a clean and understandable RAG pipeline before adding more advanced retrieval methods such as BM25, hybrid search, reranking models and systematic evaluation.

---

## Features

- Index all PDF files stored in `data/raw`
- Extract text page by page
- Split documents into overlapping chunks
- Store embeddings in a persistent Chroma vector database
- Retrieve the most relevant chunks for a user question
- Display retrieved chunks with:
  - source document
  - page number
  - chunk ID
  - Chroma distance
  - semantic score
- Optional query rewriting to improve retrieval
- Optional local answer generation with Ollama
- Streamlit interface for quick testing and debugging

---

## Tech stack

- Python
- Streamlit
- pypdf
- LangChain
- ChromaDB
- SentenceTransformers / Hugging Face embeddings
- Ollama
- python-dotenv

Current embedding model:

```text
BAAI/bge-m3
```

Current local generation model:

```text
llama3.1:8b
```

Current query rewriting model:

```text
dolphin-mixtral:latest
```

---

## Project structure

```text
.
├── app/
│   └── config.py
├── data/
│   ├── raw/
│   │   └── PDF files to index
│   └── vector_db/
│       └── Local Chroma database
├── src/
│   ├── chunking/
│   │   └── splitter.py
│   ├── embeddings/
│   │   └── embedder.py
│   ├── generation/
│   │   ├── ollama_generator.py
│   │   └── query_rewriter.py
│   ├── ingestion/
│   │   └── pdf_loader.py
│   ├── pipeline/
│   │   └── index_documents.py
│   ├── retrieval/
│   │   ├── query_classifier.py
│   │   ├── reranker.py
│   │   └── retriever.py
│   └── vectorstore/
│       └── chroma_store.py
├── streamlit_app.py
├── requirements.txt
└── .env.example
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/pierre-b-ai/engineering-rag-assistant.git
cd engineering-rag-assistant
```

Create and activate a virtual environment:

```bash
python -m venv .venv
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Ollama setup

This project can generate answers with a local LLM through Ollama.

Install Ollama, then pull the local generation model:

```bash
ollama pull llama3.1:8b
```

If you want to use the optional query rewriting feature:

```bash
ollama pull dolphin-mixtral:latest
```

Make sure Ollama is running locally:

```text
http://localhost:11434
```

---

## Usage

Place one or more PDF files inside:

```text
data/raw/
```

Launch the Streamlit app:

```bash
streamlit run streamlit_app.py
```

In the interface:

1. Click **Reindex PDFs from data/raw**
2. Select an answer mode:
   - `Retrieval only`
   - `Local LLM`
   - `API LLM (coming soon)`
3. Enter a question
4. Select the number of chunks to retrieve
5. Click **Search**

The app will display the retrieved chunks and, if `Local LLM` mode is selected, generate a grounded answer using the retrieved context.

---

## How it works

### 1. PDF ingestion

PDF files are loaded from `data/raw`.

Each PDF is read page by page. Empty pages are ignored. Each extracted page keeps metadata including the source filename and page number.

### 2. Chunking

Pages are split into overlapping text chunks using a recursive character splitter.

Current configuration:

```python
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 250
MIN_CHUNK_LENGTH = 300
```

Each chunk keeps its source document, page number and chunk ID.

### 3. Embeddings

Chunks are embedded locally using a Hugging Face embedding model.

Current model:

```text
BAAI/bge-m3
```

This model was selected as a stronger multilingual embedding model than lightweight baseline models.

### 4. Vector storage

Embeddings and chunks are stored in a local persistent ChromaDB collection.

Current collection name:

```text
engineering_docs
```

The vector database is stored locally in:

```text
data/vector_db/
```

### 5. Retrieval

For each user query, the app retrieves more chunks internally than the final number displayed.

This makes it possible to retrieve a broader candidate set before selecting the final top chunks.

Current retrieval logic:

- precise questions retrieve a moderate number of candidates
- broad questions retrieve more candidates
- Chroma returns distances
- distances are converted into readable semantic scores

The current reranking step is intentionally simple:

```text
semantic_score = 1 / (1 + distance)
```

Higher semantic score means a better match.

### 6. Query rewriting

The app includes an experimental query rewriting option.

When enabled, a local Ollama model rewrites the user question into a short enriched retrieval query. The rewritten query is appended to the original question instead of replacing it, to reduce the risk of retrieval degradation.

This feature can improve recall, but it can also hurt retrieval when the rewrite adds noise.

### 7. Local LLM generation

When `Local LLM` mode is selected, retrieved chunks are passed to a local Ollama model.

The prompt instructs the model to:

- answer only from the provided context
- compare available sources
- avoid unsupported assumptions
- say clearly when the answer is not found
- cite the source document and page

---

## Current limitations

This is a V0 prototype. Several parts are intentionally simple and will be improved later.

Current limitations:

- PDF extraction only works well on text-based PDFs
- scanned documents are not supported yet
- no OCR pipeline
- no HTML documentation ingestion yet
- no BM25 retrieval yet
- no hybrid retrieval yet
- no cross-encoder reranker yet
- no systematic Recall@k evaluation yet
- no automated test set
- no API backend yet
- API LLM mode is not implemented yet

---

## Roadmap

Planned improvements:

- Add BM25 retrieval
- Add hybrid search: BM25 + embeddings
- Add a real reranking model
- Add Recall@k evaluation on a small benchmark question set
- Compare embedding models:
  - BAAI/bge-m3
  - multilingual-e5-base
  - all-MiniLM-L6-v2
- Add HTML documentation ingestion
- Add OCR support for scanned technical documents
- Add FastAPI backend
- Add Docker / Docker Compose setup
- Add Langfuse or LangSmith tracing
- Add a small public demo dataset
- Add unit tests

---

## Why this project matters

Technical documentation is often long, fragmented and difficult to search manually.

This project explores how a RAG system can help users retrieve grounded answers from engineering documents while keeping the retrieval process transparent.

The focus is not only on generating answers, but also on showing the retrieved evidence behind each answer.

---

## Status

V0 working prototype.

The project currently supports local PDF indexing, vector search, retrieval debugging and local LLM generation.