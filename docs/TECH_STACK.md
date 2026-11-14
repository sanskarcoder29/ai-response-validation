# Technology Stack — AI Response Validation System

**Project:** Development of AI Response Validation System with Hallucination Detection Assistance
**Milestone:** 1 — Foundation and Evaluation Understanding
**Last updated:** 14 November 2026

## 1. Overview

The project uses Python-based technologies to build a modular AI response validation system.

The current implementation focuses on user input, backend validation, submission persistence, benchmark dataset processing, embedding generation, reference vector storage, and similarity-based retrieval.

The selected technologies favour local development, understandable components, and the ability to extend the system in subsequent milestones.

---

## 2. Technologies currently implemented

### 2.1 Programming language — Python 3.14.6

Python is the primary programming language used throughout the project.

It is used for:

- Backend API development.
- Data acquisition and preprocessing.
- Reference text chunking.
- Embedding generation.
- Vector database construction.
- Similarity search and retrieval.
- Automated testing.

The current development environment uses Python 3.14.6.

### 2.2 API framework — FastAPI

FastAPI provides the backend HTTP API.

It is responsible for:

- Receiving user submissions.
- Exposing health-check and knowledge-base search endpoints.
- Connecting API requests with backend modules.
- Returning structured JSON responses.
- Providing interactive API documentation.

The API documentation is available at `/docs`.

### 2.3 Data validation — Pydantic

Pydantic is used with FastAPI to validate request data.

It provides:

- Required and optional fields.
- Input validation.
- Blank-field checks.
- Configured length constraints.
- Structured request schemas.

### 2.4 Frontend — Streamlit

Streamlit provides the initial user interface.

It supports:

- Entering a question.
- Entering an AI-generated response.
- Providing an optional reference answer.
- Pasting optional source/reference material.
- Submitting information to the backend API.
- Displaying submission responses.

The current implementation uses pasted text for reference material. Full PDF and DOCX upload and parsing are not yet implemented.

### 2.5 Benchmark dataset acquisition — Hugging Face Dataset Viewer API

Public benchmark data is acquired using the Hugging Face Dataset Viewer API through Python Requests.

The current datasets are:

- **TruthfulQA:** 817 records acquired and processed.
- **SQuAD:** 500 records acquired as an initial pilot sample and processed.

The acquired records are stored locally as JSON files before preprocessing.

The current acquisition workflow does not depend on ChromaDB.

### 2.6 HTTP communication — Requests

The Python Requests library is used for HTTP communication.

It supports:

- Accessing the Hugging Face Dataset Viewer API.
- Sending embedding requests to the Gemini API.
- Handling HTTP responses.
- Managing request timeouts and error responses.

### 2.7 Embedding generation — Google Gemini Embedding 2

Google Gemini Embedding 2 is used to generate numerical vector representations of reference passages and search queries.

Current configuration:

- Model: `gemini-embedding-2`
- Expected embedding dimensions: 3072

The embedding workflow converts text into vectors that can be compared using cosine similarity.

Generated reference embeddings are stored in JSONL format before being indexed in SQLite.

**Current limitation:** Bulk embedding generation is partially completed because of the Gemini free-tier daily API quota. Further embedding requests are pending. Paid billing is not enabled for this project.

### 2.8 Vector storage — SQLite

SQLite is used as the local persistent vector database.

Current vector database:

`data/processed/reference_vectors.db`

It stores:

- Reference chunk identifiers.
- Dataset and record information.
- Embedding vectors.
- Vector norms.
- Associated reference metadata.

The project also uses a separate SQLite database for user submissions:

`data/submissions.db`

Separating submission storage from reference vector storage keeps the two responsibilities independent.

### 2.9 Numerical operations — NumPy

NumPy is used for vector operations and cosine similarity calculations.

It supports:

- Loading stored numerical vectors.
- Validating vector dimensions.
- Calculating cosine similarity.
- Ranking reference results.
- Selecting the top-k matching passages.

Reference vector searching is performed locally after the query embedding has been generated.

### 2.10 Environment configuration — python-dotenv

python-dotenv loads local environment variables from the project's `.env` file.

It is used to access API credentials without placing secrets directly in source code.

API keys and access tokens must never be committed to the Git repository.

### 2.11 Automated testing — pytest

pytest is used for automated testing.

The current formal test suite checks:

- Submission schema validation.
- Optional reference fields.
- Rejection of blank required fields.
- Knowledge-base search parameter constraints.

Additional standalone scripts verify the retrieval pipeline, vector search, evidence context construction, and metadata preservation.

---

## 3. Evaluation frameworks reviewed

### 3.1 RAGAS

RAGAS is a framework for evaluating Retrieval-Augmented Generation systems.

Relevant evaluation concepts include:

- Faithfulness.
- Answer relevancy.
- Context precision.
- Context recall.
- Factual correctness.

These metrics may help evaluate retrieval quality and factual support in later milestones.

### 3.2 TruLens

TruLens provides tools for evaluating and monitoring LLM applications.

Its RAG evaluation concepts include:

- Context relevance.
- Groundedness.
- Answer relevance.

These concepts align with the proposed evaluation dimensions of the project.

### 3.3 Current decision

RAGAS and TruLens have been reviewed as potential evaluation frameworks.

Neither framework is currently integrated into the active runtime.

The project will first establish its own explicit evaluation rubric and retrieval workflow. Selected framework metrics may be introduced after the evaluation pipeline becomes stable.

---

## 4. LLM provider and model selection

The embedding model has been selected as Google Gemini Embedding 2 for the current reference retrieval prototype.

The final LLM provider and model for automated response evaluation have not yet been selected.

The evaluation model selection should consider:

- Availability.
- Cost and free-tier limits.
- Model quality.
- Privacy requirements.
- Structured output support.
- Mentor and team preferences.

The embedding model and future evaluation model serve different purposes and should be treated as separate components.

API keys must remain in environment variables and must not be hardcoded or committed.

---

## 5. Current data and storage structure

The current project uses the following data organisation.

### Raw benchmark data

`data/raw/`

Contains acquired benchmark records.

### Processed benchmark data

`data/processed/`

Contains cleaned records and chunked reference material.

Important files include:

- `truthfulqa_clean.json`
- `squad_clean.json`
- `reference_chunks.json`
- `reference_embeddings.jsonl`
- `reference_vectors.db`

### Submission storage

`data/submissions.db`

Stores validated user submissions.

### Reference vector storage

`data/processed/reference_vectors.db`

Stores indexed reference embeddings and associated metadata.

Raw datasets, processed data, embeddings, databases, and API credentials should remain outside version control unless a particular non-sensitive sample is intentionally required for documentation or testing.

---

## 6. Local deployment

The current local development environment uses the following services.

**Streamlit frontend**

`http://localhost:8501`

**FastAPI backend**

`http://localhost:8000`

**FastAPI interactive API documentation**

`http://localhost:8000/docs`

**FastAPI health endpoint**

`http://127.0.0.1:8000/health`

**SQLite submission database**

`data/submissions.db`

**SQLite reference vector database**

`data/processed/reference_vectors.db`

The frontend communicates with the backend API for submission processing. The retrieval endpoint connects to the local reference vector search pipeline.

---

## 7. Earlier and legacy technology choices

ChromaDB was considered in an earlier version of the reference ingestion workflow.

The active retrieval implementation has since moved to SQLite and NumPy.

The older ChromaDB-oriented ingestion script remains in the repository as legacy code. It is not part of the active vector retrieval pipeline.

The earlier MiniLM-based local embedding approach was also replaced by Gemini Embedding 2.

The current documented architecture should therefore be treated as:

**Gemini Embedding 2 + SQLite + NumPy cosine similarity search.**

---

## 8. Planned technology extensions

The following tools and capabilities may be considered in later milestones:

- RAGAS or TruLens for structured RAG evaluation.
- An LLM provider for automated response judging.
- Claim extraction and structured evaluation interfaces.
- PDF and DOCX document parsing.
- Additional retrieval quality metrics.
- Human-reviewed benchmark evaluation.

These are planned extensions and are not represented as completed functionality.

---

## 9. Technology stack summary

| Area | Current technology | Status |
|---|---|---|
| Programming language | Python 3.14.6 | Implemented |
| Backend API | FastAPI | Implemented |
| Validation | Pydantic | Implemented |
| Frontend | Streamlit | Implemented |
| Dataset acquisition | Hugging Face Dataset Viewer API | Implemented |
| HTTP communication | Requests | Implemented |
| Embeddings | Gemini Embedding 2 | Partially implemented |
| Vector storage | SQLite | Implemented |
| Similarity operations | NumPy | Implemented |
| Submission persistence | SQLite | Implemented |
| Environment configuration | python-dotenv | Implemented |
| Testing | pytest | Implemented |
| RAG evaluation | RAGAS / TruLens | Reviewed; not integrated |
| Automated LLM judging | Provider/model pending | Planned |

---

**Conclusion:** The current technology stack provides a local foundation for submission handling, benchmark data processing, embedding generation, and reference retrieval. Automated response evaluation, judge-agent orchestration, and verdict generation will be developed in subsequent milestones.
