# AI Response Validation System with Hallucination Detection Assistance

A modular Python-based project for validating AI-generated responses against reference information. The system is designed to examine response relevance, factual accuracy, evidence support, and completeness, with a focus on identifying unsupported or contradicted claims.

Milestone 1 establishes the research foundation, system architecture, evaluation input module, benchmark dataset preparation, and reference retrieval foundation.

> **Current project status:** The submission interface, backend API, SQLite submission storage, benchmark preprocessing, partial embedding generation, and SQLite-based reference retrieval are implemented. Automated judge agents, scoring, and final verdict generation are planned for subsequent milestones.

---

## 1. Milestone 1 objectives

The objectives of the first milestone are:

- Research LLM evaluation, hallucination detection, factuality, RAG, and LLM-as-a-Judge approaches.
- Study evaluation frameworks such as RAGAS and TruLens.
- Define the system architecture, data flow, components, and proposed judge-agent responsibilities.
- Design a provisional evaluation scoring rubric.
- Implement an input interface for questions and AI-generated responses.
- Validate and persist user submissions through a backend API.
- Acquire and preprocess TruthfulQA and SQuAD benchmark datasets.
- Generate embeddings and build a local reference vector database.
- Implement reference retrieval and evidence context preparation.
- Verify the implemented components through automated and offline tests.

---

## 2. Current implementation status

### Implemented components

- Streamlit-based user input interface.
- FastAPI backend.
- Pydantic request validation.
- SQLite submission storage.
- Health-check and submission API endpoints.
- Knowledge-base search endpoint.
- TruthfulQA acquisition and preprocessing.
- SQuAD pilot acquisition and preprocessing.
- Reference text chunking.
- Gemini Embedding 2 integration.
- Partial reference embedding generation.
- SQLite-based vector storage.
- NumPy cosine similarity search.
- Retrieval orchestration.
- Evidence context construction and metadata preservation.

### Pending components

- Remaining TruthfulQA embeddings.
- SQuAD pilot embeddings.
- Complete vector database population.
- Independent semantic retrieval validation using fresh query embeddings.
- Manual retrieval relevance evaluation.

### Planned for subsequent milestones

- Claim extraction.
- Specialised judge-agent interfaces.
- Automated relevance and accuracy scoring.
- Hallucination classification.
- Evaluation orchestration.
- Structured verdict generation.
- Final evaluation report.

**Important:** The current application accepts and stores submissions and supports reference retrieval. It does not yet calculate automated response-quality scores or generate a final evaluation verdict.

---

## 3. Technology stack

The current project uses:

- **Programming language:** Python 3.14.6.
- **Backend API:** FastAPI.
- **Data validation:** Pydantic.
- **Frontend:** Streamlit.
- **Dataset acquisition:** Hugging Face Dataset Viewer API.
- **HTTP communication:** Python Requests.
- **Embedding model:** Google Gemini Embedding 2.
- **Embedding dimensions:** 3,072.
- **Vector database:** SQLite.
- **Numerical operations:** NumPy.
- **Submission persistence:** SQLite.
- **Environment configuration:** python-dotenv.
- **Automated testing:** pytest.

RAGAS and TruLens have been reviewed as possible future evaluation frameworks but are not currently integrated into the runtime.

For detailed technology decisions, see [`docs/TECH_STACK.md`](docs/TECH_STACK.md).

---

## 4. Repository structure

The project is organised into separate frontend, backend, retrieval, documentation, testing, and data components.

```text
ai-response-validation/
│
├── README.md
├── LICENSE
├── .gitignore
├── .env.example
├── requirements.txt
│
├── backend/
│   ├── app/
│   │   ├── main.py
│   │   ├── schemas.py
│   │   ├── storage.py
│   │   └── knowledge_base.py
│   │
│   ├── retrieval/
│   │   ├── context_builder.py
│   │   ├── query_embedder.py
│   │   ├── retriever.py
│   │   └── vector_search.py
│   │
│   └── scripts/
│       ├── download_truthfulqa.py
│       ├── preprocess_truthfulqa.py
│       ├── download_squad.py
│       ├── preprocess_squad.py
│       ├── chunk_reference_data.py
│       ├── generate_embeddings.py
│       ├── build_vector_index.py
│       ├── test_gemini_embedding.py
│       ├── test_vector_search.py
│       ├── test_retrieval.py
│       ├── test_retriever_mock.py
│       └── test_context_builder.py
│
├── frontend/
│   └── streamlit_app.py
│
├── docs/
│   ├── MILESTONE_1_REPORT.md
│   ├── ARCHITECTURE.md
│   ├── TECH_STACK.md
│   ├── SCORING_RUBRIC.md
│   └── DATASETS.md
│
├── tests/
│   └── test_schemas.py
│
└── data/
    ├── raw/
    └── processed/
```

The `data/` directory contains local benchmark files, generated embeddings, and SQLite databases. These data files are not intended to be committed to Git.

---

## 5. Prerequisites

Before running the project, install:

- Python 3.11 or later (the current development environment uses Python 3.14.6).
- Git.
- An internet connection for dataset acquisition and Gemini embedding requests.
- A Google Gemini API key for embedding generation and query-time retrieval.

The project can perform local SQLite vector searches and offline tests without making external embedding requests.

---

## 6. Installation and setup

### Step 1: Clone the repository

```bash
git clone https://github.com/sanskarcoder29/ai-response-validation.git
cd ai-response-validation
```

### Step 2: Create a virtual environment

```bash
python -m venv .venv
```

### Step 3: Activate the virtual environment

**Windows Command Prompt:**

```cmd
.venv\Scripts\activate
```

**Windows PowerShell:**

```powershell
.venv\Scripts\Activate.ps1
```

**macOS/Linux:**

```bash
source .venv/bin/activate
```

### Step 4: Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Step 5: Configure environment variables

Create a local `.env` file from the provided example.

**Windows Command Prompt:**

```cmd
copy .env.example .env
```

Open `.env` and configure the required credentials using the variable names provided in `.env.example`.

Keep API keys and access tokens private. Never commit `.env` or share screenshots containing credentials.

---

## 7. Running the application

The application uses two separate local services.

### Step 1: Start the FastAPI backend

From the repository root:

```bash
python -m uvicorn backend.app.main:app --reload
```

The backend runs at:

```text
http://127.0.0.1:8000
```

Interactive API documentation:

```text
http://127.0.0.1:8000/docs
```

Health endpoint:

```text
http://127.0.0.1:8000/health
```

### Step 2: Start the Streamlit frontend

Open a second terminal, activate the same virtual environment, and execute:

```bash
streamlit run frontend/streamlit_app.py
```

The frontend runs at:

```text
http://localhost:8501
```

The Streamlit interface communicates with the FastAPI backend.

### Step 3: Submit an evaluation input

The input interface accepts:

- Question.
- AI-generated response.
- Optional reference answer.
- Optional pasted source/reference material.

The backend validates and stores the submitted information in SQLite.

Automated judge-agent evaluation is not enabled in the current milestone.

---

## 8. Benchmark dataset preparation

The project uses two public benchmark datasets.

### TruthfulQA

- Dataset ID: `domenicrosati/TruthfulQA`
- Configuration: `default`
- Split: `train`
- Records downloaded: 817
- Processed records: 817
- Reference chunks: 892

### SQuAD

- Dataset ID: `rajpurkar/squad`
- Configuration: `plain_text`
- Split: `train`
- Available training records: 87,599
- Pilot records downloaded: 500
- Processed records: 500
- Reference chunks: 1,343

The current combined processed reference collection contains 2,235 chunks.

For dataset attribution, licensing, and complete processing details, see [`docs/DATASETS.md`](docs/DATASETS.md).

### Dataset download and preprocessing

Run the following scripts from the repository root.

**1. Download TruthfulQA**

```cmd
python backend\scripts\download_truthfulqa.py
```

**2. Preprocess TruthfulQA**

```cmd
python backend\scripts\preprocess_truthfulqa.py
```

**3. Download the SQuAD pilot sample**

```cmd
python backend\scripts\download_squad.py
```

**4. Preprocess SQuAD**

```cmd
python backend\scripts\preprocess_squad.py
```

**5. Generate combined reference chunks**

```cmd
python backend\scripts\chunk_reference_data.py
```

The expected combined chunk output is:

```text
data/processed/reference_chunks.json
```

The dataset downloaders and preprocessing scripts save their outputs in the local data directory.

---

## 9. Embedding generation and vector indexing

### Embedding model

The project uses Google Gemini Embedding 2.

Current configuration:

- Model: `gemini-embedding-2`
- Embedding dimensions: 3072
- Current maximum new embeddings per script run: 50

### Generate reference embeddings

After configuring the Gemini API key and confirming that the free-tier quota is available, run:

```cmd
python backend\scripts\generate_embeddings.py
```

The embedding script:

- Reads the combined reference chunks.
- Generates embeddings using Gemini.
- Saves generated vectors to JSONL.
- Skips chunk IDs that already have completed embeddings.
- Supports resuming from previously saved results.
- Stops when it reaches an API error or configured run limit.

**Quota note:** Embedding generation uses external Gemini API requests and is subject to the provider's free-tier limits. The current project has paused bulk generation until the quota becomes available again. Paid billing is not enabled.

### Build or update the SQLite vector index

After generating additional embeddings, run:

```cmd
python backend\scripts\build_vector_index.py
```

The vector database is located at:

```text
data/processed/reference_vectors.db
```

The database stores reference vectors and associated metadata.

NumPy performs cosine similarity calculations during local retrieval.

---

## 10. API endpoints

The current backend exposes the following endpoints.

### GET `/health`

Returns the backend service status.

Example response:

```json
{
  "status": "ok",
  "service": "ai-response-validation-api"
}
```

### POST `/submissions`

Validates and stores a user submission in SQLite.

Example request:

```json
{
  "question": "What is photosynthesis?",
  "ai_response": "Photosynthesis is the process by which green plants convert light energy into chemical energy.",
  "reference_answer": "Photosynthesis converts light energy into chemical energy stored in sugars.",
  "source_document": null
}
```

The endpoint returns HTTP 201 when the submission is successfully stored.

### POST `/knowledge-base/search`

Searches the local reference vector database using a natural-language query.

Example request:

```json
{
  "query": "How do plants convert light into chemical energy?",
  "top_k": 4
}
```

The retrieval pipeline generates a query embedding using Gemini, searches stored reference vectors locally, and returns ranked candidate passages.

**Important:** This endpoint requires a Gemini embedding API request for the query. The subsequent vector comparison is performed locally.

Retrieved passages are candidate evidence and are not proof that a generated response is factually correct.

---

## 11. Testing and verification

### Run the formal pytest suite

From the repository root:

```cmd
python -m pytest -v
```

The current formal test suite contains three schema validation tests:

1. Optional reference fields are accepted.
2. Blank required fields are rejected.
3. Invalid `top_k` values are rejected.

**Latest recorded result:** 3 passed, 0 failed.

### Run offline retrieval checks

The following standalone scripts verify the local retrieval implementation without requiring additional Gemini API requests.

**SQLite vector search:**

```cmd
python backend\scripts\test_vector_search.py
```

**Offline retrieval:**

```cmd
python backend\scripts\test_retrieval.py
```

**Mock retrieval pipeline:**

```cmd
python backend\scripts\test_retriever_mock.py
```

**Evidence context construction:**

```cmd
python backend\scripts\test_context_builder.py
```

These scripts cover vector loading, ranking, self-match, dataset filtering, retrieval integration, evidence generation, metadata preservation, similarity thresholds, and context length limits.

### Check Python compilation

```cmd
python -m compileall -q backend tests
```

No output indicates that the compilation completed without reported syntax errors.

### External API test

The script below performs a real Gemini embedding request and therefore consumes API quota:

```cmd
python backend\scripts\test_gemini_embedding.py
```

Run it only when the API quota is available.

---

## 12. Data storage

Important local files and directories include:

| Location | Purpose |
|---|---|
| `data/raw/` | Downloaded benchmark records |
| `data/processed/` | Cleaned datasets and reference chunks |
| `data/processed/reference_embeddings.jsonl` | Generated reference embeddings |
| `data/processed/reference_vectors.db` | SQLite reference vector database |
| `data/submissions.db` | SQLite submission database |

All these files are local project data and should remain outside version control.

---

## 13. Documentation

Detailed project documentation is available in the `docs/` directory.

- [`MILESTONE_1_REPORT.md`](docs/MILESTONE_1_REPORT.md) — Research summary, milestone progress, implementation status, verification evidence, risks, and next steps.
- [`ARCHITECTURE.md`](docs/ARCHITECTURE.md) — System architecture, component responsibilities, data flow, and proposed future evaluation pipeline.
- [`TECH_STACK.md`](docs/TECH_STACK.md) — Current technologies, technology decisions, and future extensions.
- [`DATASETS.md`](docs/DATASETS.md) — Benchmark dataset details, acquisition configurations, attribution, licensing, and processing status.
- [`SCORING_RUBRIC.md`](docs/SCORING_RUBRIC.md) — Proposed evaluation dimensions, weights, verdict thresholds, and calibration plan.

The scoring rubric is a proposal for mentor review, not a validated evaluation model.

---

## 14. Important limitations

- The current submission endpoint stores information but does not automatically score responses.
- Judge agents and automated hallucination detection are planned for subsequent milestones.
- The reference vector database currently contains 810 TruthfulQA embeddings.
- The remaining TruthfulQA embeddings and SQuAD pilot embeddings are pending.
- Independent semantic retrieval validation using fresh query embeddings remains pending.
- Vector similarity is not the same as factual verification.
- The current interface accepts pasted reference text rather than complete PDF or DOCX document uploads.
- Benchmark datasets do not represent every domain or every current fact.
- Dataset licences and attribution requirements are separate from the project's source-code licence.

---

## 15. Milestone 1 checklist

- [x] Research summary documented.
- [x] System architecture documented.
- [x] Proposed judge-agent responsibilities documented.
- [x] Provisional evaluation scoring rubric documented.
- [x] FastAPI backend implemented.
- [x] Streamlit input interface implemented.
- [x] SQLite submission persistence implemented.
- [x] TruthfulQA acquisition and preprocessing completed.
- [x] SQuAD pilot acquisition and preprocessing completed.
- [x] Reference chunking implemented.
- [x] Partial reference embedding generation completed.
- [x] SQLite vector search implemented.
- [x] Offline retrieval and context-building tests passed.
- [x] Formal pytest suite passed.
- [ ] Remaining reference embeddings generated.
- [ ] Independent semantic retrieval quality validated.
- [ ] Final verification screenshots added to milestone documentation.
- [ ] Final architecture and progress reviewed with the mentor.

---

## 16. Acknowledgements and references

This project uses public benchmark datasets and open-source software.

Dataset attribution, licensing information, and original research references are documented in [`docs/DATASETS.md`](docs/DATASETS.md).

---

**Project note:** The system is being developed incrementally. The current milestone establishes the input, storage, benchmark-processing, and retrieval foundation required for future evidence-based AI response evaluation.

## Milestone 1 Progress — 16 November 2026

- M1.1: Research and technical understanding documented.
- M1.2: Initial system architecture documented. Mentor review pending.
- M1.3: Core Streamlit and FastAPI evaluation input functionality, validation and SQLite submission storage implemented.
- M1.4: 2,235 reference chunks embedded using Gemini Embedding 2 and indexed in SQLite.

Offline retrieval and context tests passed. Initial live retrieval tests succeeded for TruthfulQA and SQuAD.

Retrieval-result deduplication is implemented. Source-level SQuAD deduplication remains a future improvement.

See docs/MILESTONE_1_REPORT.md and docs/ARCHITECTURE.md for implementation details and limitations.