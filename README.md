# AI Response Validation System with Hallucination Detection Assistance

A modular project for evaluating AI-generated responses for relevance, factual accuracy, evidence support, and completeness. The first milestone establishes the research foundation, system architecture, evaluation-input API, and a reference knowledge base for retrieval-grounded evaluation.

> **Project status:** Milestone 1 foundation/scaffold. The submission API and reference-retrieval foundation are provided here. The specialised judge agents, LLM-as-a-Judge scoring, and final verdict pipeline are planned components and are not claimed to be implemented yet.

## Milestone 1 goals

- Document LLM evaluation, hallucination detection, RAG, LLM-as-a-Judge, RAGAS, and TruLens.
- Define the system components, agent responsibilities, data flow, data models, and provisional scoring rubric.
- Provide a single submission interface for a question, AI-generated response, and optional reference answer/source material.
- Ingest a pilot sample of TruthfulQA and SQuAD, split reference content into chunks, index it in ChromaDB, and retrieve relevant evidence.

## Proposed technology stack

- **Language:** Python 3.11+ for the main implementation.
- **API:** FastAPI and Pydantic for endpoints and input validation.
- **UI:** Streamlit for the evaluation submission form.
- **Datasets:** Hugging Face `datasets` for TruthfulQA and SQuAD.
- **Embeddings:** ChromaDB's default local embedding function (MiniLM-based).
- **Vector store:** ChromaDB for persistent semantic indexing and retrieval.
- **Metadata storage:** SQLite for submitted evaluation inputs.
- **Evaluation research:** RAGAS and TruLens for metric/framework comparison.
- **Testing:** pytest for input and API tests.

See [`docs/TECH_STACK.md`](docs/TECH_STACK.md) and [`docs/MILESTONE_1_REPORT.md`](docs/MILESTONE_1_REPORT.md).

## Repository structure

```text
ai-response-validation/
├── README.md
├── LICENSE
├── .gitignore
├── .env.example
├── requirements.txt
├── backend/
│   ├── app/
│   │   ├── main.py
│   │   ├── schemas.py
│   │   ├── storage.py
│   │   └── knowledge_base.py
│   └── scripts/
│       └── ingest_benchmarks.py
├── frontend/
│   └── streamlit_app.py
├── docs/
│   ├── MILESTONE_1_REPORT.md
│   ├── ARCHITECTURE.md
│   ├── TECH_STACK.md
│   ├── SCORING_RUBRIC.md
│   └── DATASETS.md
├── tests/
│   └── test_schemas.py
└── data/                 # local database/vector data; not committed
```

## Setup

### 1. Clone and create a virtual environment

```bash
git clone https://github.com/sanskarcoder29/ai-response-validation.git
cd ai-response-validation
python -m venv .venv
```

Activate it:

**Windows PowerShell**

```powershell
.venv\Scripts\Activate.ps1
```

**macOS/Linux**

```bash
source .venv/bin/activate
```

### 2. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Start the API

From the repository root:

```bash
python -m uvicorn backend.app.main:app --reload
```

Open `http://127.0.0.1:8000/docs` to view and try the API. The health endpoint is `http://127.0.0.1:8000/health`.

### 4. Start the submission interface

Open a second terminal, activate the same virtual environment, and run:

```bash
streamlit run frontend/streamlit_app.py
```

The UI sends the question, response, and optional reference content to the FastAPI service.

### 5. Build the pilot reference knowledge base

From the repository root:

```bash
python -m backend.scripts.ingest_benchmarks --max-squad-rows 500
```

This indexes all 817 TruthfulQA generation examples and a 500-row SQuAD pilot sample by default. To process the complete SQuAD training split, use:

```bash
python -m backend.scripts.ingest_benchmarks --max-squad-rows 0
```

The first run downloads the datasets and embedding model, so it may take some time and disk space. Generated dataset caches and ChromaDB files remain local and are excluded from Git.

### 6. Run tests

```bash
python -m pytest -q
```

## API endpoints in this milestone

- `GET /health` — service status.
- `POST /submissions` — validates and stores an evaluation submission in SQLite.
- `POST /knowledge-base/search` — retrieves relevant indexed reference chunks for a query.

Example submission JSON:

```json
{
  "question": "What is photosynthesis?",
  "ai_response": "Photosynthesis is the process by which green plants convert light energy into chemical energy.",
  "reference_answer": "Photosynthesis converts light energy into chemical energy stored in sugars.",
  "source_document": null
}
```

Example retrieval JSON:

```json
{
  "query": "How do plants convert light into chemical energy?",
  "top_k": 4
}
```

## Important scope and limitations

- The current `/submissions` endpoint stores inputs; it does **not** yet run the four judge agents or return an actual evaluation verdict.
- The knowledge base is for reference retrieval. Retrieved passages are evidence candidates, not proof that a generated answer is correct.
- The default ingestion command uses a small SQuAD sample first so that local indexing can be tested before running a larger ingestion job.
- Dataset licenses and attribution requirements are separate from the license for this repository's original source code. See [`docs/DATASETS.md`](docs/DATASETS.md).
- Do not commit API keys, `.env` files, downloaded datasets, SQLite databases, or vector-store files.

## Milestone 1 submission checklist

- [ ] Research summary and framework comparison documented.
- [ ] Architecture diagram and agent responsibilities documented.
- [ ] Provisional scoring rubric documented and marked for mentor review.
- [ ] Evaluation input API and UI run locally.
- [ ] TruthfulQA and SQuAD ingestion tested.
- [ ] ChromaDB semantic retrieval tested with representative questions.
- [ ] Tests pass and screenshots/results are added to the milestone report.
- [ ] Repository link, architecture, and stack shared with the mentor.

## Acknowledgements

This project uses public benchmark datasets and open-source software. Dataset references, license information, and benchmark citations are recorded in [`docs/DATASETS.md`](docs/DATASETS.md).
