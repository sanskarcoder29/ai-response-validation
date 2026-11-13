# Proposed Technology Stack

The choices below favour a small, understandable prototype that can be tested locally and extended in later milestones.

- **Programming language — Python 3.11 or later:** strong ecosystem for LLM evaluation, NLP, APIs, and datasets.
- **API framework — FastAPI:** typed request models, validation, automatic OpenAPI documentation, and a clean API boundary.
- **Validation — Pydantic v2:** required/optional fields, limits, and consistent request schemas.
- **Frontend — Streamlit:** quick single-form interface without a separate JavaScript frontend.
- **Dataset loading — Hugging Face `datasets`:** direct loading of public benchmark datasets and splits.
- **Embeddings — ChromaDB default local embedding function (MiniLM-based):** avoids needing a paid embeddings API for the initial retrieval prototype.
- **Vector database — ChromaDB persistent local collection:** simple local indexing and semantic retrieval with metadata.
- **Submission storage — SQLite:** no separate database server required for the milestone prototype.
- **Evaluation frameworks — RAGAS and TruLens:** research and compare established metrics and evaluation workflows.
- **Tests — pytest:** lightweight automated tests for validation and service behaviour.

## Why RAGAS and TruLens?

RAGAS publishes metrics for RAG systems including faithfulness, response relevancy, context precision, context recall, and factual correctness. These map closely to the proposed evaluation dimensions and retrieval checks. TruLens documents the RAG triad and evaluation concepts such as groundedness, context relevance, and answer relevance. The project should study both, then select a primary framework for the first working evaluation pipeline rather than integrating both at once.

## Why not select an LLM provider yet?

The mentor's Milestone 1 brief does not prescribe a model provider. Keep an adapter/interface boundary so that Gemini, OpenAI, or a local model can be plugged in after the mentor confirms availability, cost, privacy, and team preference. Never commit API keys.

## Initial local deployment

- Streamlit UI: `http://localhost:8501`
- FastAPI backend: `http://localhost:8000`
- FastAPI interactive API docs: `http://localhost:8000/docs`
- SQLite submission database: `data/submissions.db`
- ChromaDB files: `data/chroma/`
