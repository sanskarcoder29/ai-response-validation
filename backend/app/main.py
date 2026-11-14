
"""FastAPI entry point for the Milestone 1 input and retrieval modules."""

from fastapi import FastAPI

from backend.app.schemas import EvaluationSubmission, KnowledgeSearchRequest
from backend.app.storage import initialize_database, save_submission

app = FastAPI(
    title="AI Response Validation API",
    description=(
        "Milestone 1 API for submitting an AI response and retrieving reference evidence. "
        "Automated judge-agent evaluation will be added in a later milestone."
    ),
    version="0.1.0",
)


@app.on_event("startup")
def on_startup() -> None:
    initialize_database()


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok", "service": "ai-response-validation-api"}


@app.post("/submissions", status_code=201)
def create_submission(submission: EvaluationSubmission) -> dict[str, str]:
    """Validate and persist a user submission; this endpoint does not score it yet."""
    saved = save_submission(submission.model_dump())
    return {
        **saved,
        "status": "accepted",
        "message": (
            "Submission stored successfully. "
            "Automated evaluation is not enabled in Milestone 1."
        ),
    }


@app.post("/knowledge-base/search")
def search_knowledge_base(request: KnowledgeSearchRequest) -> dict:
    """Search the reference knowledge base when this endpoint is requested."""

    # Import SQLite-based knowledge retrieval only when needed.
    from backend.app.knowledge_base import search_reference

    matches = search_reference(request.query, request.top_k)

    return {
        "query": request.query,
        "match_count": len(matches),
        "matches": matches,
        "notice": "Retrieved passages are candidate evidence and require evaluation.",
    }
