"""Unit tests for Milestone 1 request validation."""

import pytest
from pydantic import ValidationError

from backend.app.schemas import EvaluationSubmission, KnowledgeSearchRequest


def test_valid_submission_accepts_optional_reference_fields() -> None:
    submission = EvaluationSubmission(
        question="What is RAG?",
        ai_response="RAG retrieves external context before generating an answer.",
    )
    assert submission.reference_answer is None
    assert submission.source_document is None


def test_submission_rejects_blank_required_fields() -> None:
    with pytest.raises(ValidationError):
        EvaluationSubmission(question="   ", ai_response="A response")


def test_search_top_k_is_bounded() -> None:
    with pytest.raises(ValidationError):
        KnowledgeSearchRequest(query="Find relevant context", top_k=25)
