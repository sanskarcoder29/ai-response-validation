"""Pydantic schemas for evaluation submissions and reference retrieval."""

from pydantic import BaseModel, Field, field_validator


class EvaluationSubmission(BaseModel):
    question: str = Field(min_length=3, max_length=5_000)
    ai_response: str = Field(min_length=1, max_length=20_000)
    reference_answer: str | None = Field(default=None, max_length=10_000)
    source_document: str | None = Field(default=None, max_length=50_000)

    @field_validator("question", "ai_response")
    @classmethod
    def required_text_must_not_be_blank(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("This field cannot contain only whitespace.")
        return cleaned

    @field_validator("reference_answer", "source_document")
    @classmethod
    def optional_text_is_normalized(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = value.strip()
        return cleaned or None


class KnowledgeSearchRequest(BaseModel):
    query: str = Field(min_length=3, max_length=5_000)
    top_k: int = Field(default=4, ge=1, le=10)

    @field_validator("query")
    @classmethod
    def query_must_not_be_blank(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("Query cannot contain only whitespace.")
        return cleaned
