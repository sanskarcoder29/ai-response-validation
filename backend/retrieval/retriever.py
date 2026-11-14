
"""
Retrieval pipeline orchestrator.

Connects:
1. User question validation.
2. Gemini query embedding.
3. Local SQLite vector search.
4. Ranked reference results.

Importing this module does not make API requests.
"""

from typing import Any

from .query_embedder import embed_query, validate_query
from .vector_search import search_similar


def retrieve_for_question(
    question: str,
    top_k: int = 5,
    dataset: str | None = None,
) -> list[dict[str, Any]]:
    """
    Retrieve relevant reference chunks for a user question.

    Args:
        question: User's natural-language question.
        top_k: Number of reference results.
        dataset: Optional dataset filter.

    Returns:
        Ranked reference chunks with similarity scores.
    """

    question = validate_query(question)

    # Convert the question into a numerical vector.
    query_vector = embed_query(question)

    # Find the most similar reference vectors locally.
    results = search_similar(
        query_vector=query_vector,
        top_k=top_k,
        dataset=dataset,
    )

    return results


__all__ = [
    "retrieve_for_question",
]
