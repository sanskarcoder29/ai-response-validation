
from typing import Any

from backend.retrieval.retriever import retrieve_for_question


def search_reference(
    query: str,
    top_k: int = 4,
) -> list[dict[str, Any]]:
    """
    Search the local SQLite reference vector database.

    Query embeddings are generated using Gemini.
    Reference vectors are searched locally using NumPy.

    The returned structure preserves compatibility with
    the existing FastAPI knowledge-base search endpoint.
    """

    if not isinstance(query, str) or not query.strip():
        raise ValueError("Search query must not be empty.")

    if (
        not isinstance(top_k, int)
        or isinstance(top_k, bool)
        or top_k < 1
    ):
        raise ValueError("top_k must be a positive integer.")

    retrieved_results = retrieve_for_question(
        question=query,
        top_k=top_k,
    )

    matches: list[dict[str, Any]] = []

    for result in retrieved_results:
        similarity_score = float(result["similarity_score"])

        original_metadata = result.get("metadata") or {}

        if not isinstance(original_metadata, dict):
            original_metadata = {}

        metadata = dict(original_metadata)

        metadata.update(
            {
                "chunk_id": result.get("chunk_id"),
                "dataset": result.get("dataset"),
                "record_id": result.get("record_id"),
                "chunk_index": result.get("chunk_index"),
            }
        )

        match = {
            "text": result.get("text", ""),
            "metadata": metadata,
            "distance": 1.0 - similarity_score,
            "rank": int(result["rank"]),
            "similarity_score": similarity_score,
        }

        matches.append(match)

    return matches
