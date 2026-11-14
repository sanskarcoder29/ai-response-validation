
"""
Evidence context builder for hallucination detection.

Converts retrieved reference chunks into structured,
traceable evidence for downstream answer validation.

No external API requests are made.
"""

import math
from typing import Any


def build_evidence_context(
    results: list[dict[str, Any]],
    max_chars: int = 12000,
    min_similarity: float | None = None,
) -> dict[str, Any]:
    """
    Build a structured grounding context.

    Args:
        results: Ranked results from vector search.
        max_chars: Maximum context characters.
        min_similarity: Optional minimum similarity threshold.

    Returns:
        Dictionary containing formatted context and evidence.
    """

    if not isinstance(results, list):
        raise TypeError("Results must be a list.")

    if isinstance(max_chars, bool) or not isinstance(max_chars, int):
        raise ValueError("max_chars must be a positive integer.")

    if max_chars <= 0:
        raise ValueError("max_chars must be greater than zero.")

    if min_similarity is not None:
        if (
            isinstance(min_similarity, bool)
            or not isinstance(min_similarity, (int, float))
            or not math.isfinite(float(min_similarity))
            or not -1.0 <= min_similarity <= 1.0
        ):
            raise ValueError("Invalid minimum similarity threshold.")

    evidence_items = []
    context_sections = []

    omitted_count = 0

    for result in results:

        if not isinstance(result, dict):
            raise ValueError("Each result must be a dictionary.")

        chunk_id = result.get("chunk_id")
        dataset = result.get("dataset")
        text = result.get("text")
        score = result.get("similarity_score")

        if not isinstance(chunk_id, str) or not chunk_id:
            raise ValueError("Missing reference chunk ID.")

        if not isinstance(dataset, str) or not dataset:
            raise ValueError("Missing dataset name.")

        if not isinstance(text, str) or not text.strip():
            raise ValueError("Missing reference text.")

        if (
            isinstance(score, bool)
            or not isinstance(score, (int, float))
            or not math.isfinite(float(score))
            or not -1.0 <= score <= 1.0
        ):
            raise ValueError("Invalid similarity score.")

        if (
            min_similarity is not None
            and score < min_similarity
        ):
            continue

        evidence_id = f"E{len(evidence_items) + 1}"

        metadata = result.get("metadata", {})

        if not isinstance(metadata, dict):
            raise ValueError("Invalid reference metadata.")

        source_url = metadata.get("source_url")

        section = (
            f"[{evidence_id}]\n"
            f"Dataset: {dataset}\n"
            f"Chunk ID: {chunk_id}\n"
            f"Similarity: {float(score):.6f}\n"
            f"Reference:\n{text.strip()}\n"
        )

        separator = "\n"

        additional_length = (
            len(separator) + len(section)
            if context_sections else len(section)
        )

        current_length = sum(
            len(part) for part in context_sections
        ) + max(0, len(context_sections) - 1)

        if current_length + additional_length > max_chars:
            omitted_count += 1
            continue

        evidence_items.append({
            "evidence_id": evidence_id,
            "chunk_id": chunk_id,
            "dataset": dataset,
            "record_id": result.get("record_id"),
            "similarity_score": float(score),
            "text": text.strip(),
            "source_url": source_url,
            "metadata": metadata,
        })

        context_sections.append(section)

    context = "\n".join(context_sections)

    return {
        "context": context,
        "evidence": evidence_items,
        "included_count": len(evidence_items),
        "omitted_count": omitted_count,
        "context_characters": len(context),
    }


__all__ = [
    "build_evidence_context",
]
