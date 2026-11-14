
"""
Offline tests for the evidence context builder.

Uses existing SQLite reference embeddings.
No external API requests are made.

Tests:
1. Reference retrieval.
2. Evidence ID generation.
3. Context formatting.
4. Similarity score preservation.
5. Evidence filtering.
6. Context length limits.
"""

import sqlite3
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATABASE_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "reference_vectors.db"
)

sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from retrieval.vector_search import search_similar
from retrieval.context_builder import build_evidence_context


def main() -> int:
    """Run offline context builder tests."""

    print("=" * 55)
    print("EVIDENCE CONTEXT BUILDER TEST")
    print("=" * 55)

    if not DATABASE_FILE.exists():
        print("ERROR: Vector database not found.")
        return 1

    try:

        # ------------------------------------------
        # Load an existing vector for offline testing
        # ------------------------------------------

        with sqlite3.connect(DATABASE_FILE) as connection:

            row = connection.execute(
                """
                SELECT embedding_vector
                FROM reference_embeddings
                ORDER BY rowid
                LIMIT 1
                """
            ).fetchone()

        if row is None:
            raise ValueError("No reference vectors found.")

        query_vector = np.frombuffer(
            row[0],
            dtype="<f4",
        ).copy()

        print("Query vector dimensions:", len(query_vector))

        # ------------------------------------------
        # Test 1: Retrieve reference results
        # ------------------------------------------

        print()
        print("TEST 1: Reference retrieval")

        results = search_similar(
            query_vector=query_vector,
            top_k=5,
            dataset="TruthfulQA",
        )

        if len(results) != 5:
            raise AssertionError(
                f"Expected 5 results, received {len(results)}."
            )

        print("Retrieved references:", len(results))
        print("Reference retrieval: PASSED")

        # ------------------------------------------
        # Test 2: Build evidence context
        # ------------------------------------------

        print()
        print("TEST 2: Evidence context generation")

        output = build_evidence_context(
            results=results,
            max_chars=12000,
        )

        evidence = output["evidence"]
        context = output["context"]

        if len(evidence) != 5:
            raise AssertionError(
                "Expected 5 evidence records."
            )

        if output["included_count"] != 5:
            raise AssertionError(
                "Incorrect included evidence count."
            )

        print("Evidence records:", output["included_count"])
        print("Context characters:", output["context_characters"])
        print("Context generation: PASSED")

        # ------------------------------------------
        # Test 3: Evidence ID generation
        # ------------------------------------------

        print()
        print("TEST 3: Evidence ID generation")

        expected_ids = [
            "E1", "E2", "E3", "E4", "E5"
        ]

        actual_ids = [
            item["evidence_id"]
            for item in evidence
        ]

        if actual_ids != expected_ids:
            raise AssertionError(
                f"Incorrect evidence IDs: {actual_ids}"
            )

        print("Evidence IDs:", actual_ids)
        print("Evidence ID generation: PASSED")

        # ------------------------------------------
        # Test 4: Metadata preservation
        # ------------------------------------------

        print()
        print("TEST 4: Metadata preservation")

        for original, item in zip(results, evidence):

            if item["chunk_id"] != original["chunk_id"]:
                raise AssertionError("Chunk ID mismatch.")

            if item["dataset"] != original["dataset"]:
                raise AssertionError("Dataset mismatch.")

            if (
                abs(
                    item["similarity_score"]
                    - original["similarity_score"]
                ) > 1e-9
            ):
                raise AssertionError("Similarity score mismatch.")

        print("Chunk IDs preserved.")
        print("Dataset names preserved.")
        print("Similarity scores preserved.")
        print("Metadata preservation: PASSED")

        # ------------------------------------------
        # Test 5: Context formatting
        # ------------------------------------------

        print()
        print("TEST 5: Context formatting")

        if "[E1]" not in context:
            raise AssertionError("E1 missing from context.")

        if "[E5]" not in context:
            raise AssertionError("E5 missing from context.")

        if "Dataset: TruthfulQA" not in context:
            raise AssertionError("Dataset missing from context.")

        print("Evidence labels found.")
        print("Dataset information found.")
        print("Context formatting: PASSED")

        # ------------------------------------------
        # Test 6: Similarity filtering
        # ------------------------------------------

        print()
        print("TEST 6: Similarity threshold filtering")

        filtered = build_evidence_context(
            results=results,
            max_chars=12000,
            min_similarity=0.999,
        )

        if filtered["included_count"] != 1:
            raise AssertionError(
                "Expected only the exact self-match "
                "at a similarity threshold of 0.999."
            )

        if filtered["evidence"][0]["evidence_id"] != "E1":
            raise AssertionError(
                "Filtered evidence ID is incorrect."
            )

        print("Filtered evidence:", filtered["included_count"])
        print("Threshold filtering: PASSED")

        # ------------------------------------------
        # Test 7: Context length limit
        # ------------------------------------------

        print()
        print("TEST 7: Context length limit")

        limited = build_evidence_context(
            results=results,
            max_chars=1,
        )

        if limited["context_characters"] > 1:
            raise AssertionError(
                "Context length limit was exceeded."
            )

        if limited["included_count"] != 0:
            raise AssertionError(
                "Expected no records to fit in one character."
            )

        print("Context length:", limited["context_characters"])
        print("Omitted references:", limited["omitted_count"])
        print("Context length validation: PASSED")

        # ------------------------------------------
        # Final result
        # ------------------------------------------

        print()
        print("=" * 55)
        print("ALL CONTEXT BUILDER TESTS PASSED!")
        print("=" * 55)
        print("Evidence context builder is operational.")
        print("Gemini API requests: 0")

        return 0

    except (
        sqlite3.Error,
        OSError,
        ValueError,
        AssertionError,
        TypeError,
        KeyError,
    ) as error:

        print()
        print("CONTEXT BUILDER TEST FAILED.")
        print("Reason:", error)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
