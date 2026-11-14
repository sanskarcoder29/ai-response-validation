
"""
Offline retrieval pipeline integration test.

Uses a stored reference embedding as a mocked query.
No Gemini API requests are made.
"""

import sqlite3
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np


# Correct project root:
# test file is located in backend/scripts/
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATABASE_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "reference_vectors.db"
)

# Add the actual backend directory to Python's import path.
BACKEND_DIRECTORY = PROJECT_ROOT / "backend"

sys.path.insert(0, str(BACKEND_DIRECTORY))

from retrieval.retriever import retrieve_for_question


def main() -> int:

    print("=" * 55)
    print("OFFLINE RETRIEVAL PIPELINE INTEGRATION TEST")
    print("=" * 55)

    if not DATABASE_FILE.exists():
        print("ERROR: Vector database not found.")
        print(DATABASE_FILE)
        return 1

    try:

        # Load an existing vector from SQLite.
        with sqlite3.connect(DATABASE_FILE) as connection:

            row = connection.execute(
                """
                SELECT chunk_id, embedding_vector
                FROM reference_embeddings
                ORDER BY rowid
                LIMIT 1
                """
            ).fetchone()

        if row is None:
            print("ERROR: No reference vectors available.")
            return 1

        expected_chunk_id, vector_blob = row

        mock_vector = np.frombuffer(
            vector_blob,
            dtype="<f4",
        ).copy().tolist()

        question = (
            "What happens if you swallow chewing gum?"
        )

        print("Test question:", question)
        print("Mock embedding dimensions:", len(mock_vector))
        print()

        # Mock Gemini embedding generation.
        with patch(
            "retrieval.retriever.embed_query",
            return_value=mock_vector,
        ) as mocked_embed:

            results = retrieve_for_question(
                question=question,
                top_k=5,
                dataset="TruthfulQA",
            )

            mocked_embed.assert_called_once_with(question)

        # Test 1: Retrieval pipeline.
        print("TEST 1: Retrieval pipeline")

        if len(results) != 5:
            raise AssertionError(
                f"Expected 5 results, got {len(results)}."
            )

        print("Retrieved results:", len(results))
        print("Pipeline execution: PASSED")

        # Test 2: Exact reference matching.
        print()
        print("TEST 2: Reference matching")

        if results[0]["chunk_id"] != expected_chunk_id:
            raise AssertionError(
                "Expected reference chunk was not ranked first."
            )

        if results[0]["similarity_score"] < 0.999:
            raise AssertionError(
                "Self-match similarity is unexpectedly low."
            )

        print("Expected chunk:", expected_chunk_id)
        print("Retrieved chunk:", results[0]["chunk_id"])
        print(
            "Similarity:",
            round(results[0]["similarity_score"], 6),
        )

        # Test 3: Similarity ranking.
        print()
        print("TEST 3: Similarity ranking")

        scores = [
            result["similarity_score"]
            for result in results
        ]

        if any(
            scores[i] < scores[i + 1]
            for i in range(len(scores) - 1)
        ):
            raise AssertionError(
                "Results are not correctly ranked."
            )

        print("Similarity ranking: PASSED")

        # Test 4: Dataset filtering.
        print()
        print("TEST 4: Dataset filtering")

        if any(
            result["dataset"] != "TruthfulQA"
            for result in results
        ):
            raise AssertionError(
                "Incorrect dataset returned."
            )

        print("Dataset filtering: PASSED")

        print()
        print("=" * 55)
        print("ALL PIPELINE INTEGRATION TESTS PASSED!")
        print("=" * 55)
        print("Gemini API requests: 0")

        return 0

    except (
        sqlite3.Error,
        OSError,
        ValueError,
        AssertionError,
        TypeError,
        ImportError,
    ) as error:

        print()
        print("INTEGRATION TEST FAILED.")
        print("Reason:", error)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
