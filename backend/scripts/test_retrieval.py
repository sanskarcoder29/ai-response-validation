
"""
Test the reusable vector retrieval module.

Uses an existing reference embedding as an offline query.
No Gemini API requests are made.

Tests:
1. Import the retrieval module.
2. Load a query vector from SQLite.
3. Retrieve the top 5 similar chunks.
4. Verify ranking and similarity scores.
5. Verify dataset filtering.
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

# Make the backend package available for imports.
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from retrieval.vector_search import search_similar


def main() -> int:
    """Run offline retrieval tests."""

    print("=" * 55)
    print("REUSABLE VECTOR RETRIEVAL TEST")
    print("=" * 55)

    if not DATABASE_FILE.exists():
        print("ERROR: Vector database not found.")
        return 1

    try:

        # Load an existing vector for offline testing.
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
            print("ERROR: No reference vectors found.")
            return 1

        query_chunk_id, vector_blob = row

        query_vector = np.frombuffer(
            vector_blob,
            dtype="<f4",
        ).copy()

        print("Query chunk ID:", query_chunk_id)
        print("Query dimensions:", len(query_vector))
        print()

        # Test 1: Retrieve top 5 results.
        print("TEST 1: Top 5 retrieval")

        results = search_similar(
            query_vector=query_vector,
            top_k=5,
        )

        if len(results) != 5:
            raise AssertionError(
                f"Expected 5 results, received {len(results)}."
            )

        print("Results returned:", len(results))

        for result in results:

            print()
            print("Rank:", result["rank"])
            print("Chunk ID:", result["chunk_id"])
            print("Dataset:", result["dataset"])
            print(
                "Similarity:",
                round(result["similarity_score"], 6),
            )
            print(
                "Reference text:",
                result["text"][:200],
            )

        # Test 2: Confirm the query retrieves itself.
        print()
        print("TEST 2: Exact self-match")

        first_result = results[0]

        if first_result["chunk_id"] != query_chunk_id:
            raise AssertionError(
                "The original query chunk was not ranked first."
            )

        if first_result["similarity_score"] < 0.999:
            raise AssertionError(
                "Self-match similarity is unexpectedly low."
            )

        print("Self-match:", first_result["chunk_id"])
        print(
            "Self-match similarity:",
            round(first_result["similarity_score"], 6),
        )

        # Test 3: Verify dataset filtering.
        print()
        print("TEST 3: TruthfulQA dataset filtering")

        filtered_results = search_similar(
            query_vector=query_vector,
            top_k=5,
            dataset="TruthfulQA",
        )

        if not filtered_results:
            raise AssertionError(
                "Dataset filtering returned no results."
            )

        if any(
            result["dataset"] != "TruthfulQA"
            for result in filtered_results
        ):
            raise AssertionError(
                "Dataset filtering returned an incorrect dataset."
            )

        print("Filtered results:", len(filtered_results))
        print("Dataset filter: TruthfulQA")
        print("Dataset filtering passed.")

        # All tests passed.
        print()
        print("=" * 55)
        print("ALL RETRIEVAL TESTS PASSED!")
        print("=" * 55)
        print("Reusable vector search module is operational.")
        print("No external API requests were made.")

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
        print("RETRIEVAL TEST FAILED.")
        print("Reason:", error)

        return 1


if __name__ == "__main__":
    raise SystemExit(main())
