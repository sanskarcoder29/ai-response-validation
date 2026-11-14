
"""
Test local cosine similarity search using SQLite vectors.

This offline test uses the first stored reference vector
as a sample query. No Gemini API request is made.

The test verifies:
1. SQLite vector loading.
2. NumPy vector reconstruction.
3. Cosine similarity calculation.
4. Top-K reference retrieval.
"""

import sqlite3
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATABASE_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "reference_vectors.db"
)

EXPECTED_DIMENSIONS = 3072
TOP_K = 5


def load_vectors(connection):
    """Load all reference vectors from SQLite."""

    rows = connection.execute(
        """
        SELECT
            chunk_id,
            dataset,
            text,
            embedding_vector,
            vector_norm
        FROM reference_embeddings
        ORDER BY rowid
        """
    ).fetchall()

    if not rows:
        raise ValueError("No reference vectors found.")

    vectors = []
    metadata = []
    stored_norms = []

    for chunk_id, dataset, text, blob, norm in rows:

        vector = np.frombuffer(
            blob,
            dtype="<f4",
        )

        if len(vector) != EXPECTED_DIMENSIONS:
            raise ValueError(
                f"Invalid vector dimensions for {chunk_id}."
            )

        if not np.all(np.isfinite(vector)):
            raise ValueError(
                f"Invalid numerical values in {chunk_id}."
            )

        if not np.isfinite(norm) or norm <= 0:
            raise ValueError(
                f"Invalid stored norm for {chunk_id}."
            )

        vectors.append(vector)
        stored_norms.append(norm)

        metadata.append({
            "chunk_id": chunk_id,
            "dataset": dataset,
            "text": text,
        })

    matrix = np.vstack(vectors)

    norms = np.asarray(
        stored_norms,
        dtype=np.float32,
    )

    return matrix, norms, metadata


def cosine_similarity_search(
    query_vector: np.ndarray,
    matrix: np.ndarray,
    norms: np.ndarray,
    top_k: int = 5,
):
    """Return the top-K most similar reference chunks."""

    query_norm = float(np.linalg.norm(query_vector))

    if not np.isfinite(query_norm) or query_norm == 0:
        raise ValueError("Query vector is invalid.")

    if matrix.shape[1] != len(query_vector):
        raise ValueError("Query and reference dimensions differ.")

    scores = (
        matrix @ query_vector
    ) / (
        norms * query_norm
    )

    scores = np.clip(scores, -1.0, 1.0)

    top_indices = np.argsort(scores)[::-1][:top_k]

    return [
        (int(index), float(scores[index]))
        for index in top_indices
    ]


def main() -> int:
    """Run an offline vector search verification."""

    print("=" * 55)
    print("LOCAL COSINE SIMILARITY SEARCH TEST")
    print("=" * 55)

    if not DATABASE_FILE.exists():
        print("ERROR: SQLite database not found.")
        return 1

    try:

        with sqlite3.connect(DATABASE_FILE) as connection:

            integrity = connection.execute(
                "PRAGMA integrity_check"
            ).fetchone()[0]

            if integrity != "ok":
                print("ERROR: Database integrity check failed.")
                return 1

            matrix, norms, metadata = load_vectors(
                connection
            )

            print("Database integrity:", integrity)
            print("Reference vectors loaded:", len(matrix))
            print("Vector matrix shape:", matrix.shape)

            # Use an existing stored vector for an
            # offline mathematical verification.
            query_vector = matrix[0].copy()

            print("Offline query vector:", metadata[0]["chunk_id"])
            print("Query dimensions:", len(query_vector))

            results = cosine_similarity_search(
                query_vector,
                matrix,
                norms,
                TOP_K,
            )

            print()
            print("TOP", TOP_K, "SIMILAR REFERENCE CHUNKS")
            print("-" * 55)

            for rank, (index, score) in enumerate(
                results,
                start=1,
            ):

                item = metadata[index]

                print()
                print("Rank:", rank)
                print("Chunk ID:", item["chunk_id"])
                print("Dataset:", item["dataset"])
                print("Cosine similarity:", round(score, 6))
                print("Reference text:", item["text"][:250])

            # The query is the first stored vector.
            # It should retrieve itself with similarity
            # approximately equal to 1.0.
            if results[0][0] != 0 or results[0][1] < 0.999:
                print()
                print("WARNING: Self-match validation failed.")
                return 1

            print()
            print("=" * 55)
            print("COSINE SIMILARITY TEST PASSED!")
            print("=" * 55)
            print("Local vector retrieval is operational.")
            print("No external API requests were made.")

            return 0

    except (
        sqlite3.Error,
        ValueError,
        OSError,
    ) as error:

        print("ERROR:", error)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
