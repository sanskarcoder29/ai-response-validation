
"""
Reusable local vector search module.

Features:
1. Loads reference embeddings from SQLite.
2. Accepts a query embedding vector.
3. Calculates cosine similarity using NumPy.
4. Returns the top-K most relevant chunks.
5. Supports dataset filtering.
6. Returns reference metadata and similarity scores.
7. Makes no external API requests.

Database:
data/processed/reference_vectors.db
"""

import json
import math
import sqlite3

from pathlib import Path
from typing import Any

import numpy as np


# --------------------------------------------------
# 1. CONFIGURATION
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_DATABASE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "reference_vectors.db"
)

EXPECTED_DIMENSIONS = 3072


# --------------------------------------------------
# 2. VALIDATE QUERY VECTOR
# --------------------------------------------------

def validate_query_vector(
    query_vector: Any,
) -> tuple[np.ndarray, float]:
    """
    Validate and normalize the query vector.

    Returns:
        A float32 NumPy vector and its norm.
    """

    vector = np.asarray(
        query_vector,
        dtype=np.float32,
    )

    if vector.ndim != 1:
        raise ValueError(
            "Query vector must be one-dimensional."
        )

    if len(vector) != EXPECTED_DIMENSIONS:
        raise ValueError(
            f"Expected {EXPECTED_DIMENSIONS} dimensions, "
            f"received {len(vector)}."
        )

    if not np.all(np.isfinite(vector)):
        raise ValueError(
            "Query vector contains invalid numerical values."
        )

    norm = float(np.linalg.norm(vector))

    if not math.isfinite(norm) or norm == 0:
        raise ValueError(
            "Query vector has an invalid or zero norm."
        )

    return vector, norm


# --------------------------------------------------
# 3. LOAD REFERENCE VECTORS
# --------------------------------------------------

def load_reference_vectors(
    connection: sqlite3.Connection,
    dataset: str | None = None,
) -> list[tuple]:
    """
    Load reference vectors from SQLite.

    An optional dataset filter can restrict retrieval
    to TruthfulQA or SQuAD.
    """

    query = """
        SELECT
            chunk_id,
            dataset,
            record_id,
            chunk_index,
            text,
            metadata_json,
            embedding_vector,
            vector_norm
        FROM reference_embeddings
    """

    parameters = ()

    if dataset is not None:
        query += " WHERE dataset = ?"
        parameters = (dataset,)

    query += " ORDER BY rowid"

    return connection.execute(
        query,
        parameters,
    ).fetchall()


# --------------------------------------------------
# 4. CALCULATE COSINE SIMILARITY
# --------------------------------------------------

def calculate_cosine_similarity(
    query_vector: np.ndarray,
    reference_matrix: np.ndarray,
    reference_norms: np.ndarray,
    query_norm: float,
) -> np.ndarray:
    """
    Calculate cosine similarity between one query
    and all reference vectors.
    """

    dot_products = reference_matrix @ query_vector

    similarities = (
        dot_products
        / (reference_norms * query_norm)
    )

    return np.clip(
        similarities,
        -1.0,
        1.0,
    )


# --------------------------------------------------
# 5. SEARCH REFERENCE KNOWLEDGE BASE
# --------------------------------------------------

def search_similar(
    query_vector: Any,
    top_k: int = 5,
    dataset: str | None = None,
    database_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    """
    Retrieve the most similar reference chunks.

    Args:
        query_vector:
            A 3072-dimensional query embedding.

        top_k:
            Number of results to return.

        dataset:
            Optional dataset filter, for example
            "TruthfulQA" or "SQuAD".

        database_path:
            Optional custom SQLite database location.

    Returns:
        A ranked list of reference records, including
        chunk ID, reference text and similarity score.
    """

    if isinstance(top_k, bool) or not isinstance(top_k, int):
        raise ValueError("top_k must be a positive integer.")

    if top_k <= 0:
        raise ValueError("top_k must be greater than zero.")

    if dataset is not None:
        if not isinstance(dataset, str) or not dataset.strip():
            raise ValueError("Dataset filter must be a non-empty string.")

    db_path = (
        Path(database_path)
        if database_path is not None
        else DEFAULT_DATABASE
    )

    if not db_path.is_file():
        raise FileNotFoundError(
            f"Vector database not found: {db_path}"
        )

    query, query_norm = validate_query_vector(
        query_vector
    )

    connection = None

    try:

        connection = sqlite3.connect(db_path)

        rows = load_reference_vectors(
            connection,
            dataset,
        )

        if not rows:
            return []

        vectors = []
        norms = []
        metadata_records = []

        for row in rows:

            (
                chunk_id,
                dataset_name,
                record_id,
                chunk_index,
                text,
                metadata_json,
                vector_blob,
                vector_norm,
            ) = row

            reference_vector = np.frombuffer(
                vector_blob,
                dtype="<f4",
            )

            if len(reference_vector) != EXPECTED_DIMENSIONS:
                raise ValueError(
                    f"Invalid vector dimensions for {chunk_id}."
                )

            if not np.all(np.isfinite(reference_vector)):
                raise ValueError(
                    f"Invalid vector values for {chunk_id}."
                )

            if (
                not math.isfinite(vector_norm)
                or vector_norm <= 0
            ):
                raise ValueError(
                    f"Invalid stored vector norm for {chunk_id}."
                )

            metadata = json.loads(
                metadata_json
            )

            if not isinstance(metadata, dict):
                raise ValueError(
                    f"Invalid metadata for {chunk_id}."
                )

            vectors.append(reference_vector)
            norms.append(vector_norm)

            metadata_records.append({
                "chunk_id": chunk_id,
                "dataset": dataset_name,
                "record_id": record_id,
                "chunk_index": chunk_index,
                "text": text,
                "metadata": metadata,
            })

        reference_matrix = np.vstack(vectors)

        reference_norms = np.asarray(
            norms,
            dtype=np.float32,
        )

        scores = calculate_cosine_similarity(
            query,
            reference_matrix,
            reference_norms,
            query_norm,
        )

        # Select the highest similarity scores.
        best_indices = np.argsort(scores)[::-1][:top_k]

        results = []

        for rank, index in enumerate(
            best_indices,
            start=1,
        ):

            result = dict(
                metadata_records[int(index)]
            )

            result["rank"] = rank
            result["similarity_score"] = float(
                scores[int(index)]
            )

            results.append(result)

        return results

    finally:

        if connection is not None:
            connection.close()


# --------------------------------------------------
# 6. PUBLIC MODULE EXPORTS
# --------------------------------------------------

__all__ = [
    "search_similar",
    "calculate_cosine_similarity",
    "validate_query_vector",
]
