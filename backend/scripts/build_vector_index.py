
"""
Build a local SQLite vector database from saved Gemini embeddings.

Features:
1. Reads saved embeddings from JSONL.
2. Validates vector dimensions and numerical values.
3. Converts vectors into compact float32 binary format.
4. Stores vectors and reference metadata in SQLite.
5. Creates indexes for efficient metadata filtering.
6. Supports safe reruns without duplicating chunk IDs.
7. Makes no external API requests.

Input:
data/processed/reference_embeddings.jsonl

Output:
data/processed/reference_vectors.db
"""

import json
import math
import sqlite3
from pathlib import Path

import numpy as np


# --------------------------------------------------
# 1. PROJECT CONFIGURATION
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "reference_embeddings.jsonl"
)

DATABASE_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "reference_vectors.db"
)

EXPECTED_DIMENSIONS = 3072
COMMIT_EVERY = 100


# --------------------------------------------------
# 2. CREATE DATABASE TABLES
# --------------------------------------------------

def create_database(connection: sqlite3.Connection) -> None:
    """Create the reference vector database schema."""

    connection.execute("""
        CREATE TABLE IF NOT EXISTS reference_embeddings (
            chunk_id TEXT PRIMARY KEY,
            dataset TEXT NOT NULL,
            record_id TEXT NOT NULL,
            chunk_index INTEGER NOT NULL,
            text TEXT NOT NULL,
            metadata_json TEXT NOT NULL,
            embedding_model TEXT NOT NULL,
            embedding_dimensions INTEGER NOT NULL,
            embedding_vector BLOB NOT NULL,
            vector_norm REAL NOT NULL
        )
    """)

    connection.execute("""
        CREATE INDEX IF NOT EXISTS idx_reference_dataset
        ON reference_embeddings(dataset)
    """)

    connection.execute("""
        CREATE INDEX IF NOT EXISTS idx_reference_model
        ON reference_embeddings(embedding_model)
    """)

    connection.execute("""
        CREATE TABLE IF NOT EXISTS index_metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )
    """)

    connection.execute(
        "PRAGMA user_version = 1"
    )


# --------------------------------------------------
# 3. VALIDATE AND CONVERT VECTORS
# --------------------------------------------------

def prepare_vector(values: list) -> tuple[bytes, float]:
    """Validate a vector and convert it to float32 bytes."""

    if not isinstance(values, list):
        raise ValueError("Embedding must be a list.")

    if len(values) != EXPECTED_DIMENSIONS:
        raise ValueError(
            f"Expected {EXPECTED_DIMENSIONS} dimensions, "
            f"received {len(values)}."
        )

    if not all(
        isinstance(value, (int, float))
        and math.isfinite(float(value))
        for value in values
    ):
        raise ValueError(
            "Embedding contains invalid numerical values."
        )

    vector = np.asarray(values, dtype="<f4")

    norm = float(np.linalg.norm(vector))

    if not math.isfinite(norm) or norm == 0:
        raise ValueError(
            "Embedding has an invalid or zero vector norm."
        )

    return vector.tobytes(), norm


# --------------------------------------------------
# 4. STORE ONE EMBEDDING
# --------------------------------------------------

def store_embedding(
    connection: sqlite3.Connection,
    record: dict,
) -> None:
    """Insert or update one reference embedding."""

    vector_blob, vector_norm = prepare_vector(
        record.get("embedding", [])
    )

    chunk_id = record.get("chunk_id")

    if not isinstance(chunk_id, str) or not chunk_id:
        raise ValueError("Missing chunk_id.")

    dataset = record.get("dataset")
    record_id = record.get("record_id")
    chunk_index = record.get("chunk_index")
    text = record.get("text")
    metadata = record.get("metadata", {})

    if not isinstance(dataset, str) or not dataset:
        raise ValueError("Missing dataset.")

    if not isinstance(record_id, str) or not record_id:
        raise ValueError("Missing record_id.")

    if not isinstance(chunk_index, int):
        raise ValueError("Invalid chunk_index.")

    if not isinstance(text, str) or not text.strip():
        raise ValueError("Missing reference text.")

    if not isinstance(metadata, dict):
        raise ValueError("Invalid metadata.")

    model = record.get("embedding_model")

    if not isinstance(model, str) or not model:
        raise ValueError("Missing embedding model.")

    if record.get("embedding_dimensions") != EXPECTED_DIMENSIONS:
        raise ValueError("Incorrect embedding_dimensions metadata.")

    metadata_json = json.dumps(
        metadata,
        ensure_ascii=False,
    )

    connection.execute(
        """
        INSERT OR REPLACE INTO reference_embeddings (
            chunk_id,
            dataset,
            record_id,
            chunk_index,
            text,
            metadata_json,
            embedding_model,
            embedding_dimensions,
            embedding_vector,
            vector_norm
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            chunk_id,
            dataset,
            record_id,
            chunk_index,
            text,
            metadata_json,
            model,
            EXPECTED_DIMENSIONS,
            sqlite3.Binary(vector_blob),
            vector_norm,
        ),
    )


# --------------------------------------------------
# 5. BUILD THE VECTOR DATABASE
# --------------------------------------------------

def main() -> int:
    """Build the local SQLite vector database."""

    print("=" * 55)
    print("SQLITE REFERENCE VECTOR DATABASE BUILDER")
    print("=" * 55)

    if not INPUT_FILE.exists():
        print("ERROR: Embedding JSONL file not found.")
        print(INPUT_FILE)
        return 1

    DATABASE_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    connection = None

    try:

        connection = sqlite3.connect(
            DATABASE_FILE,
            timeout=30,
        )

        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("PRAGMA synchronous = NORMAL")
        connection.execute("PRAGMA temp_store = MEMORY")

        create_database(connection)
        connection.commit()

        processed = 0

        print("Input file:", INPUT_FILE)
        print("Database:", DATABASE_FILE)
        print("Expected dimensions:", EXPECTED_DIMENSIONS)
        print()

        with INPUT_FILE.open(
            "r",
            encoding="utf-8",
        ) as file:

            for line_number, line in enumerate(file, start=1):

                if not line.strip():
                    continue

                try:
                    record = json.loads(line)

                    store_embedding(
                        connection,
                        record,
                    )

                    processed += 1

                    if processed % COMMIT_EVERY == 0:
                        connection.commit()

                        print(
                            "Stored records:",
                            processed,
                        )

                except (
                    json.JSONDecodeError,
                    ValueError,
                    TypeError,
                    KeyError,
                    OverflowError,
                ) as error:

                    connection.rollback()

                    raise ValueError(
                        f"Invalid record at line {line_number}: "
                        f"{error}"
                    ) from error

        connection.commit()

        total_records = connection.execute(
            "SELECT COUNT(*) FROM reference_embeddings"
        ).fetchone()[0]

        dataset_counts = connection.execute(
            """
            SELECT dataset, COUNT(*)
            FROM reference_embeddings
            GROUP BY dataset
            ORDER BY dataset
            """
        ).fetchall()

        connection.execute(
            """
            INSERT OR REPLACE INTO index_metadata (key, value)
            VALUES (?, ?)
            """,
            ("embedding_model", "gemini-embedding-2"),
        )

        connection.execute(
            """
            INSERT OR REPLACE INTO index_metadata (key, value)
            VALUES (?, ?)
            """,
            ("embedding_dimensions", str(EXPECTED_DIMENSIONS)),
        )

        connection.commit()

        print()
        print("=" * 55)
        print("VECTOR DATABASE CREATED SUCCESSFULLY!")
        print("=" * 55)

        print("Records processed:", processed)
        print("Total unique database records:", total_records)
        print("Vector dimensions:", EXPECTED_DIMENSIONS)

        print()
        print("Dataset distribution:")

        for dataset, count in dataset_counts:
            print(f"{dataset}: {count}")

        print()
        print("Database saved at:")
        print(DATABASE_FILE)

        return 0

    except (
        sqlite3.Error,
        OSError,
        ValueError,
    ) as error:

        if connection is not None:
            connection.rollback()

        print()
        print("DATABASE BUILD FAILED.")
        print("Reason:", error)
        print("Existing committed records are preserved.")

        return 1

    finally:

        if connection is not None:
            connection.close()


# --------------------------------------------------
# 6. PROGRAM ENTRY POINT
# --------------------------------------------------

if __name__ == "__main__":
    raise SystemExit(main())
