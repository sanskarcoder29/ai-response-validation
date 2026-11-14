
"""
Generate Gemini Embedding 2 vectors for reference chunks.

Features:
1. Processes up to 50 new chunks per execution.
2. Saves each successful embedding immediately.
3. Skips already completed chunk IDs.
4. Resumes safely after interruptions.
5. Validates embedding dimensions and numerical values.
6. Handles API errors without automatic repeated requests.
7. Keeps the Gemini API key private.
8. Processes both TruthfulQA and SQuAD from the
   combined reference chunks file.

Input:
data/processed/reference_chunks.json

Output:
data/processed/reference_embeddings.jsonl
"""

import json
import math
import os
import time
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv


# --------------------------------------------------
# 1. PROJECT CONFIGURATION
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "reference_chunks.json"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "reference_embeddings.jsonl"
)

MODEL = "gemini-embedding-2"

API_URL = (
    "https://generativelanguage.googleapis.com/v1beta/"
    f"models/{MODEL}:embedContent"
)


# --------------------------------------------------
# 2. EMBEDDING SETTINGS
# --------------------------------------------------

# Conservative batch size to control API usage.
MAX_NEW_EMBEDDINGS = 50

EXPECTED_DIMENSIONS = 3072

# Delay between consecutive API requests.
REQUEST_DELAY_SECONDS = 1.0

REQUEST_TIMEOUT = 60


# --------------------------------------------------
# 3. LOAD COMPLETED EMBEDDINGS
# --------------------------------------------------

def load_completed_ids() -> set[str]:
    """
    Read the JSONL output file and identify successfully
    saved embeddings.

    Existing valid records are never regenerated.
    """

    completed_ids: set[str] = set()

    if not OUTPUT_FILE.exists():
        return completed_ids

    with OUTPUT_FILE.open(
        "r",
        encoding="utf-8",
    ) as file:

        for line_number, line in enumerate(file, start=1):

            if not line.strip():
                continue

            try:
                record = json.loads(line)

                chunk_id = record.get("chunk_id")
                embedding = record.get("embedding")

                if (
                    isinstance(chunk_id, str)
                    and chunk_id
                    and isinstance(embedding, list)
                    and len(embedding) == EXPECTED_DIMENSIONS
                    and all(
                        isinstance(value, (int, float))
                        and not isinstance(value, bool)
                        and math.isfinite(float(value))
                        for value in embedding
                    )
                ):
                    completed_ids.add(chunk_id)

            except (
                json.JSONDecodeError,
                TypeError,
                ValueError,
            ):
                print(
                    "Warning: Invalid saved record at line:",
                    line_number,
                )

    return completed_ids


# --------------------------------------------------
# 4. PREPARE DOCUMENT TEXT
# --------------------------------------------------

def prepare_document(
    chunk: dict[str, Any],
) -> str:
    """
    Format a reference chunk for Gemini Embedding 2.

    Preserves the reference title and text.
    """

    metadata = chunk.get("metadata", {})

    title = metadata.get("title") or "none"
    text = chunk.get("text", "")

    return f"title: {title} | text: {text}"


# --------------------------------------------------
# 5. GENERATE ONE EMBEDDING
# --------------------------------------------------

def generate_embedding(
    session: requests.Session,
    api_key: str,
    document_text: str,
) -> list[float]:
    """
    Send one document to Gemini and retrieve its vector.

    No automatic retry is performed. This prevents
    unexpected repeated requests if an API limit occurs.
    """

    payload = {
        "content": {
            "parts": [
                {
                    "text": document_text,
                }
            ]
        }
    }

    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": api_key,
    }

    response = session.post(
        API_URL,
        headers=headers,
        json=payload,
        timeout=REQUEST_TIMEOUT,
    )

    if response.status_code == 429:
        raise RuntimeError(
            "Gemini quota or rate limit reached. "
            "Stop processing and check Google AI Studio "
            "before making more requests. "
            "Previously saved embeddings are safe."
        )

    if response.status_code != 200:

        try:
            error_data = response.json()

            error_message = (
                error_data.get("error", {}).get(
                    "message",
                    "Unknown API error.",
                )
            )

        except ValueError:
            error_message = response.text[:500]

        raise RuntimeError(
            f"Gemini HTTP {response.status_code}: "
            f"{error_message}"
        )

    result = response.json()

    embedding = result.get("embedding", {})
    values = embedding.get("values", [])

    if not isinstance(values, list):
        raise ValueError(
            "Invalid embedding response."
        )

    if len(values) != EXPECTED_DIMENSIONS:
        raise ValueError(
            f"Expected {EXPECTED_DIMENSIONS} dimensions, "
            f"received {len(values)}."
        )

    if not all(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        for value in values
    ):
        raise ValueError(
            "Embedding contains invalid numerical values."
        )

    return values


# --------------------------------------------------
# 6. SAVE EMBEDDING IMMEDIATELY
# --------------------------------------------------

def save_embedding(
    chunk: dict[str, Any],
    values: list[float],
) -> None:
    """
    Append one completed embedding to the JSONL file.

    Each record is saved separately so completed work
    survives interruptions.
    """

    record = {
        "chunk_id": chunk["chunk_id"],
        "dataset": chunk["dataset"],
        "record_id": chunk["record_id"],
        "chunk_index": chunk["chunk_index"],
        "text": chunk["text"],
        "metadata": chunk["metadata"],
        "embedding_model": MODEL,
        "embedding_dimensions": len(values),
        "embedding": values,
    }

    with OUTPUT_FILE.open(
        "a",
        encoding="utf-8",
    ) as file:

        file.write(
            json.dumps(
                record,
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n"
        )

        file.flush()
        os.fsync(file.fileno())


# --------------------------------------------------
# 7. MAIN EXECUTION
# --------------------------------------------------

def main() -> int:
    """Generate, validate and save reference embeddings."""

    print("=" * 55)
    print("GEMINI REFERENCE EMBEDDING GENERATION")
    print("=" * 55)

    # Load environment variables securely.
    load_dotenv(PROJECT_ROOT / ".env")

    api_key = os.getenv(
        "GEMINI_API_KEY",
        "",
    ).strip()

    if not api_key:
        print(
            "ERROR: GEMINI_API_KEY not found in .env."
        )
        return 1

    # Check input dataset.
    if not INPUT_FILE.exists():
        print(
            "ERROR: Reference chunks file not found."
        )
        print(INPUT_FILE)
        return 1

    try:

        with INPUT_FILE.open(
            "r",
            encoding="utf-8",
        ) as file:

            data = json.load(file)

        chunks = data.get("chunks", [])

        if not isinstance(chunks, list) or not chunks:
            print(
                "ERROR: No reference chunks found."
            )
            return 1

        completed_ids = load_completed_ids()

        # Select chunks that are not already embedded.
        pending_chunks = [
            chunk
            for chunk in chunks
            if (
                isinstance(chunk, dict)
                and isinstance(chunk.get("chunk_id"), str)
                and chunk["chunk_id"]
                and chunk["chunk_id"] not in completed_ids
                and isinstance(chunk.get("text"), str)
                and chunk.get("text", "").strip()
            )
        ]

        # Select at most 50 new embeddings.
        selected_chunks = pending_chunks[
            :MAX_NEW_EMBEDDINGS
        ]

        OUTPUT_FILE.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        print("Model:", MODEL)
        print("Total reference chunks:", len(chunks))
        print("Already completed:", len(completed_ids))
        print("Remaining chunks:", len(pending_chunks))
        print(
            "Maximum new embeddings this run:",
            MAX_NEW_EMBEDDINGS,
        )
        print(
            "Selected for this run:",
            len(selected_chunks),
        )
        print()

        if not selected_chunks:
            print("No new embeddings to generate.")
            return 0

        successful = 0

        # Use one reusable HTTP session.
        with requests.Session() as session:

            for index, chunk in enumerate(
                selected_chunks,
                start=1,
            ):

                chunk_id = chunk["chunk_id"]

                print(
                    f"Processing chunk {index} "
                    f"of {len(selected_chunks)}..."
                )

                try:

                    document_text = prepare_document(
                        chunk
                    )

                    values = generate_embedding(
                        session,
                        api_key,
                        document_text,
                    )

                    save_embedding(
                        chunk,
                        values,
                    )

                    completed_ids.add(chunk_id)
                    successful += 1

                    print(
                        "Saved:",
                        chunk_id,
                        "| Dimensions:",
                        len(values),
                    )

                except (
                    requests.RequestException,
                    RuntimeError,
                    ValueError,
                    KeyError,
                    OSError,
                ) as error:

                    print()
                    print(
                        "EMBEDDING GENERATION STOPPED."
                    )
                    print("Reason:", error)

                    print(
                        "Successfully saved in this run:",
                        successful,
                    )

                    print(
                        "Total valid saved embeddings:",
                        len(completed_ids),
                    )

                    print(
                        "Previously saved embeddings are preserved."
                    )

                    return 1

                # Space requests apart.
                if index < len(selected_chunks):

                    time.sleep(
                        REQUEST_DELAY_SECONDS
                    )

        print()
        print("=" * 55)
        print("EMBEDDING GENERATION COMPLETED")
        print("=" * 55)

        print(
            "New embeddings generated:",
            successful,
        )

        print(
            "Total saved embeddings:",
            len(completed_ids),
        )

        print(
            "Remaining chunks:",
            len(pending_chunks) - successful,
        )

        print()
        print("Output file:")
        print(OUTPUT_FILE)

        return 0

    except (
        OSError,
        json.JSONDecodeError,
        ValueError,
    ) as error:

        print("ERROR:", error)
        return 1


# --------------------------------------------------
# 8. PROGRAM ENTRY POINT
# --------------------------------------------------

if __name__ == "__main__":
    raise SystemExit(main())
