
"""
Gemini Embedding 2 query embedding module.

Creates query vectors compatible with our local
question-answering reference retrieval system.

No API request is made when this module is imported.
"""

import math
import os
from pathlib import Path

import requests
from dotenv import load_dotenv


# --------------------------------------------------
# 1. CONFIGURATION
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL = "gemini-embedding-2"

API_URL = (
    "https://generativelanguage.googleapis.com/v1beta/"
    f"models/{MODEL}:embedContent"
)

EXPECTED_DIMENSIONS = 3072
REQUEST_TIMEOUT = 60

TASK_PREFIX = "task: question answering | query: "


# --------------------------------------------------
# 2. LOAD API KEY
# --------------------------------------------------

def load_api_key() -> str:
    """Load Gemini API key from the project .env file."""

    load_dotenv(PROJECT_ROOT / ".env")

    api_key = os.getenv(
        "GEMINI_API_KEY",
        "",
    ).strip()

    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY was not found in .env."
        )

    return api_key


# --------------------------------------------------
# 3. VALIDATE QUERY
# --------------------------------------------------

def validate_query(query: str) -> str:
    """Validate and normalize user question."""

    if not isinstance(query, str):
        raise TypeError("Query must be a string.")

    query = query.strip()

    if not query:
        raise ValueError("Query cannot be empty.")

    return query


# --------------------------------------------------
# 4. PREPARE QUERY PROMPT
# --------------------------------------------------

def prepare_query(query: str) -> str:
    """
    Apply the Gemini Embedding 2 task instruction.

    Question-answering retrieval format:
    task: question answering | query: {question}
    """

    query = validate_query(query)

    return f"{TASK_PREFIX}{query}"


# --------------------------------------------------
# 5. GENERATE QUERY EMBEDDING
# --------------------------------------------------

def embed_query(query: str) -> list[float]:
    """Generate a 3072-dimensional query embedding."""

    api_key = load_api_key()

    prepared_query = prepare_query(query)

    payload = {
        "content": {
            "parts": [
                {"text": prepared_query}
            ]
        }
    }

    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": api_key,
    }

    try:

        response = requests.post(
            API_URL,
            headers=headers,
            json=payload,
            timeout=REQUEST_TIMEOUT,
        )

        if response.status_code == 429:
            raise RuntimeError(
                "Gemini quota or rate limit reached. "
                "Check Google AI Studio before retrying."
            )

        if response.status_code != 200:

            try:
                error_data = response.json()

                message = (
                    error_data.get("error", {}).get(
                        "message",
                        "Unknown API error.",
                    )
                )

            except ValueError:
                message = response.text[:500]

            raise RuntimeError(
                f"Gemini HTTP {response.status_code}: {message}"
            )

        result = response.json()

        embedding = result.get("embedding", {})
        values = embedding.get("values", [])

        if not isinstance(values, list):
            raise ValueError("Invalid embedding response.")

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

        return values

    except requests.RequestException as error:

        raise RuntimeError(
            f"Network error: {error}"
        ) from error


# --------------------------------------------------
# 6. PUBLIC MODULE EXPORTS
# --------------------------------------------------

__all__ = [
    "embed_query",
    "validate_query",
    "prepare_query",
]
